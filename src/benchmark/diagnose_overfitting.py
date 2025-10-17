"""
Diagnostic script to check if model is actually using trace information
or just memorizing fixed outputs.
"""

import torch
import numpy as np
from model import build_model
import json
import sys


def load_model(checkpoint_path, device):
    """Load trained model"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    model = build_model(
        num_queries=config.get('num_queries', 30),
        d_model=config.get('d_model', 128)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def test_trace_sensitivity(model, device):
    """Test if model predictions change with different traces"""

    print("\n" + "=" * 60)
    print("Test 1: Trace Sensitivity")
    print("=" * 60)

    # Create 3 very different synthetic traces
    traces = [
        torch.randn(1, 1000, 4).to(device) * 2.0,  # Random trace 1
        torch.randn(1, 1000, 4).to(device) * 2.0,  # Random trace 2
        torch.zeros(1, 1000, 4).to(device),  # All zeros
    ]

    outputs = []
    with torch.no_grad():
        for i, trace in enumerate(traces):
            output = model(trace)
            outputs.append(output['pred_boxes'][0])  # [Q, 6]
            print(f"\nTrace {i + 1} predictions (first 3 boxes):")
            print(outputs[i][:3])

    # Compare predictions
    diff_1_2 = (outputs[0] - outputs[1]).abs().mean().item()
    diff_1_3 = (outputs[0] - outputs[2]).abs().mean().item()
    diff_2_3 = (outputs[1] - outputs[2]).abs().mean().item()

    print(f"\nPrediction differences:")
    print(f"  Random1 vs Random2: {diff_1_2:.6f}")
    print(f"  Random1 vs Zeros:   {diff_1_3:.6f}")
    print(f"  Random2 vs Zeros:   {diff_2_3:.6f}")

    threshold = 0.1
    if diff_1_2 < threshold and diff_1_3 < threshold:
        print(f"\n❌ PROBLEM: Predictions are too similar (< {threshold})")
        print("   Model is NOT using trace information!")
        return False
    else:
        print(f"\n✅ OK: Predictions differ significantly")
        print("   Model responds to different traces")
        return True


def test_trace_shuffling(model, device, trace_file):
    """Test if model is sensitive to trace order"""

    print("\n" + "=" * 60)
    print("Test 2: Trace Order Sensitivity")
    print("=" * 60)

    # Load real trace
    with open(trace_file, 'r') as f:
        trace_data = json.load(f)

    if isinstance(trace_data, list):
        traces = trace_data
    else:
        traces = trace_data.get('traces', [])

    # Convert to tensor
    trace_array = np.array([
        [p['x'], p['y'], p['z'], p['timestamp']]
        for p in traces
    ], dtype=np.float32)

    # Downsample
    if len(trace_array) > 1000:
        indices = np.linspace(0, len(trace_array) - 1, 1000, dtype=int)
        trace_array = trace_array[indices]

    trace_original = torch.from_numpy(trace_array).unsqueeze(0).to(device)

    # Shuffle trace
    shuffled_indices = torch.randperm(trace_original.size(1))
    trace_shuffled = trace_original[:, shuffled_indices, :]

    with torch.no_grad():
        output_orig = model(trace_original)
        output_shuf = model(trace_shuffled)

    diff = (output_orig['pred_boxes'] - output_shuf['pred_boxes']).abs().mean().item()

    print(f"\nPrediction difference (original vs shuffled): {diff:.6f}")

    if diff < 0.1:
        print("\n❌ PROBLEM: Shuffling trace doesn't change predictions")
        print("   Model is ignoring temporal/spatial order!")
        return False
    else:
        print("\n✅ OK: Predictions change when trace is shuffled")
        return True


def test_trace_scaling(model, device, trace_file):
    """Test if model responds to trace scaling"""

    print("\n" + "=" * 60)
    print("Test 3: Trace Scaling Sensitivity")
    print("=" * 60)

    # Load real trace
    with open(trace_file, 'r') as f:
        trace_data = json.load(f)

    if isinstance(trace_data, list):
        traces = trace_data
    else:
        traces = trace_data.get('traces', [])

    trace_array = np.array([
        [p['x'], p['y'], p['z'], p['timestamp']]
        for p in traces
    ], dtype=np.float32)

    if len(trace_array) > 1000:
        indices = np.linspace(0, len(trace_array) - 1, 1000, dtype=int)
        trace_array = trace_array[indices]

    # Create scaled versions
    trace_1x = torch.from_numpy(trace_array).unsqueeze(0).to(device)
    trace_2x = trace_1x.clone()
    trace_2x[:, :, :3] *= 2.0  # Scale spatial coordinates by 2x

    trace_05x = trace_1x.clone()
    trace_05x[:, :, :3] *= 0.5  # Scale by 0.5x

    with torch.no_grad():
        output_1x = model(trace_1x)
        output_2x = model(trace_2x)
        output_05x = model(trace_05x)

    # Check if box predictions scale accordingly
    boxes_1x = output_1x['pred_boxes'][0].mean(dim=0)[:3]  # Average center
    boxes_2x = output_2x['pred_boxes'][0].mean(dim=0)[:3]
    boxes_05x = output_05x['pred_boxes'][0].mean(dim=0)[:3]

    print(f"\nAverage box centers:")
    print(f"  1x scale: {boxes_1x.cpu().numpy()}")
    print(f"  2x scale: {boxes_2x.cpu().numpy()}")
    print(f"  0.5x scale: {boxes_05x.cpu().numpy()}")

    # Ideally: boxes_2x ≈ 2 * boxes_1x
    ratio_2x = (boxes_2x / (boxes_1x + 1e-6)).cpu().numpy()
    ratio_05x = (boxes_05x / (boxes_1x + 1e-6)).cpu().numpy()

    print(f"\nScaling ratios:")
    print(f"  2x trace → box scale ratio: {ratio_2x}")
    print(f"  0.5x trace → box scale ratio: {ratio_05x}")

    # Check if roughly proportional
    if np.abs(ratio_2x - 2.0).mean() < 0.5:
        print("\n✅ OK: Boxes scale with trace")
        return True
    else:
        print("\n❌ PROBLEM: Boxes don't scale with trace")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_overfitting.py <checkpoint_path> [trace_file]")
        print(
            "Example: python diagnose_overfitting.py ./checkpoints/best_model.pth ../../dataset/human_data_20251015_181004.json")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    trace_file = sys.argv[2] if len(sys.argv) > 2 else "../../dataset/human_data_20251015_181004.json"

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, device)

    # Run tests
    results = []

    try:
        results.append(("Trace Sensitivity", test_trace_sensitivity(model, device)))
    except Exception as e:
        print(f"\nTest 1 failed: {e}")
        results.append(("Trace Sensitivity", False))

    try:
        results.append(("Trace Order", test_trace_shuffling(model, device, trace_file)))
    except Exception as e:
        print(f"\nTest 2 failed: {e}")
        results.append(("Trace Order", False))

    try:
        results.append(("Trace Scaling", test_trace_scaling(model, device, trace_file)))
    except Exception as e:
        print(f"\nTest 3 failed: {e}")
        results.append(("Trace Scaling", False))

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\nPassed: {passed_count}/{total_count}")

    if passed_count == 0:
        print("\n⚠️  SEVERE OVERFITTING DETECTED!")
        print("    Model is NOT learning from traces.")
        print("\n💡 Recommended fixes:")
        print("    1. Add more aggressive data augmentation")
        print("    2. Add dropout and regularization")
        print("    3. Reduce model capacity")
        print("    4. Collect data from different rooms")
    elif passed_count < total_count:
        print("\n⚠️  PARTIAL OVERFITTING")
        print("    Model uses traces but may still memorize patterns.")
    else:
        print("\n✅ Model appears to be learning from traces!")


if __name__ == "__main__":
    main()