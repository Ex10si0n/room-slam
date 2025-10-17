import torch
import json
from pathlib import Path
import argparse
from model import build_model


def load_model(checkpoint_path: str, device):
    """Load trained model"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    model = build_model(
        num_queries=config.get('num_queries', 50),
        d_model=config.get('d_model', 256)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def process_traces(traces):
    """Convert trace list to tensor"""
    import numpy as np

    trace_array = np.array([
        [p['x'], p['y'], p['z'], p['timestamp']]
        for p in traces
    ], dtype=np.float32)

    # Normalize timestamp
    if trace_array.shape[0] > 0:
        trace_array[:, 3] -= trace_array[:, 3].min()

    # Downsample if too long (to avoid memory issues)
    max_len = 3000
    if len(trace_array) > max_len:
        print(f"Downsampling traces from {len(trace_array)} to {max_len} points")
        indices = np.linspace(0, len(trace_array) - 1, max_len, dtype=int)
        trace_array = trace_array[indices]

    return torch.from_numpy(trace_array)


def compute_iou_3d(box1, box2):
    """
    Compute 3D IoU between two boxes.
    box: [cx, cy, cz, sx, sy, sz]
    """
    # Convert to corner format
    box1_min = box1[:3] - box1[3:] / 2
    box1_max = box1[:3] + box1[3:] / 2
    box2_min = box2[:3] - box2[3:] / 2
    box2_max = box2[:3] + box2[3:] / 2

    # Compute intersection
    inter_min = torch.maximum(box1_min, box2_min)
    inter_max = torch.minimum(box1_max, box2_max)
    inter_size = torch.clamp(inter_max - inter_min, min=0)
    inter_volume = inter_size.prod()

    # Compute union
    box1_volume = box1[3:].prod()
    box2_volume = box2[3:].prod()
    union_volume = box1_volume + box2_volume - inter_volume

    # IoU
    iou = inter_volume / (union_volume + 1e-6)
    return iou.item()


def nms_3d(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression for 3D boxes.

    Args:
        boxes: [N, 6] tensor
        scores: [N] tensor
        iou_threshold: IoU threshold for suppression

    Returns:
        keep_indices: List of indices to keep
    """
    if len(boxes) == 0:
        return []

    # Sort by scores
    sorted_indices = torch.argsort(scores, descending=True)

    keep = []
    while len(sorted_indices) > 0:
        # Keep the highest scoring box
        current = sorted_indices[0].item()
        keep.append(current)

        if len(sorted_indices) == 1:
            break

        # Compute IoU with remaining boxes
        current_box = boxes[current]
        remaining_indices = sorted_indices[1:]

        # Filter out boxes with high IoU
        new_remaining = []
        for idx in remaining_indices:
            iou = compute_iou_3d(current_box, boxes[idx])
            if iou < iou_threshold:
                new_remaining.append(idx)

        sorted_indices = torch.tensor(new_remaining, dtype=torch.long)

    return keep


def post_process_predictions(boxes, classes, confidence_threshold=0.7, nms_threshold=0.3):
    """Filter and format predictions with NMS"""
    # boxes: [Q, 6] (cx, cy, cz, sx, sy, sz)
    # classes: [Q, 4] (logits)

    label_map = {0: 'BLOCK', 1: 'LOW', 2: 'MID', 3: 'HIGH'}

    # Get class probabilities
    probs = torch.softmax(classes, dim=-1)
    max_probs, pred_labels = probs.max(dim=-1)

    # Filter by confidence
    valid_mask = max_probs > confidence_threshold
    valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)

    if len(valid_indices) == 0:
        return []

    # Get valid predictions
    valid_boxes = boxes[valid_indices]
    valid_scores = max_probs[valid_indices]
    valid_labels = pred_labels[valid_indices]

    # Apply NMS per class
    final_indices = []
    for label_id in range(4):
        class_mask = valid_labels == label_id
        class_indices = class_mask.nonzero(as_tuple=False).squeeze(-1)

        if len(class_indices) == 0:
            continue

        class_boxes = valid_boxes[class_indices]
        class_scores = valid_scores[class_indices]

        # NMS for this class (returns indices into class_boxes)
        keep_in_class = nms_3d(class_boxes, class_scores, nms_threshold)

        # Map back to valid_boxes indices
        for k in keep_in_class:
            final_indices.append(class_indices[k].item())

    # Format predictions
    predictions = []
    for idx in final_indices:
        box = valid_boxes[idx].cpu().numpy()
        label = label_map[valid_labels[idx].item()]
        conf = valid_scores[idx].item()

        predictions.append({
            'type': 'BoxCollider',
            'label': label,
            'confidence': float(conf),
            'center': {
                'x': float(box[0]),
                'y': float(box[1]),
                'z': float(box[2])
            },
            'size': {
                'x': float(box[3]),
                'y': float(box[4]),
                'z': float(box[5])
            },
            'radius': 0.0,
            'height': 0.0
        })

    return predictions


def predict(model, traces_file, device, confidence_threshold=0.7, nms_threshold=0.3):
    """Run prediction on a trace file"""

    # Load traces
    with open(traces_file, 'r') as f:
        data = json.load(f)

    # Handle both formats:
    # 1. Direct list: [{x, y, z, timestamp}, ...]
    # 2. Dict with traces: {"traces": [...]}
    if isinstance(data, list):
        traces = data
    else:
        traces = data.get('traces', data.get('trajectory', []))

    if len(traces) == 0:
        print("Warning: No traces found in file")
        return []

    # Process traces
    trace_tensor = process_traces(traces).unsqueeze(0).to(device)  # [1, N, 4]

    # Run inference
    with torch.no_grad():
        outputs = model(trace_tensor)

    # Post-process with NMS
    pred_boxes = outputs['pred_boxes'][0]  # [Q, 6]
    pred_classes = outputs['pred_classes'][0]  # [Q, 4]

    predictions = post_process_predictions(
        pred_boxes, pred_classes, confidence_threshold, nms_threshold
    )

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input trace file (JSON)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for predictions')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Confidence threshold (default: 0.7)')
    parser.add_argument('--nms', type=float, default=0.3,
                        help='NMS IoU threshold (default: 0.3)')
    args = parser.parse_args()

    # Setup device - use CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"CUDA not available, using CPU")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # Run prediction
    print(f"Processing {args.input}")
    predictions = predict(model, args.input, device, args.threshold, args.nms)

    print(f"\nFound {len(predictions)} colliders:")
    for i, pred in enumerate(predictions):
        print(f"  {i + 1}. {pred['label']} at "
              f"({pred['center']['x']:.2f}, {pred['center']['y']:.2f}, {pred['center']['z']:.2f}) "
              f"- confidence: {pred['confidence']:.3f}")

    # Save results
    if args.output:
        output_data = {
            'colliders': predictions,
            'metadata': {
                'num_colliders': len(predictions),
                'confidence_threshold': args.threshold,
                'nms_threshold': args.nms
            }
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()