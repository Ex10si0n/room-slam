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
        d_model=config.get('d_model', 256),
        model_type=config.get('model_type', 'transformer')
    )

    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval()
    return model, config


def process_traces(traces, max_len: int = 3000):
    """Convert trace list to a [N,11] tensor:
       [x,y,z,t, vx,vy,vz, ax,ay,az, speed]
    """
    import numpy as np

    # raw array [N,4]
    arr = np.array([[p['x'], p['y'], p['z'], p['timestamp']] for p in traces], dtype=np.float32)

    if arr.shape[0] == 0:
        return torch.zeros((1, 11), dtype=torch.float32)

    # sort by time & normalize time to start at 0
    order = np.argsort(arr[:, 3])
    arr = arr[order]
    arr[:, 3] -= arr[0, 3]

    # kinematic features (order-sensitive)
    diffs = np.diff(arr, axis=0, prepend=arr[[0], :])
    dt = np.clip(diffs[:, 3], 1e-3, None)
    vel = diffs[:, :3] / dt[:, None]                    # [N,3]
    acc = np.diff(vel, axis=0, prepend=vel[[0], :])     # [N,3]
    speed = np.linalg.norm(vel, axis=1, keepdims=True)  # [N,1]
    kin = np.concatenate([vel, acc, speed], axis=1)     # [N,7]

    feats = np.concatenate([arr, kin], axis=1).astype(np.float32)  # [N,11]

    # downsample to max_len
    if feats.shape[0] > max_len:
        print(f"Downsampling traces from {feats.shape[0]} to {max_len} points")
        idx = np.linspace(0, feats.shape[0] - 1, max_len, dtype=int)
        feats = feats[idx]

    return torch.from_numpy(feats)  # [N,11]


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
    """Run prediction on a trace file with 11-D features and pass mask to the model."""
    # Load traces
    with open(traces_file, 'r') as f:
        data = json.load(f)

    traces = data if isinstance(data, list) else data.get('traces', data.get('trajectory', []))
    if len(traces) == 0:
        print("Warning: No traces found in file")
        return []

    # Build features and mask
    trace_tensor, mask = process_traces(traces)             # [N,11], [N]
    trace_tensor = trace_tensor.unsqueeze(0).to(device)     # [1,N,11]
    mask = mask.unsqueeze(0).to(device)                     # [1,N]

    # --- Safety: adapt to model's expected input feature dim (11 vs 4) ---
    in_feat = None
    try:
        in_feat = getattr(getattr(model, 'encoder', None).input_proj, 'in_features', None)
    except Exception:
        pass
    if in_feat is None:
        # Fallback: try to infer from first Linear in the encoder
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                in_feat = m.in_features
                break

    if in_feat is not None and trace_tensor.shape[-1] != in_feat:
        if trace_tensor.shape[-1] > in_feat:
            # Truncate (use first in_feat columns, e.g., drop kinematic features for old 4-D models)
            trace_tensor = trace_tensor[..., :in_feat]
        else:
            # Pad with zeros to match (rare)
            pad = torch.zeros(trace_tensor.size(0), trace_tensor.size(1), in_feat - trace_tensor.size(-1),
                              device=trace_tensor.device, dtype=trace_tensor.dtype)
            trace_tensor = torch.cat([trace_tensor, pad], dim=-1)

    # Forward (pass mask if the model supports it)
    with torch.no_grad():
        try:
            outputs = model(trace_tensor, mask)  # new models expect (traces, mask)
        except TypeError:
            outputs = model(trace_tensor)        # fallback for legacy signature

    # Post-process
    pred_boxes = outputs['pred_boxes'][0]      # [Q,6]
    pred_classes = outputs['pred_classes'][0]  # [Q,4]
    predictions = post_process_predictions(pred_boxes, pred_classes,
                                           confidence_threshold, nms_threshold)
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