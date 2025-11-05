import torch
import json
import argparse
import numpy as np
from scipy.ndimage import distance_transform_edt

from model import build_model


def load_model(checkpoint_path: str, device):
    """
    Load trained Transformer trace+grid model from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    model = build_model(
        num_queries=config.get('num_queries', 30),
        d_model=config.get('d_model', 128),
    )

    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval()
    return model


def process_traces(traces, max_len: int = 3000):
    """
    Convert trace list to a [N,7] tensor (2D version):
        [x, z, vx, vz, ax, az, speed]
    """
    if len(traces) == 0:
        return torch.zeros((1, 7), dtype=torch.float32)

    # Collect x, z, timestamp
    trace_list = []
    for p in traces:
        trace_list.append([
            p.get('x', 0.0),
            p.get('z', 0.0),
            p.get('timestamp', 0.0)
        ])

    trace_array = np.array(trace_list, dtype=np.float32)  # [N, 3]

    # Sort by timestamp
    if trace_array.shape[0] > 1:
        order = np.argsort(trace_array[:, 2])
        trace_array = trace_array[order]

    # Normalize time
    if trace_array.shape[0] > 0:
        trace_array[:, 2] -= trace_array[0, 2]

    # Compute 2D kinematics
    diffs = np.diff(trace_array, axis=0, prepend=trace_array[[0], :])  # [N, 3]
    dt = np.clip(diffs[:, 2], 1e-3, None)  # [N]

    # 2D velocity (x, z only)
    vel = diffs[:, :2] / dt[:, None]  # [N, 2]
    vel = np.clip(vel, -10.0, 10.0)

    # 2D acceleration
    vel_diffs = np.diff(vel, axis=0, prepend=vel[[0], :])  # [N, 2]
    acc = vel_diffs / dt[:, None]  # [N, 2]
    acc = np.clip(acc, -20.0, 20.0)

    # 2D speed
    speed = np.linalg.norm(vel, axis=1, keepdims=True)  # [N, 1]

    # Concatenate: [x, z] + [vx, vz] + [ax, az] + [speed]
    # Result: [N, 2+2+2+1 = 7]
    kin = np.concatenate([vel, acc, speed], axis=1)  # [N, 5]
    feats = np.concatenate([trace_array[:, :2], kin], axis=1)  # [N, 7]

    # Downsample to max_len
    if feats.shape[0] > max_len:
        print(f"Downsampling traces from {feats.shape[0]} to {max_len} points")
        idx = np.linspace(0, feats.shape[0] - 1, max_len, dtype=int)
        feats = feats[idx]

    return torch.from_numpy(feats.astype(np.float32))  # [N, 7]


def rasterize_traces_to_grid(traces, grid_size: int = 64):
    """
    Rasterize traces into a top-down 2D grid (Walk2Map-style inverse-distance map).

    Args:
        traces: list of dicts, each with keys x, z (y ignored)
        grid_size: output grid size (H = W = grid_size)

    Returns:
        grid: [1, H, W] float32 tensor in [0,1]
    """
    H = W = grid_size

    if len(traces) == 0:
        return torch.zeros((1, H, W), dtype=torch.float32)

    xs = np.array([p['x'] for p in traces], dtype=np.float32)
    zs = np.array([p['z'] for p in traces], dtype=np.float32)

    eps = 1e-3
    min_x, max_x = xs.min(), xs.max()
    min_z, max_z = zs.min(), zs.max()
    width = max(max_x - min_x, eps)
    height = max(max_z - min_z, eps)

    scale = max(width, height)
    nx = (xs - min_x) / scale  # 0~1
    nz = (zs - min_z) / scale  # 0~1

    ix = np.clip((nx * (W - 1)).astype(np.int32), 0, W - 1)
    iz = np.clip((nz * (H - 1)).astype(np.int32), 0, H - 1)

    free = np.ones((H, W), dtype=bool)
    free[iz, ix] = False

    dist = distance_transform_edt(free)
    if dist.max() > 0:
        dist = dist / dist.max()

    inv = 1.0 - dist  # closer to trace = higher value

    return torch.from_numpy(inv.astype(np.float32)).unsqueeze(0)  # [1, H, W]


def compute_iou_2d(box1, box2):
    """
    Compute 2D IoU between two boxes.
    box: [cx, cz, sx, sz]
    """
    # Convert to corner format
    box1_min = box1[:2] - box1[2:] / 2
    box1_max = box1[:2] + box1[2:] / 2
    box2_min = box2[:2] - box2[2:] / 2
    box2_max = box2[:2] + box2[2:] / 2

    # Intersection
    inter_min = torch.maximum(box1_min, box2_min)
    inter_max = torch.minimum(box1_max, box2_max)
    inter_size = torch.clamp(inter_max - inter_min, min=0)
    inter_area = inter_size.prod()

    # Union
    box1_area = box1[2:].prod()
    box2_area = box2[2:].prod()
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou.item()


def nms_2d(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression for 2D boxes.

    Args:
        boxes: [N, 4] tensor (cx, cz, sx, sz)
        scores: [N] tensor
        iou_threshold: IoU threshold for suppression

    Returns:
        keep_indices: List of indices to keep
    """
    if len(boxes) == 0:
        return []

    sorted_indices = torch.argsort(scores, descending=True)
    keep = []

    while len(sorted_indices) > 0:
        current = sorted_indices[0].item()
        keep.append(current)

        if len(sorted_indices) == 1:
            break

        current_box = boxes[current]
        remaining_indices = sorted_indices[1:]

        new_remaining = []
        for idx in remaining_indices:
            iou = compute_iou_2d(current_box, boxes[idx])
            if iou < iou_threshold:
                new_remaining.append(idx)

        if len(new_remaining) == 0:
            break

        sorted_indices = torch.tensor(new_remaining, dtype=torch.long, device=boxes.device)

    return keep


def post_process_predictions(boxes, classes, confidence_threshold=0.7, nms_threshold=0.3):
    """
    Filter and format predictions with per-class NMS.

    Args:
        boxes: [Q, 4] (cx, cz, sx, sz)
        classes: [Q, 3] logits

    Returns:
        List[dict]: collider objects in BoxCollider-style format (with y=0)
    """
    label_map = {0: 'BLOCK', 1: 'LOW', 2: 'MID'}

    probs = torch.softmax(classes, dim=-1)  # [Q, 3]
    max_probs, pred_labels = probs.max(dim=-1)

    valid_mask = max_probs > confidence_threshold
    valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)

    if len(valid_indices) == 0:
        return []

    valid_boxes = boxes[valid_indices]
    valid_scores = max_probs[valid_indices]
    valid_labels = pred_labels[valid_indices]

    final_indices = []

    # Per-class NMS
    for label_id in range(3):
        class_mask = valid_labels == label_id
        class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(-1)
        if class_indices.numel() == 0:
            continue
        if class_indices.dim() == 0:
            class_indices = class_indices.unsqueeze(0)

        class_boxes = valid_boxes[class_indices]
        class_scores = valid_scores[class_indices]

        keep_in_class = nms_2d(class_boxes, class_scores, nms_threshold)

        for k in keep_in_class:
            final_indices.append(class_indices[k].item())

    predictions = []
    for idx in final_indices:
        box = valid_boxes[idx].cpu().numpy()
        label = label_map[valid_labels[idx].item()]
        conf = valid_scores[idx].item()

        # Convert 2D box to 3D format (y=0, sy=default height)
        predictions.append({
            'type': 'BoxCollider',
            'label': label,
            'confidence': float(conf),
            'center': {
                'x': float(box[0]),
                'y': 0.0,  # Default ground level
                'z': float(box[1])
            },
            'size': {
                'x': float(box[2]),
                'y': 2.0,  # Default height for visualization
                'z': float(box[3])
            },
            'radius': 0.0,
            'height': 0.0
        })

    return predictions


def predict(model, traces_file, device, confidence_threshold=0.7, nms_threshold=0.3):
    """
    Run prediction on a trace JSON file with a 2D trace+grid Transformer model.
    """
    # Load traces
    with open(traces_file, 'r') as f:
        data = json.load(f)

    traces = data if isinstance(data, list) else data.get('traces', data.get('trajectory', []))
    if len(traces) == 0:
        print("Warning: No traces found in file")
        return []

    # Build 2D trace features [N, 7]
    trace_tensor = process_traces(traces)

    if not isinstance(trace_tensor, torch.Tensor):
        trace_tensor = torch.as_tensor(trace_tensor, dtype=torch.float32)

    # Add batch dimension
    trace_tensor = trace_tensor.unsqueeze(0).to(device)  # [1, N, 7]
    mask = torch.ones(trace_tensor.shape[1], dtype=torch.bool)
    mask = mask.unsqueeze(0).to(device)  # [1, N]

    # Match feature dimension if needed
    in_feat = None
    enc = getattr(model, 'encoder', None)
    if enc is not None:
        ip = getattr(enc, 'input_proj', None)
        if isinstance(ip, torch.nn.Linear):
            in_feat = ip.in_features

    if in_feat is not None and trace_tensor.shape[-1] != in_feat:
        cur_feat = trace_tensor.shape[-1]
        if cur_feat > in_feat:
            trace_tensor = trace_tensor[..., :in_feat]
        else:
            pad = torch.zeros(
                trace_tensor.size(0),
                trace_tensor.size(1),
                in_feat - cur_feat,
                device=trace_tensor.device,
                dtype=trace_tensor.dtype
            )
            trace_tensor = torch.cat([trace_tensor, pad], dim=-1)

    # Build grid map from traces (Walk2Map-style inverse-distance map)
    grid_map = rasterize_traces_to_grid(traces, grid_size=64)  # [1, H, W]
    grid_map = grid_map.unsqueeze(0).to(device)  # [1, 1, H, W]

    with torch.no_grad():
        outputs = model(trace_tensor, mask, grid_maps=grid_map)

    pred_boxes = outputs['pred_boxes'][0]  # [Q, 4]
    pred_classes = outputs['pred_classes'][0]  # [Q, 3]
    predictions = post_process_predictions(
        pred_boxes, pred_classes,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold
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

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # Run prediction
    print(f"Processing {args.input}")
    predictions = predict(model, args.input, device, args.threshold, args.nms)

    print(f"\nFound {len(predictions)} colliders:")
    for i, pred in enumerate(predictions):
        print(
            f"  {i + 1}. {pred['label']} at "
            f"({pred['center']['x']:.2f}, {pred['center']['z']:.2f}) "
            f"size: ({pred['size']['x']:.2f} x {pred['size']['z']:.2f}) "
            f"- confidence: {pred['confidence']:.3f}"
        )

    # Save results
    if args.output:
        output_data = {
            'colliders': predictions,
            'metadata': {
                'num_colliders': len(predictions),
                'confidence_threshold': args.threshold,
                'nms_threshold': args.nms,
                'note': '2D model - y coordinates set to 0, y sizes set to 2.0 for visualization'
            }
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()