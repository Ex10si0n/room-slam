import torch
import json
import argparse

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
    Convert trace list to a [N,11] tensor:
        [x,y,z,t, vx,vy,vz, ax,ay,az, speed]
    """
    import numpy as np

    if len(traces) == 0:
        return torch.zeros((1, 11), dtype=torch.float32)

    # Raw array [N, 4]
    arr = np.array(
        [[p['x'], p['y'], p['z'], p['timestamp']] for p in traces],
        dtype=np.float32
    )

    # Sort by time & normalize time to start at 0
    order = np.argsort(arr[:, 3])
    arr = arr[order]
    arr[:, 3] -= arr[0, 3]

    # Kinematic features (order-sensitive)
    diffs = np.diff(arr, axis=0, prepend=arr[[0], :])
    dt = np.clip(diffs[:, 3], 1e-3, None)
    vel = diffs[:, :3] / dt[:, None]                    # [N,3]
    acc = np.diff(vel, axis=0, prepend=vel[[0], :])     # [N,3]
    speed = np.linalg.norm(vel, axis=1, keepdims=True)  # [N,1]
    kin = np.concatenate([vel, acc, speed], axis=1)     # [N,7]

    feats = np.concatenate([arr, kin], axis=1).astype(np.float32)  # [N,11]

    # Downsample to max_len
    if feats.shape[0] > max_len:
        print(f"Downsampling traces from {feats.shape[0]} to {max_len} points")
        idx = np.linspace(0, feats.shape[0] - 1, max_len, dtype=int)
        feats = feats[idx]

    return torch.from_numpy(feats)  # [N,11]


def rasterize_traces_to_grid(traces, grid_size: int = 64):
    """
    Rasterize 3D traces into a top-down 2D grid (Walk2Map-style inverse-distance map).

    Args:
        traces: list of dicts, each with keys x, y, z, timestamp
        grid_size: output grid size (H = W = grid_size)

    Returns:
        grid: [1, H, W] float32 tensor in [0,1]
    """
    import numpy as np

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

    return torch.from_numpy(inv.astype(np.float32)).unsqueeze(0)  # [1,H,W]


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

    # Intersection
    inter_min = torch.maximum(box1_min, box2_min)
    inter_max = torch.minimum(box1_max, box2_max)
    inter_size = torch.clamp(inter_max - inter_min, min=0)
    inter_volume = inter_size.prod()

    # Union
    box1_volume = box1[3:].prod()
    box2_volume = box2[3:].prod()
    union_volume = box1_volume + box2_volume - inter_volume

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
            iou = compute_iou_3d(current_box, boxes[idx])
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
        boxes: [Q, 6]
        classes: [Q, 3] logits

    Returns:
        List[dict]: collider objects in BoxCollider-style format
    """
    label_map = {0: 'BLOCK', 1: 'LOW', 2: 'MID'}

    probs = torch.softmax(classes, dim=-1)   # [Q,3]
    max_probs, pred_labels = probs.max(dim=-1)

    valid_mask = max_probs > confidence_threshold
    valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)

    if len(valid_indices) == 0:
        return []

    valid_boxes = boxes[valid_indices]
    valid_scores = max_probs[valid_indices]
    valid_labels = pred_labels[valid_indices]

    final_indices = []

    for label_id in range(3):
        class_mask = valid_labels == label_id
        class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(-1)
        if class_indices.numel() == 0:
            continue
        if class_indices.dim() == 0:
            class_indices = class_indices.unsqueeze(0)

        class_boxes = valid_boxes[class_indices]
        class_scores = valid_scores[class_indices]

        keep_in_class = nms_3d(class_boxes, class_scores, nms_threshold)

        for k in keep_in_class:
            final_indices.append(class_indices[k].item())

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
    """
    Run prediction on a trace JSON file with a trace+grid Transformer model.
    """
    # Load traces
    with open(traces_file, 'r') as f:
        data = json.load(f)

    traces = data if isinstance(data, list) else data.get('traces', data.get('trajectory', []))
    if len(traces) == 0:
        print("Warning: No traces found in file")
        return []

    # Build trace features
    out = process_traces(traces)
    if isinstance(out, tuple):
        if len(out) >= 2:
            trace_tensor, mask = out[0], out[1]
        elif len(out) == 1:
            trace_tensor = out[0]
            mask = torch.ones(trace_tensor.shape[0], dtype=torch.bool)
        else:
            raise ValueError("process_traces returned an empty tuple.")
    else:
        trace_tensor = out
        mask = torch.ones(trace_tensor.shape[0], dtype=torch.bool)

    if not isinstance(trace_tensor, torch.Tensor):
        trace_tensor = torch.as_tensor(trace_tensor, dtype=torch.float32)
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, dtype=torch.bool)

    # Add batch dimension
    trace_tensor = trace_tensor.unsqueeze(0).to(device)  # [1, N, F]
    mask = mask.unsqueeze(0).to(device)                  # [1, N]

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
    grid_map = grid_map.unsqueeze(0).to(device)                # [1,1,H,W]

    with torch.no_grad():
        outputs = model(trace_tensor, mask, grid_maps=grid_map)

    pred_boxes = outputs['pred_boxes'][0]      # [Q, 6]
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
            f"({pred['center']['x']:.2f}, {pred['center']['y']:.2f}, {pred['center']['z']:.2f}) "
            f"- confidence: {pred['confidence']:.3f}"
        )

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