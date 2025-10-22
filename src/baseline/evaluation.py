import json
import argparse
import numpy as np
from pathlib import Path


def load_colliders(filepath):
    """Load colliders from JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['colliders']


def rasterize_colliders(colliders, map_bounds, grid_size=0.05):
    """Convert colliders to grid representation"""
    min_x, max_x, min_z, max_z = map_bounds
    x_bins = int((max_x - min_x) / grid_size)
    z_bins = int((max_z - min_z) / grid_size)

    # Create grid with class labels (0=empty, 1=BLOCK, 2=MID, 3=LOW)
    label_map = {'BLOCK': 1, 'MID': 2, 'LOW': 3}
    grid = np.zeros((z_bins, x_bins), dtype=np.int32)

    for col in colliders:
        label = label_map.get(col['label'], 0)
        if label == 0:
            continue

        center = col['center']
        size = col['size']

        # Calculate box bounds in XZ plane
        x_min = center['x'] - size['x'] / 2
        x_max = center['x'] + size['x'] / 2
        z_min = center['z'] - size['z'] / 2
        z_max = center['z'] + size['z'] / 2

        # Convert to grid indices
        j_min = int((x_min - min_x) / grid_size)
        j_max = int((x_max - min_x) / grid_size)
        i_min = int((z_min - min_z) / grid_size)
        i_max = int((z_max - min_z) / grid_size)

        # Clip to grid bounds
        j_min = max(0, min(j_min, x_bins))
        j_max = max(0, min(j_max, x_bins))
        i_min = max(0, min(i_min, z_bins))
        i_max = max(0, min(i_max, z_bins))

        # Fill grid
        grid[i_min:i_max, j_min:j_max] = label

    return grid


def calculate_metrics(pred_grid, gt_grid, num_classes=4):
    """Calculate segmentation metrics

    Classes: 0=empty, 1=BLOCK, 2=MID, 3=LOW
    """
    # Flatten grids
    pred_flat = pred_grid.flatten()
    gt_flat = gt_grid.flatten()

    # Confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            confusion[i, j] = np.sum((gt_flat == i) & (pred_flat == j))

    # Per-class metrics
    per_class_metrics = {}
    ious = []

    for cls in range(num_classes):
        tp = confusion[cls, cls]
        fp = confusion[:, cls].sum() - tp
        fn = confusion[cls, :].sum() - tp

        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        ious.append(iou)

        cls_name = ['empty', 'BLOCK', 'MID', 'LOW'][cls]
        per_class_metrics[cls_name] = {
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }

    # Overall metrics
    miou = np.mean(ious)

    # Overall precision, recall, f1 (micro-average)
    tp_total = np.diag(confusion).sum()
    fp_total = confusion.sum() - tp_total
    fn_total = confusion.sum() - tp_total

    precision = tp_total / (tp_total + fp_total + 1e-8)
    recall = tp_total / (tp_total + fn_total + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Classification accuracy
    cls_correct = np.diag(confusion).sum()
    cls_total = confusion.sum()
    cls_acc = cls_correct / cls_total

    return {
        'mIoU': float(miou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'cls_acc': float(cls_acc),
        'tp': int(tp_total),
        'fp': int(fp_total),
        'fn': int(fn_total),
        'per_class': per_class_metrics,
        'confusion_matrix': confusion.tolist(),
        'cls_correct': int(cls_correct),
        'cls_total': int(cls_total)
    }


def evaluate_colliders(pred_file, gt_file, map_bounds, grid_size=0.05):
    """Evaluate predicted colliders against ground truth"""
    pred_colliders = load_colliders(pred_file)
    gt_colliders = load_colliders(gt_file)

    print(f"Loaded {len(pred_colliders)} predicted colliders")
    print(f"Loaded {len(gt_colliders)} ground truth colliders")

    # Rasterize
    pred_grid = rasterize_colliders(pred_colliders, map_bounds, grid_size)
    gt_grid = rasterize_colliders(gt_colliders, map_bounds, grid_size)

    # Calculate metrics
    metrics = calculate_metrics(pred_grid, gt_grid)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate collider accuracy')
    parser.add_argument('--pred', required=True, help='Predicted colliders JSON')
    parser.add_argument('--gt', required=True, help='Ground truth colliders JSON')
    parser.add_argument('--bounds', nargs=4, type=float, default=[-2, 2, -6, 3],
                        metavar=('MIN_X', 'MAX_X', 'MIN_Z', 'MAX_Z'),
                        help='Map bounds')
    parser.add_argument('--grid-size', type=float, default=0.05,
                        help='Grid resolution for evaluation')
    parser.add_argument('--output', '-o', help='Output JSON file for metrics')

    args = parser.parse_args()

    # Evaluate
    metrics = evaluate_colliders(
        args.pred,
        args.gt,
        args.bounds,
        args.grid_size
    )

    # Print results
    print("\n=== Overall Metrics ===")
    print(f"mIoU:      {metrics['mIoU']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Accuracy:  {metrics['cls_acc']:.4f}")

    print("\n=== Per-Class Metrics ===")
    for cls_name, cls_metrics in metrics['per_class'].items():
        print(f"\n{cls_name}:")
        print(f"  IoU:       {cls_metrics['iou']:.4f}")
        print(f"  Precision: {cls_metrics['precision']:.4f}")
        print(f"  Recall:    {cls_metrics['recall']:.4f}")
        print(f"  F1:        {cls_metrics['f1']:.4f}")

    # Save to file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {args.output}")


if __name__ == '__main__':
    main()