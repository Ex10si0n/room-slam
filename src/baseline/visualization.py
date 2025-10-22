import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def visualize(trace_file, collider_file, gt_file=None, output_file=None):
    """Visualize trace, predicted colliders, and optionally ground truth colliders"""
    trace_data = load_json(trace_file) if trace_file else None
    collider_data = load_json(collider_file) if collider_file else None
    gt_data = load_json(gt_file) if gt_file else None

    fig, ax = plt.subplots(figsize=(10, 12))

    # Plot ground truth colliders (green)
    if gt_data:
        for col in gt_data['colliders']:
            center = col['center']
            size = col['size']
            label = col.get('label', 'GT')
            rect = patches.Rectangle(
                (center['x'] - size['x'] / 2, center['z'] - size['z'] / 2),
                size['x'], size['z'],
                linewidth=2, edgecolor='green', facecolor='green', alpha=0.2,
                linestyle='--'
            )
            ax.add_patch(rect)

        # Add legend entry for GT
        ax.plot([], [], 's', color='green', alpha=0.2, markersize=10,
                markeredgecolor='green', markeredgewidth=2, linestyle='--',
                label='GT Colliders')

    # Plot predicted colliders (red)
    if collider_data:
        for col in collider_data['colliders']:
            center = col['center']
            size = col['size']
            rect = patches.Rectangle(
                (center['x'] - size['x'] / 2, center['z'] - size['z'] / 2),
                size['x'], size['z'],
                linewidth=2, edgecolor='red', facecolor='red', alpha=0.3
            )
            ax.add_patch(rect)

        # Add legend entry for predicted
        ax.plot([], [], 's', color='red', alpha=0.3, markersize=10,
                markeredgecolor='red', markeredgewidth=2,
                label='Pred Colliders')

    # Plot trace
    if trace_data:
        xs = [p['x'] for p in trace_data]
        zs = [p['z'] for p in trace_data]
        ax.plot(xs, zs, 'b-', alpha=0.6, linewidth=1, label='Trace')

        # Plot start/end
        ax.plot(xs[0], zs[0], 'go', markersize=10, label='Start')
        ax.plot(xs[-1], zs[-1], 'ro', markersize=10, label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    title = 'Trace and Colliders'
    if gt_data and collider_data:
        title += ' (Red=Pred, Green=GT)'
    elif gt_data:
        title += ' (Ground Truth)'
    ax.set_title(title)

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization -> {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize trace and colliders')
    parser.add_argument('--trace', '-t', help='Trace JSON file')
    parser.add_argument('--colliders', '-c', help='Predicted colliders JSON file')
    parser.add_argument('--gt', '-g', help='Ground truth colliders JSON file')
    parser.add_argument('--output', '-o', help='Output image file (optional)')

    args = parser.parse_args()
    visualize(args.trace, args.colliders, args.gt, args.output)


if __name__ == '__main__':
    main()