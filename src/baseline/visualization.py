import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def visualize(trace_file, collider_file, output_file=None):
    """Visualize trace and colliders"""
    trace_data = load_json(trace_file)
    collider_data = load_json(collider_file)

    fig, ax = plt.subplots(figsize=(10, 12))

    # Plot colliders
    for col in collider_data['colliders']:
        center = col['center']
        size = col['size']
        rect = patches.Rectangle(
            (center['x'] - size['x'] / 2, center['z'] - size['z'] / 2),
            size['x'], size['z'],
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.3,
            label=col['label']
        )
        ax.add_patch(rect)

    # Plot trace
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
    ax.set_title('Trace and Colliders')

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization -> {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize trace and colliders')
    parser.add_argument('--trace', '-t', help='Trace JSON file')
    parser.add_argument('--colliders', '-c', help='Colliders JSON file')
    parser.add_argument('--output', '-o', help='Output image file (optional)')

    args = parser.parse_args()
    visualize(args.trace, args.colliders, args.output)


if __name__ == '__main__':
    main()