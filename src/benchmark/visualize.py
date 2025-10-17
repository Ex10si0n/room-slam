import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import argparse
from pathlib import Path


def plot_top_view(traces, colliders, predictions=None, title="Top View"):
    """Plot top-down view (X-Z plane)"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot traces
    if traces:
        x = [p['x'] for p in traces]
        z = [p['z'] for p in traces]
        ax.plot(x, z, 'b-', alpha=0.5, linewidth=0.5, label='Trace')
        ax.plot(x[0], z[0], 'go', markersize=10, label='Start')
        ax.plot(x[-1], z[-1], 'ro', markersize=10, label='End')

    # Plot ground truth colliders
    if colliders:
        for col in colliders:
            center = col['center']
            size = col['size']
            label = col.get('label', 'BLOCK')

            # Rectangle in X-Z plane
            rect = Rectangle(
                (center['x'] - size['x'] / 2, center['z'] - size['z'] / 2),
                size['x'], size['z'],
                linewidth=2, edgecolor='red', facecolor='red', alpha=0.3,
                label='GT' if colliders.index(col) == 0 else ''
            )
            ax.add_patch(rect)

            # Add label
            ax.text(center['x'], center['z'], label,
                    ha='center', va='center', fontsize=8, color='red')

    # Plot predictions
    if predictions:
        for pred in predictions:
            center = pred['center']
            size = pred['size']
            label = pred.get('label', 'PRED')
            conf = pred.get('confidence', 1.0)

            rect = Rectangle(
                (center['x'] - size['x'] / 2, center['z'] - size['z'] / 2),
                size['x'], size['z'],
                linewidth=2, edgecolor='blue', facecolor='none',
                linestyle='--', alpha=0.8,
                label='Pred' if predictions.index(pred) == 0 else ''
            )
            ax.add_patch(rect)

            # Add label with confidence
            ax.text(center['x'], center['z'], f"{label}\n{conf:.2f}",
                    ha='center', va='center', fontsize=7, color='blue')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Z Position')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    return fig


def plot_side_view(traces, colliders, predictions=None, title="Side View"):
    """Plot side view (X-Y plane)"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot traces
    if traces:
        x = [p['x'] for p in traces]
        y = [p['y'] for p in traces]
        ax.plot(x, y, 'b-', alpha=0.5, linewidth=0.5, label='Trace')

    # Plot ground truth colliders
    if colliders:
        for col in colliders:
            center = col['center']
            size = col['size']
            label = col.get('label', 'BLOCK')

            rect = Rectangle(
                (center['x'] - size['x'] / 2, center['y'] - size['y'] / 2),
                size['x'], size['y'],
                linewidth=2, edgecolor='red', facecolor='red', alpha=0.3,
                label='GT' if colliders.index(col) == 0 else ''
            )
            ax.add_patch(rect)

            ax.text(center['x'], center['y'], label,
                    ha='center', va='center', fontsize=8, color='red')

    # Plot predictions
    if predictions:
        for pred in predictions:
            center = pred['center']
            size = pred['size']
            label = pred.get('label', 'PRED')

            rect = Rectangle(
                (center['x'] - size['x'] / 2, center['y'] - size['y'] / 2),
                size['x'], size['y'],
                linewidth=2, edgecolor='blue', facecolor='none',
                linestyle='--', alpha=0.8,
                label='Pred' if predictions.index(pred) == 0 else ''
            )
            ax.add_patch(rect)

            ax.text(center['x'], center['y'], label,
                    ha='center', va='center', fontsize=7, color='blue')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position (Height)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Input data file (JSON with traces and colliders)')
    parser.add_argument('--colliders', type=str, default=None,
                        help='Separate colliders file (optional)')
    parser.add_argument('--predictions', type=str, default=None,
                        help='Prediction file (optional)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image file')
    args = parser.parse_args()

    # Load trace data
    with open(args.input, 'r') as f:
        data = json.load(f)

    # Handle both formats for traces
    if isinstance(data, list):
        traces = data
        colliders = []
    else:
        traces = data.get('traces', data.get('trajectory', []))
        colliders = data.get('colliders', [])

    # Load separate colliders file if provided
    if args.colliders:
        with open(args.colliders, 'r') as f:
            collider_data = json.load(f)
            if isinstance(collider_data, dict):
                colliders = collider_data.get('colliders', [])
            else:
                colliders = collider_data

    # Load predictions
    predictions = None
    if args.predictions:
        with open(args.predictions, 'r') as f:
            pred_data = json.load(f)
            predictions = pred_data.get('colliders', [])

    # Create plots
    fig_top = plot_top_view(traces, colliders, predictions,
                            title=f"Top View - {Path(args.input).name}")

    if traces and any('y' in p for p in traces):
        fig_side = plot_side_view(traces, colliders, predictions,
                                  title=f"Side View - {Path(args.input).name}")

    # Save or show
    if args.output:
        fig_top.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {args.output}")

        if traces and any('y' in p for p in traces):
            side_output = args.output.replace('.png', '_side.png')
            fig_side.savefig(side_output, dpi=150, bbox_inches='tight')
            print(f"Saved side view to {side_output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()