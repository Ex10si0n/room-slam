import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from pathlib import Path
import torch

# Import model inference utilities
from inference import load_model, predict


def plot_top_view(traces, colliders, predictions=None, title="Top View (X-Z Plane)"):
    """Plot top-down view (X-Z plane) - main view for 2D model."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot traces
    if traces:
        x = [p['x'] for p in traces]
        z = [p['z'] for p in traces]
        ax.plot(x, z, 'b-', alpha=0.5, linewidth=0.5, label='Trace')
        ax.plot(x[0], z[0], 'go', markersize=10, label='Start')
        ax.plot(x[-1], z[-1], 'ro', markersize=10, label='End')

    # Plot ground-truth colliders
    if colliders:
        for idx, col in enumerate(colliders):
            center = col['center']
            size = col['size']
            label = col.get('label', 'BLOCK')

            rect = Rectangle(
                (center['x'] - size['x'] / 2, center['z'] - size['z'] / 2),
                size['x'], size['z'],
                linewidth=2,
                edgecolor='red',
                facecolor='red',
                alpha=0.3,
                label='GT' if idx == 0 else ''
            )
            ax.add_patch(rect)

            ax.text(
                center['x'], center['z'], label,
                ha='center', va='center',
                fontsize=8, color='red', weight='bold'
            )

    # Plot predictions
    if predictions:
        for idx, pred in enumerate(predictions):
            center = pred['center']
            size = pred['size']
            label = pred.get('label', 'PRED')
            conf = pred.get('confidence', 1.0)

            rect = Rectangle(
                (center['x'] - size['x'] / 2, center['z'] - size['z'] / 2),
                size['x'], size['z'],
                linewidth=2,
                edgecolor='blue',
                facecolor='none',
                linestyle='--',
                alpha=0.8,
                label='Pred' if idx == 0 else ''
            )
            ax.add_patch(rect)

            ax.text(
                center['x'], center['z'], f"{label}\n{conf:.2f}",
                ha='center', va='center',
                fontsize=7, color='blue', weight='bold'
            )

    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Z Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    return fig


def plot_side_view(traces, colliders, predictions=None, title="Side View (X-Y Plane)"):
    """
    Plot side view (X-Y plane).
    Note: For 2D models, y coordinates are set to 0, so this view may not be meaningful.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot traces
    if traces:
        x = [p['x'] for p in traces]
        y = [p.get('y', 0.0) for p in traces]  # Default to 0 if y not present
        ax.plot(x, y, 'b-', alpha=0.5, linewidth=0.5, label='Trace')

    # Plot ground-truth colliders
    if colliders:
        for idx, col in enumerate(colliders):
            center = col['center']
            size = col['size']
            label = col.get('label', 'BLOCK')

            rect = Rectangle(
                (center['x'] - size['x'] / 2, center.get('y', 0.0) - size.get('y', 2.0) / 2),
                size['x'], size.get('y', 2.0),
                linewidth=2,
                edgecolor='red',
                facecolor='red',
                alpha=0.3,
                label='GT' if idx == 0 else ''
            )
            ax.add_patch(rect)

            ax.text(
                center['x'], center.get('y', 0.0), label,
                ha='center', va='center',
                fontsize=8, color='red', weight='bold'
            )

    # Plot predictions
    if predictions:
        for idx, pred in enumerate(predictions):
            center = pred['center']
            size = pred['size']
            label = pred.get('label', 'PRED')

            rect = Rectangle(
                (center['x'] - size['x'] / 2, center.get('y', 0.0) - size.get('y', 2.0) / 2),
                size['x'], size.get('y', 2.0),
                linewidth=2,
                edgecolor='blue',
                facecolor='none',
                linestyle='--',
                alpha=0.8,
                label='Pred' if idx == 0 else ''
            )
            ax.add_patch(rect)

            ax.text(
                center['x'], center.get('y', 0.0), label,
                ha='center', va='center',
                fontsize=7, color='blue', weight='bold'
            )

    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position / Height (m)', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Add note for 2D models
    if predictions and all(p['center'].get('y', 0.0) == 0.0 for p in predictions):
        ax.text(
            0.5, 0.02,
            'Note: Y coordinates from 2D model (all set to ground level)',
            transform=ax.transAxes,
            ha='center', fontsize=9, style='italic', color='gray'
        )

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Input data file (JSON with traces and/or colliders)')
    parser.add_argument('--colliders', type=str, default=None,
                        help='Separate colliders file (optional)')
    parser.add_argument('--predictions', type=str, default=None,
                        help='Prediction file (optional, JSON with "colliders")')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='If set (and --predictions is not), run model inference to get predictions')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image file (top view). If not set, show interactively.')
    parser.add_argument('--side_output', type=str, default=None,
                        help='Optional output file for side view')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Confidence threshold when running inference')
    parser.add_argument('--nms', type=float, default=0.3,
                        help='NMS IoU threshold when running inference')
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

    # Load or compute predictions
    predictions = None
    if args.predictions:
        with open(args.predictions, 'r') as f:
            pred_data = json.load(f)
            predictions = pred_data.get('colliders', pred_data)
    elif args.checkpoint:
        # Run model inference directly
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")

        print(f"Loading model from {args.checkpoint}")
        model = load_model(args.checkpoint, device)

        print(f"Running inference on {args.input}")
        predictions = predict(
            model,
            args.input,
            device,
            confidence_threshold=args.threshold,
            nms_threshold=args.nms
        )

    # Create plots
    input_name = Path(args.input).name
    fig_top = plot_top_view(
        traces,
        colliders,
        predictions,
        title=f"Top View (X-Z) - {input_name}"
    )

    fig_side = plot_side_view(
        traces,
        colliders,
        predictions,
        title=f"Side View (X-Y) - {input_name}"
    )

    # Save or show
    if args.output:
        fig_top.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved top view to {args.output}")

        if args.side_output:
            fig_side.savefig(args.side_output, dpi=150, bbox_inches='tight')
            print(f"Saved side view to {args.side_output}")

    else:
        plt.show()


if __name__ == "__main__":
    main()