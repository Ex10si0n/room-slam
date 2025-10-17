"""
Test script to visualize rotation augmentation.
Generates 4 plots showing the same scene rotated at 0°, 90°, 180°, 270°.
"""

import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from dataloader import TraceColliderDataset


def plot_augmented_views(dataset, sample_idx=0):
    """
    Plot all 4 rotated versions of a single sample.

    Args:
        dataset: TraceColliderDataset with augmentation enabled
        sample_idx: Base sample index (will show all 4 rotations)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    # Get 4 augmented versions of the same base sample
    angles = [0, 90, 180, 270]

    for i, (ax, angle) in enumerate(zip(axes, angles)):
        # Calculate actual index (each base sample has 4 augmented versions)
        aug_idx = sample_idx * 4 + i

        if aug_idx >= len(dataset):
            print(f"Warning: Sample index {aug_idx} out of range")
            continue

        sample = dataset[aug_idx]

        # Extract data
        traces = sample['traces'].numpy()  # [N, 4]
        boxes = sample['boxes'].numpy()  # [M, 6]
        labels = sample['labels'].numpy()  # [M]
        valid_mask = sample['valid_mask'].numpy()  # [M]
        rotation = sample['rotation'].item()

        # Plot traces (X-Z plane)
        ax.plot(traces[:, 0], traces[:, 2], 'b-', alpha=0.3, linewidth=0.5, label='Trace')
        ax.plot(traces[0, 0], traces[0, 2], 'go', markersize=10, label='Start')
        ax.plot(traces[-1, 0], traces[-1, 2], 'ro', markersize=10, label='End')

        # Plot colliders
        label_map = {0: 'BLOCK', 1: 'LOW', 2: 'MID', 3: 'HIGH'}
        colors = {'BLOCK': 'red', 'LOW': 'orange', 'MID': 'purple', 'HIGH': 'brown'}

        for j in range(len(boxes)):
            if not valid_mask[j]:
                continue

            box = boxes[j]
            label = label_map[labels[j]]

            # Rectangle in X-Z plane
            rect = Rectangle(
                (box[0] - box[3] / 2, box[2] - box[5] / 2),  # (x-sx/2, z-sz/2)
                box[3], box[5],  # sx, sz
                linewidth=2,
                edgecolor=colors[label],
                facecolor=colors[label],
                alpha=0.3
            )
            ax.add_patch(rect)

            # Add label text
            ax.text(box[0], box[2], label,
                    ha='center', va='center', fontsize=7,
                    color=colors[label], fontweight='bold')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Z Position')
        ax.set_title(f'Rotation: {int(rotation)}°')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def main():
    # Create dataset with augmentation
    data_dir = "../../dataset"

    print("Loading dataset with rotation augmentation...")
    dataset = TraceColliderDataset(
        data_dir=data_dir,
        max_trace_len=3000,
        max_colliders=50,
        augment_rotation=True,
        rotation_angles=[0, 90, 180, 270]
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Base samples: {len(dataset) // 4}")
    print(f"Augmentation factor: 4x (0°, 90°, 180°, 270°)")

    # Test single sample
    print("\nTesting sample 0...")
    sample = dataset[0]
    print(f"  Traces shape: {sample['traces'].shape}")
    print(f"  Boxes shape: {sample['boxes'].shape}")
    print(f"  Valid colliders: {sample['valid_mask'].sum().item()}")
    print(f"  Rotation: {sample['rotation'].item()}°")
    print(f"  Filename: {sample['filename']}")

    # Visualize augmentation
    print("\nGenerating visualization...")
    fig = plot_augmented_views(dataset, sample_idx=0)

    output_path = "augmentation_test.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

    # Test rotation correctness
    print("\nVerifying rotation correctness...")
    sample_0 = dataset[0]  # 0°
    sample_90 = dataset[1]  # 90°
    sample_180 = dataset[2]  # 180°
    sample_270 = dataset[3]  # 270°

    # Check that collider centers are rotated correctly
    box_0 = sample_0['boxes'][0, :3].numpy()  # First collider center at 0°
    box_90 = sample_90['boxes'][0, :3].numpy()  # Same collider at 90°

    print(f"  Original center (0°): x={box_0[0]:.3f}, z={box_0[2]:.3f}")
    print(f"  Rotated center (90°): x={box_90[0]:.3f}, z={box_90[2]:.3f}")

    # At 90°: (x, z) -> (-z, x)
    expected_x = -box_0[2]
    expected_z = box_0[0]
    print(f"  Expected at 90°: x={expected_x:.3f}, z={expected_z:.3f}")

    error = np.sqrt((box_90[0] - expected_x) ** 2 + (box_90[2] - expected_z) ** 2)
    print(f"  Rotation error: {error:.6f}")

    if error < 0.01:
        print("  ✅ Rotation is correct!")
    else:
        print("  ⚠️ Rotation may have issues")

    print("\n✓ Test completed!")


if __name__ == "__main__":
    main()