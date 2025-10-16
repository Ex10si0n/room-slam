"""
Plot real dataset from /dataset directory
Handles movement traces and colliders with y as height dimension
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import List, Dict, Tuple
import glob


class RealDataPlotter:
    """Plotter for real dataset with 3D movement traces and colliders"""

    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = dataset_dir
        self.movement_files = []
        self.colliders_data = None
        self.load_data()

    def load_data(self):
        """Load all movement trace files and colliders"""
        # Load movement trace files
        pattern = os.path.join(self.dataset_dir, "*_data_*.json")
        self.movement_files = glob.glob(pattern)
        self.movement_files.sort()

        print(f"Found {len(self.movement_files)} movement trace files:")
        for file in self.movement_files:
            print(f"  - {os.path.basename(file)}")

        # Load colliders
        colliders_path = os.path.join(self.dataset_dir, "colliders.json")
        if os.path.exists(colliders_path):
            with open(colliders_path, "r") as f:
                self.colliders_data = json.load(f)
            print(f"Loaded {len(self.colliders_data['colliders'])} colliders")
        else:
            print("No colliders.json found")

    def load_movement_trace(self, file_path: str) -> Dict:
        """Load a single movement trace file"""
        with open(file_path, "r") as f:
            data = json.load(f)

        # Extract coordinates and timestamps
        timestamps = np.array([point["timestamp"] for point in data])
        x_coords = np.array([point["x"] for point in data])
        y_coords = np.array([point["y"] for point in data])  # Height
        z_coords = np.array([point["z"] for point in data])

        return {
            "timestamps": timestamps,
            "x": x_coords,
            "y": y_coords,  # Height
            "z": z_coords,
            "filename": os.path.basename(file_path),
            "num_points": len(data),
        }

    def plot_2d_top_view(
        self,
        trace_data: Dict,
        ax=None,
        show_colliders=True,
        color="blue",
        alpha=0.7,
        linewidth=1,
    ):
        """Plot 2D top view (x-z plane) with height as color"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        # Create scatter plot with height as color
        scatter = ax.scatter(
            trace_data["x"],
            trace_data["z"],
            c=trace_data["y"],
            cmap="viridis",
            s=20,
            alpha=alpha,
        )

        # Add colorbar for height
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Height (y)", fontsize=12)

        # Plot trajectory line
        ax.plot(
            trace_data["x"],
            trace_data["z"],
            color="red",
            alpha=0.3,
            linewidth=linewidth,
        )

        # Mark start and end points
        ax.scatter(
            trace_data["x"][0],
            trace_data["z"][0],
            color="green",
            s=100,
            marker="o",
            label="Start",
            zorder=5,
        )
        ax.scatter(
            trace_data["x"][-1],
            trace_data["z"][-1],
            color="red",
            s=100,
            marker="s",
            label="End",
            zorder=5,
        )

        # Plot colliders if available
        if show_colliders and self.colliders_data:
            self.plot_colliders_2d(ax)

        ax.set_xlabel("X Position", fontsize=12)
        ax.set_ylabel("Z Position", fontsize=12)
        ax.set_title(
            f'Top View: {trace_data["filename"]}\n'
            f'Points: {trace_data["num_points"]}, '
            f'Duration: {trace_data["timestamps"][-1] - trace_data["timestamps"][0]:.1f}s',
            fontsize=14,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        return ax

    def plot_colliders_2d(self, ax):
        """Plot colliders as rectangles in 2D top view"""
        for collider in self.colliders_data["colliders"]:
            if collider["type"] == "BoxCollider":
                center = collider["center"]
                size = collider["size"]

                # Create rectangle for box collider
                rect = plt.Rectangle(
                    (center["x"] - size["x"] / 2, center["z"] - size["z"] / 2),
                    size["x"],
                    size["z"],
                    facecolor="gray",
                    alpha=0.5,
                    edgecolor="black",
                    linewidth=1,
                )
                ax.add_patch(rect)

    def plot_3d_trajectory(self, trace_data: Dict, ax=None):
        """Plot 3D trajectory with height as vertical dimension"""
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

        # Plot 3D trajectory
        ax.plot(
            trace_data["x"],
            trace_data["z"],
            trace_data["y"],
            color="blue",
            alpha=0.6,
            linewidth=1,
        )

        # Scatter plot with height coloring
        scatter = ax.scatter(
            trace_data["x"],
            trace_data["z"],
            trace_data["y"],
            c=trace_data["y"],
            cmap="viridis",
            s=10,
            alpha=0.8,
        )

        # Mark start and end points
        ax.scatter(
            trace_data["x"][0],
            trace_data["z"][0],
            trace_data["y"][0],
            color="green",
            s=100,
            marker="o",
            label="Start",
        )
        ax.scatter(
            trace_data["x"][-1],
            trace_data["z"][-1],
            trace_data["y"][-1],
            color="red",
            s=100,
            marker="s",
            label="End",
        )

        ax.set_xlabel("X Position", fontsize=12)
        ax.set_ylabel("Z Position", fontsize=12)
        ax.set_zlabel("Height (Y)", fontsize=12)
        ax.set_title(f'3D Trajectory: {trace_data["filename"]}', fontsize=14)
        ax.legend()

        return ax

    def plot_height_profile(self, trace_data: Dict, ax=None):
        """Plot height profile over time"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # Calculate relative time
        relative_time = trace_data["timestamps"] - trace_data["timestamps"][0]

        ax.plot(relative_time, trace_data["y"], color="blue", linewidth=1)
        ax.fill_between(relative_time, trace_data["y"], alpha=0.3)

        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_ylabel("Height (y)", fontsize=12)
        ax.set_title(f'Height Profile: {trace_data["filename"]}', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add statistics
        height_mean = np.mean(trace_data["y"])
        height_std = np.std(trace_data["y"])
        height_min = np.min(trace_data["y"])
        height_max = np.max(trace_data["y"])

        ax.axhline(
            height_mean,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {height_mean:.3f}",
        )
        ax.axhline(
            height_mean + height_std,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label=f"+1σ: {height_mean + height_std:.3f}",
        )
        ax.axhline(
            height_mean - height_std,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label=f"-1σ: {height_mean - height_std:.3f}",
        )

        ax.legend()

        return ax

    def plot_all_traces_overview(self, save_path: str = None):
        """Plot overview of all movement traces"""
        num_files = len(self.movement_files)
        if num_files == 0:
            print("No movement trace files found!")
            return

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: All traces 2D top view
        ax1 = axes[0, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, num_files))

        for i, file_path in enumerate(self.movement_files):
            trace_data = self.load_movement_trace(file_path)
            ax1.plot(
                trace_data["x"],
                trace_data["z"],
                color=colors[i],
                alpha=0.7,
                linewidth=1,
                label=os.path.basename(file_path),
            )

        # Plot colliders
        if self.colliders_data:
            self.plot_colliders_2d(ax1)

        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Z Position")
        ax1.set_title("All Movement Traces (Top View)")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal")

        # Plot 2: Height distributions
        ax2 = axes[0, 1]
        all_heights = []
        labels = []

        for file_path in self.movement_files:
            trace_data = self.load_movement_trace(file_path)
            all_heights.append(trace_data["y"])
            labels.append(os.path.basename(file_path))

        ax2.boxplot(
            all_heights, labels=[os.path.basename(f) for f in self.movement_files]
        )
        ax2.set_ylabel("Height (y)")
        ax2.set_title("Height Distributions")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Movement duration vs points
        ax3 = axes[1, 0]
        durations = []
        point_counts = []

        for file_path in self.movement_files:
            trace_data = self.load_movement_trace(file_path)
            duration = trace_data["timestamps"][-1] - trace_data["timestamps"][0]
            durations.append(duration)
            point_counts.append(trace_data["num_points"])

        scatter = ax3.scatter(
            durations, point_counts, c=range(len(durations)), cmap="viridis", s=100
        )

        for i, file_path in enumerate(self.movement_files):
            ax3.annotate(
                os.path.basename(file_path),
                (durations[i], point_counts[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

        ax3.set_xlabel("Duration (seconds)")
        ax3.set_ylabel("Number of Points")
        ax3.set_title("Trace Duration vs Point Count")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Height range for each trace
        ax4 = axes[1, 1]
        height_ranges = []
        height_means = []

        for file_path in self.movement_files:
            trace_data = self.load_movement_trace(file_path)
            height_range = np.max(trace_data["y"]) - np.min(trace_data["y"])
            height_mean = np.mean(trace_data["y"])
            height_ranges.append(height_range)
            height_means.append(height_mean)

        bars = ax4.bar(
            range(len(self.movement_files)),
            height_ranges,
            color=plt.cm.viridis(np.array(height_means) / max(height_means)),
        )

        ax4.set_xlabel("Movement Trace")
        ax4.set_ylabel("Height Range")
        ax4.set_title("Height Range per Trace")
        ax4.set_xticks(range(len(self.movement_files)))
        ax4.set_xticklabels(
            [os.path.basename(f) for f in self.movement_files], rotation=45, ha="right"
        )
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Overview plot saved to {save_path}")

        plt.show()

    def plot_single_trace_detailed(self, file_path: str, save_path: str = None):
        """Plot detailed analysis of a single movement trace"""
        trace_data = self.load_movement_trace(file_path)

        # Create subplots
        fig = plt.figure(figsize=(16, 12))

        # Plot 1: 2D top view
        ax1 = plt.subplot(2, 3, 1)
        self.plot_2d_top_view(trace_data, ax1)

        # Plot 2: 3D trajectory
        ax2 = plt.subplot(2, 3, 2, projection="3d")
        self.plot_3d_trajectory(trace_data, ax2)

        # Plot 3: Height profile
        ax3 = plt.subplot(2, 3, 3)
        self.plot_height_profile(trace_data, ax3)

        # Plot 4: Speed profile
        ax4 = plt.subplot(2, 3, 4)
        relative_time = trace_data["timestamps"] - trace_data["timestamps"][0]

        # Calculate speed (magnitude of velocity)
        dt = np.diff(relative_time)
        dx = np.diff(trace_data["x"])
        dy = np.diff(trace_data["y"])
        dz = np.diff(trace_data["z"])

        speed = np.sqrt(dx**2 + dy**2 + dz**2) / dt

        ax4.plot(relative_time[1:], speed, color="green", linewidth=1)
        ax4.set_xlabel("Time (seconds)")
        ax4.set_ylabel("Speed (units/s)")
        ax4.set_title("Speed Profile")
        ax4.grid(True, alpha=0.3)

        # Plot 5: XY trajectory (side view)
        ax5 = plt.subplot(2, 3, 5)
        scatter = ax5.scatter(
            trace_data["x"], trace_data["y"], c=relative_time, cmap="plasma", s=20
        )
        ax5.plot(trace_data["x"], trace_data["y"], color="red", alpha=0.3, linewidth=1)
        ax5.set_xlabel("X Position")
        ax5.set_ylabel("Height (Y)")
        ax5.set_title("Side View (X-Y)")
        plt.colorbar(scatter, ax=ax5, label="Time")
        ax5.grid(True, alpha=0.3)

        # Plot 6: Statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis("off")

        # Calculate statistics
        stats_text = f"""
Statistics for {trace_data['filename']}:

Duration: {relative_time[-1]:.1f} seconds
Points: {trace_data['num_points']}

Position Ranges:
  X: [{np.min(trace_data['x']):.3f}, {np.max(trace_data['x']):.3f}]
  Y: [{np.min(trace_data['y']):.3f}, {np.max(trace_data['y']):.3f}]
  Z: [{np.min(trace_data['z']):.3f}, {np.max(trace_data['z']):.3f}]

Height Statistics:
  Mean: {np.mean(trace_data['y']):.3f}
  Std:  {np.std(trace_data['y']):.3f}
  Min:  {np.min(trace_data['y']):.3f}
  Max:  {np.max(trace_data['y']):.3f}

Movement Statistics:
  Total Distance: {np.sum(speed * dt):.3f}
  Avg Speed: {np.mean(speed):.3f}
  Max Speed: {np.max(speed):.3f}
        """

        ax6.text(
            0.1,
            0.9,
            stats_text,
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Detailed plot saved to {save_path}")

        plt.show()

    def plot_all_traces_comparison(self, save_path: str = None):
        """Plot comparison of all traces in a grid"""
        num_files = len(self.movement_files)
        if num_files == 0:
            print("No movement trace files found!")
            return

        # Calculate grid dimensions
        cols = min(3, num_files)
        rows = (num_files + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i, file_path in enumerate(self.movement_files):
            row = i // cols
            col = i % cols

            trace_data = self.load_movement_trace(file_path)
            self.plot_2d_top_view(trace_data, axes[row, col], show_colliders=False)
            axes[row, col].set_title(f"{os.path.basename(file_path)}", fontsize=10)

        # Hide empty subplots
        for i in range(num_files, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Comparison plot saved to {save_path}")

        plt.show()


def main():
    """Main function to demonstrate the plotting capabilities"""
    print("Real Dataset Plotter")
    print("=" * 50)

    # Create plotter
    plotter = RealDataPlotter()

    if len(plotter.movement_files) == 0:
        print("No movement trace files found in dataset directory!")
        return

    # Plot overview of all traces
    print("\n1. Plotting overview of all movement traces...")
    plotter.plot_all_traces_overview("real_data_overview.png")

    # Plot comparison of all traces
    print("\n2. Plotting comparison of all movement traces...")
    plotter.plot_all_traces_comparison("real_data_comparison.png")

    # Plot detailed analysis of first trace
    if plotter.movement_files:
        print(f"\n3. Plotting detailed analysis of {plotter.movement_files[0]}...")
        plotter.plot_single_trace_detailed(
            plotter.movement_files[0], "real_data_detailed.png"
        )

    print("\nPlotting completed!")


if __name__ == "__main__":
    main()
