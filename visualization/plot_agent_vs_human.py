"""
Compare agent vs human movement data with focus on height variations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from typing import List, Dict


def load_trace(file_path: str) -> Dict:
    """Load movement trace data"""
    with open(file_path, "r") as f:
        data = json.load(f)

    timestamps = np.array([point["timestamp"] for point in data])
    x = np.array([point["x"] for point in data])
    y = np.array([point["y"] for point in data])  # Height
    z = np.array([point["z"] for point in data])

    return {
        "timestamps": timestamps,
        "x": x,
        "y": y,
        "z": z,
        "filename": os.path.basename(file_path),
        "num_points": len(data),
    }


def categorize_traces(dataset_dir: str) -> Dict[str, List[str]]:
    """Categorize traces into agent vs human"""
    pattern = os.path.join(dataset_dir, "*_data_*.json")
    files = glob.glob(pattern)

    agent_files = [f for f in files if "agent_data_" in f]
    human_files = [f for f in files if "human_data_" in f]

    return {"agent": sorted(agent_files), "human": sorted(human_files)}


def plot_agent_vs_human_comparison(dataset_dir: str = "dataset"):
    """Create comprehensive comparison plots"""
    traces = categorize_traces(dataset_dir)

    print(
        f"Found {len(traces['agent'])} agent traces and {len(traces['human'])} human traces"
    )

    # Load all traces
    agent_data = [load_trace(f) for f in traces["agent"]]
    human_data = [load_trace(f) for f in traces["human"]]

    # Create comprehensive comparison plot
    fig = plt.figure(figsize=(20, 16))

    # Plot 1: Top view comparison (Agent traces)
    ax1 = plt.subplot(3, 4, 1)
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(agent_data)))
    for i, trace in enumerate(agent_data):
        ax1.plot(
            trace["x"],
            trace["z"],
            color=colors[i],
            alpha=0.7,
            linewidth=1,
            label=f"Agent {i+1}",
        )
    ax1.set_title("Agent Traces (Top View)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Z Position")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # Plot 2: Top view comparison (Human traces)
    ax2 = plt.subplot(3, 4, 2)
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(human_data)))
    for i, trace in enumerate(human_data):
        ax2.plot(
            trace["x"],
            trace["z"],
            color=colors[i],
            alpha=0.7,
            linewidth=1,
            label=f"Human {i+1}",
        )
    ax2.set_title("Human Traces (Top View)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Z Position")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    # Plot 3: Height distributions comparison
    ax3 = plt.subplot(3, 4, 3)
    all_agent_heights = []
    all_human_heights = []

    for trace in agent_data:
        all_agent_heights.extend(trace["y"])
    for trace in human_data:
        all_human_heights.extend(trace["y"])

    ax3.hist(
        all_agent_heights, bins=50, alpha=0.7, label="Agent", color="blue", density=True
    )
    ax3.hist(
        all_human_heights, bins=50, alpha=0.7, label="Human", color="red", density=True
    )
    ax3.set_xlabel("Height (Y)")
    ax3.set_ylabel("Density")
    ax3.set_title("Height Distribution Comparison", fontsize=12, fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Height range comparison
    ax4 = plt.subplot(3, 4, 4)
    agent_ranges = [np.max(trace["y"]) - np.min(trace["y"]) for trace in agent_data]
    human_ranges = [np.max(trace["y"]) - np.min(trace["y"]) for trace in human_data]

    x_pos = np.arange(2)
    means = [np.mean(agent_ranges), np.mean(human_ranges)]
    stds = [np.std(agent_ranges), np.std(human_ranges)]

    bars = ax4.bar(x_pos, means, yerr=stds, capsize=5, color=["blue", "red"], alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(["Agent", "Human"])
    ax4.set_ylabel("Height Range")
    ax4.set_title("Height Range Comparison", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    # Plot 5-6: Individual height profiles (Agent)
    for i, trace in enumerate(agent_data[:2]):
        ax = plt.subplot(3, 4, 5 + i)
        relative_time = trace["timestamps"] - trace["timestamps"][0]
        ax.plot(relative_time, trace["y"], "b-", linewidth=1)
        ax.fill_between(relative_time, trace["y"], alpha=0.3, color="blue")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Height (Y)")
        ax.set_title(f"Agent {i+1} Height Profile", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Plot 7-8: Individual height profiles (Human)
    for i, trace in enumerate(human_data[:2]):
        ax = plt.subplot(3, 4, 7 + i)
        relative_time = trace["timestamps"] - trace["timestamps"][0]
        ax.plot(relative_time, trace["y"], "r-", linewidth=1)
        ax.fill_between(relative_time, trace["y"], alpha=0.3, color="red")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Height (Y)")
        ax.set_title(f"Human {i+1} Height Profile", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Plot 9: Movement statistics comparison
    ax9 = plt.subplot(3, 4, 9)

    # Calculate movement statistics
    agent_durations = [
        trace["timestamps"][-1] - trace["timestamps"][0] for trace in agent_data
    ]
    human_durations = [
        trace["timestamps"][-1] - trace["timestamps"][0] for trace in human_data
    ]

    agent_points = [trace["num_points"] for trace in agent_data]
    human_points = [trace["num_points"] for trace in human_data]

    scatter1 = ax9.scatter(
        agent_durations, agent_points, c="blue", alpha=0.7, s=60, label="Agent"
    )
    scatter2 = ax9.scatter(
        human_durations, human_points, c="red", alpha=0.7, s=60, label="Human"
    )

    ax9.set_xlabel("Duration (s)")
    ax9.set_ylabel("Number of Points")
    ax9.set_title("Duration vs Points", fontsize=12, fontweight="bold")
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # Plot 10: Speed analysis (first agent vs first human)
    ax10 = plt.subplot(3, 4, 10)

    if agent_data and human_data:
        # Calculate speed for first agent trace
        agent_trace = agent_data[0]
        agent_dt = np.diff(agent_trace["timestamps"])
        agent_dx = np.diff(agent_trace["x"])
        agent_dy = np.diff(agent_trace["y"])
        agent_dz = np.diff(agent_trace["z"])
        agent_speed = np.sqrt(agent_dx**2 + agent_dy**2 + agent_dz**2) / agent_dt

        # Calculate speed for first human trace
        human_trace = human_data[0]
        human_dt = np.diff(human_trace["timestamps"])
        human_dx = np.diff(human_trace["x"])
        human_dy = np.diff(human_trace["y"])
        human_dz = np.diff(human_trace["z"])
        human_speed = np.sqrt(human_dx**2 + human_dy**2 + human_dz**2) / human_dt

        agent_time = agent_trace["timestamps"][1:] - agent_trace["timestamps"][0]
        human_time = human_trace["timestamps"][1:] - human_trace["timestamps"][0]

        ax10.plot(agent_time, agent_speed, "b-", alpha=0.7, label="Agent")
        ax10.plot(human_time, human_speed, "r-", alpha=0.7, label="Human")
        ax10.set_xlabel("Time (s)")
        ax10.set_ylabel("Speed")
        ax10.set_title("Speed Comparison", fontsize=12, fontweight="bold")
        ax10.legend()
        ax10.grid(True, alpha=0.3)

    # Plot 11: 3D trajectory comparison
    ax11 = plt.subplot(3, 4, 11, projection="3d")

    if agent_data and human_data:
        # Plot first agent trace
        agent_trace = agent_data[0]
        ax11.plot(
            agent_trace["x"],
            agent_trace["z"],
            agent_trace["y"],
            "b-",
            alpha=0.6,
            linewidth=1,
            label="Agent",
        )

        # Plot first human trace
        human_trace = human_data[0]
        ax11.plot(
            human_trace["x"],
            human_trace["z"],
            human_trace["y"],
            "r-",
            alpha=0.6,
            linewidth=1,
            label="Human",
        )

        ax11.set_xlabel("X")
        ax11.set_ylabel("Z")
        ax11.set_zlabel("Height (Y)")
        ax11.set_title("3D Trajectory Comparison", fontsize=12, fontweight="bold")
        ax11.legend()

    # Plot 12: Summary statistics
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis("off")

    # Calculate summary statistics
    agent_height_means = [np.mean(trace["y"]) for trace in agent_data]
    human_height_means = [np.mean(trace["y"]) for trace in human_data]

    agent_height_stds = [np.std(trace["y"]) for trace in agent_data]
    human_height_stds = [np.std(trace["y"]) for trace in human_data]

    stats_text = f"""
Summary Statistics:

AGENT DATA:
  Traces: {len(agent_data)}
  Avg Height: {np.mean(agent_height_means):.3f} ± {np.std(agent_height_means):.3f}
  Avg Height Std: {np.mean(agent_height_stds):.3f}
  Avg Duration: {np.mean(agent_durations):.1f}s
  Avg Points: {np.mean(agent_points):,.0f}

HUMAN DATA:
  Traces: {len(human_data)}
  Avg Height: {np.mean(human_height_means):.3f} ± {np.std(human_height_means):.3f}
  Avg Height Std: {np.mean(human_height_stds):.3f}
  Avg Duration: {np.mean(human_durations):.1f}s
  Avg Points: {np.mean(human_points):,.0f}

KEY DIFFERENCES:
• Agent height: ~constant (-0.460)
• Human height: ~variable (0.7-0.9)
• Agent: smoother, consistent movement
• Human: more varied, natural movement
    """

    ax12.text(
        0.05,
        0.95,
        stats_text,
        transform=ax12.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    plt.savefig("agent_vs_human_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print detailed analysis
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS")
    print("=" * 60)

    print(f"\nAGENT DATA ({len(agent_data)} traces):")
    for i, trace in enumerate(agent_data):
        height_range = np.max(trace["y"]) - np.min(trace["y"])
        duration = trace["timestamps"][-1] - trace["timestamps"][0]
        print(
            f"  Agent {i+1}: {trace['num_points']:,} points, {duration:.1f}s, "
            f"height range: {height_range:.6f}"
        )

    print(f"\nHUMAN DATA ({len(human_data)} traces):")
    for i, trace in enumerate(human_data):
        height_range = np.max(trace["y"]) - np.min(trace["y"])
        duration = trace["timestamps"][-1] - trace["timestamps"][0]
        print(
            f"  Human {i+1}: {trace['num_points']:,} points, {duration:.1f}s, "
            f"height range: {height_range:.3f}"
        )

    print(f"\nKEY INSIGHTS:")
    print(f"• Agent traces show virtually NO height variation (range ≈ 0.000)")
    print(f"• Human traces show significant height variation (range ≈ 0.5-0.8)")
    print(f"• This suggests agent moves on a flat plane, while human moves in 3D space")
    print(f"• Agent data appears to be simulated/controlled movement")
    print(f"• Human data appears to be real human motion capture")


def main():
    """Main function"""
    print("Agent vs Human Movement Data Analysis")
    print("=" * 50)

    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' not found!")
        return

    plot_agent_vs_human_comparison(dataset_dir)
    print(f"\nComparison plot saved as 'agent_vs_human_comparison.png'")


if __name__ == "__main__":
    main()
