import json
import subprocess
import argparse
from pathlib import Path
from itertools import product
import numpy as np

tests = {
    "dataset1": {
        "trace_input": "../../dataset/val/human_data_20251016_204024.json",
        "gt_input": "../../dataset/val/colliders.json",
        "colliders_output": "out/human_data_20251016_204024_colliders.json",
        "metrics_output": "out/metrics_204024.json",
    },
    "dataset2": {
        "trace_input": "../../dataset/train/human_data_20251015_181004.json",
        "gt_input": "../../dataset/train/colliders.json",
        "colliders_output": "out/human_data_20251015_181004_colliders.json",
        "metrics_output": "out/metrics_181004.json",
    }
}


def run_collider_generation(trace_file, grid_size, min_block_size, margin=0.3, max_distance=2.0):
    """Run collider generation with given parameters"""
    cmd = [
        'python', 'gen_colliders.py',
        trace_file,
        '--grid-size', str(grid_size),
        '--min-block-size', str(min_block_size),
        '--margin', str(margin),
        '--max-distance', str(max_distance)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating colliders: {e}")
        print(f"stderr: {e.stderr}")
        return False


def run_evaluation(pred_file, gt_file, output_file, bounds=None):
    """Run evaluation and return metrics"""
    cmd = [
        'python', 'evaluation.py',
        '--pred', pred_file,
        '--gt', gt_file,
        '--output', output_file
    ]

    if bounds:
        cmd.extend(['--bounds'] + [str(b) for b in bounds])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Load metrics
        with open(output_file, 'r') as f:
            metrics = json.load(f)
        return metrics
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        print(f"stderr: {e.stderr}")
        return None


def grid_search(datasets, grid_sizes, min_block_sizes, margins=None, max_distances=None, bounds=None):
    """Perform grid search over hyperparameters

    Args:
        datasets: Dict of dataset configurations
        grid_sizes: List of grid size values to try
        min_block_sizes: List of min block size values to try
        margins: List of margin values to try (optional)
        max_distances: List of max distance values to try (optional)
        bounds: Map bounds for evaluation
    """
    # Default single values if not provided
    if margins is None:
        margins = [0.3]
    if max_distances is None:
        max_distances = [2.0]

    results = []
    total_combinations = (len(grid_sizes) * len(min_block_sizes) *
                          len(margins) * len(max_distances) * len(datasets))
    current = 0

    print(f"Starting grid search with {total_combinations} total evaluations...")
    print(f"Grid sizes: {grid_sizes}")
    print(f"Min block sizes: {min_block_sizes}")
    print(f"Margins: {margins}")
    print(f"Max distances: {max_distances}")
    print(f"Datasets: {list(datasets.keys())}\n")

    # Try all combinations
    for dataset_name, dataset in datasets.items():
        for grid_size, min_block_size, margin, max_distance in product(
                grid_sizes, min_block_sizes, margins, max_distances
        ):
            current += 1
            print(f"[{current}/{total_combinations}] Testing {dataset_name}: "
                  f"grid_size={grid_size}, min_block_size={min_block_size}, "
                  f"margin={margin}, max_distance={max_distance}")

            # Generate colliders
            success = run_collider_generation(
                dataset['trace_input'],
                grid_size,
                min_block_size,
                margin,
                max_distance
            )

            if not success:
                print("  [FAILED] Generation failed, skipping...")
                continue

            # Evaluate
            metrics = run_evaluation(
                dataset['colliders_output'],
                dataset['gt_input'],
                dataset['metrics_output'],
                bounds
            )

            if metrics is None:
                print("  [FAILED] Evaluation failed, skipping...")
                continue

            # Store results
            result = {
                'dataset': dataset_name,
                'grid_size': grid_size,
                'min_block_size': min_block_size,
                'margin': margin,
                'max_distance': max_distance,
                'mIoU': metrics['mIoU'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'cls_acc': metrics['cls_acc']
            }
            results.append(result)

            print(f"  [SUCCESS] mIoU: {metrics['mIoU']:.4f}, "
                  f"F1: {metrics['f1']:.4f}, "
                  f"Acc: {metrics['cls_acc']:.4f}\n")

    return results


def analyze_results(results):
    """Analyze grid search results"""
    if not results:
        print("No results to analyze!")
        return

    # Convert to structured format for analysis
    datasets = {}
    for result in results:
        dataset_name = result['dataset']
        if dataset_name not in datasets:
            datasets[dataset_name] = []
        datasets[dataset_name].append(result)

    print("\n" + "=" * 80)
    print("GRID SEARCH RESULTS")
    print("=" * 80)

    # Best per dataset
    for dataset_name, dataset_results in datasets.items():
        print(f"\n[{dataset_name.upper()}]")
        print("-" * 80)

        # Sort by mIoU
        sorted_results = sorted(dataset_results, key=lambda x: x['mIoU'], reverse=True)

        # Top 5
        print("\nTop 5 configurations:")
        for i, res in enumerate(sorted_results[:5], 1):
            print(f"\n{i}. mIoU: {res['mIoU']:.4f}")
            print(f"   grid_size={res['grid_size']}, min_block_size={res['min_block_size']}, "
                  f"margin={res['margin']}, max_distance={res['max_distance']}")
            print(f"   Precision: {res['precision']:.4f}, Recall: {res['recall']:.4f}, "
                  f"F1: {res['f1']:.4f}, Acc: {res['cls_acc']:.4f}")

    # Overall best (average across datasets)
    print("\n" + "=" * 80)
    print("OVERALL BEST (averaged across datasets)")
    print("=" * 80)

    # Group by parameters
    param_groups = {}
    for result in results:
        key = (result['grid_size'], result['min_block_size'],
               result['margin'], result['max_distance'])
        if key not in param_groups:
            param_groups[key] = []
        param_groups[key].append(result['mIoU'])

    # Calculate average mIoU for each parameter combination
    avg_results = []
    for params, mious in param_groups.items():
        avg_results.append({
            'grid_size': params[0],
            'min_block_size': params[1],
            'margin': params[2],
            'max_distance': params[3],
            'avg_mIoU': np.mean(mious),
            'std_mIoU': np.std(mious)
        })

    # Sort by average mIoU
    avg_results.sort(key=lambda x: x['avg_mIoU'], reverse=True)

    print("\nTop 5 overall configurations:")
    for i, res in enumerate(avg_results[:5], 1):
        print(f"\n{i}. Avg mIoU: {res['avg_mIoU']:.4f} (+/- {res['std_mIoU']:.4f})")
        print(f"   grid_size={res['grid_size']}, min_block_size={res['min_block_size']}, "
              f"margin={res['margin']}, max_distance={res['max_distance']}")


def main():
    parser = argparse.ArgumentParser(description='Grid search for optimal collider generation parameters')
    parser.add_argument('--grid-sizes', nargs='+', type=float,
                        default=[0.05, 0.1, 0.15, 0.2],
                        help='Grid size values to try')
    parser.add_argument('--min-block-sizes', nargs='+', type=float,
                        default=[0.3, 0.5, 0.8, 1.0, 1.5],
                        help='Min block size values to try')
    parser.add_argument('--margins', nargs='+', type=float,
                        default=[0.3],
                        help='Margin values to try')
    parser.add_argument('--max-distances', nargs='+', type=float,
                        default=[2.0],
                        help='Max distance values to try')
    parser.add_argument('--bounds', nargs=4, type=float, default=[-2, 2, -6, 3],
                        metavar=('MIN_X', 'MAX_X', 'MIN_Z', 'MAX_Z'),
                        help='Map bounds for evaluation')
    parser.add_argument('--output', '-o', default='grid_search_results.json',
                        help='Output file for results')

    args = parser.parse_args()

    # Create output directory
    Path('out').mkdir(exist_ok=True)

    # Run grid search
    results = grid_search(
        tests,
        args.grid_sizes,
        args.min_block_sizes,
        args.margins,
        args.max_distances,
        args.bounds
    )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Analyze and display results
    analyze_results(results)


if __name__ == '__main__':
    main()