import json
import argparse
from pathlib import Path
import numpy as np
from shapely.geometry import LineString, box, Point


def load_trace_json(filepath):
    """Load trace data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_trace_bounds(trace_data, padding=1.0):
    """Get bounding box around trace with padding"""
    xs = [p['x'] for p in trace_data]
    zs = [p['z'] for p in trace_data]
    return (min(xs) - padding, max(xs) + padding,
            min(zs) - padding, max(zs) + padding)


def create_grid_mask(trace_data, map_bounds, margin=0.3, max_distance=None, grid_size=0.1):
    """Create grid mask where True = blocked by trace or too far from trace"""
    min_x, max_x, min_z, max_z = map_bounds

    # Create grid
    x_bins = int((max_x - min_x) / grid_size)
    z_bins = int((max_z - min_z) / grid_size)

    grid = np.zeros((z_bins, x_bins), dtype=bool)

    # Mark cells intersecting with trace
    points = [(p['x'], p['z']) for p in trace_data]
    line = LineString(points)
    buffered = line.buffer(margin)

    # Create distance limit buffer if specified
    if max_distance:
        valid_region = line.buffer(max_distance)

    for i in range(z_bins):
        for j in range(x_bins):
            cell_x = min_x + (j + 0.5) * grid_size
            cell_z = min_z + (i + 0.5) * grid_size
            point = Point(cell_x, cell_z)

            # Mark as occupied if:
            # 1. Inside trace buffer (walkable area)
            if buffered.contains(point):
                grid[i, j] = True
            # 2. OR outside max_distance from trace (ignore these areas)
            elif max_distance and not valid_region.contains(point):
                grid[i, j] = True

    return grid, (min_x, min_z, grid_size)


def merge_rectangles(grid, grid_info, min_block_size=0.5):
    """Find rectangles covering False cells (blocked areas)"""
    min_x, min_z, grid_size = grid_info
    z_bins, x_bins = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    rectangles = []

    min_cells = int(min_block_size / grid_size)

    for i in range(z_bins):
        for j in range(x_bins):
            if grid[i, j] or visited[i, j]:
                continue

            # Find maximal rectangle starting at (i, j)
            max_width = x_bins - j
            for w in range(max_width):
                if j + w >= x_bins or grid[i, j + w]:
                    max_width = w
                    break

            max_height = 1
            for h in range(1, z_bins - i):
                valid = True
                for w in range(max_width):
                    if grid[i + h, j + w] or visited[i + h, j + w]:
                        valid = False
                        break
                if not valid:
                    break
                max_height = h + 1

            # Skip if too small
            if max_width < min_cells and max_height < min_cells:
                visited[i, j] = True
                continue

            # Mark as visited
            for di in range(max_height):
                for dj in range(max_width):
                    visited[i + di, j + dj] = True

            # Convert to world coordinates
            x1 = min_x + j * grid_size
            z1 = min_z + i * grid_size
            x2 = min_x + (j + max_width) * grid_size
            z2 = min_z + (i + max_height) * grid_size

            width = x2 - x1
            height = z2 - z1

            # Filter by actual size
            if width >= min_block_size or height >= min_block_size:
                rectangles.append({
                    'x': (x1 + x2) / 2,
                    'z': (z1 + z2) / 2,
                    'width': width,
                    'height': height
                })

    return rectangles


def find_blocking_rectangles(trace_file, map_bounds=None, margin=0.3,
                             max_distance=2.0, grid_size=0.1,
                             min_block_size=0.5, y_center=0, y_size=5):
    """Generate blocking colliders using grid-based approach

    Args:
        trace_file: Path to trace JSON
        map_bounds: Optional (min_x, max_x, min_z, max_z). If None, auto-calculate from trace
        margin: Buffer around trace (walkable area)
        max_distance: Maximum distance from trace to generate blocks. Blocks beyond this are ignored.
        grid_size: Grid cell size
        min_block_size: Minimum block size
        y_center: Y coordinate for colliders
        y_size: Height of colliders
    """
    trace_data = load_trace_json(trace_file)

    # Auto-calculate bounds if not provided
    if map_bounds is None:
        map_bounds = get_trace_bounds(trace_data, padding=max_distance + 1.0)

    # Create grid mask
    grid, grid_info = create_grid_mask(trace_data, map_bounds, margin, max_distance, grid_size)

    # Find rectangles with minimum size constraint
    rectangles = merge_rectangles(grid, grid_info, min_block_size)

    # Convert to colliders
    colliders = []
    for rect in rectangles:
        colliders.append({
            "type": "BoxCollider",
            "label": "BLOCK",
            "center": {"x": rect['x'], "y": y_center, "z": rect['z']},
            "size": {"x": rect['width'], "y": y_size, "z": rect['height']}
        })

    return colliders


def main():
    parser = argparse.ArgumentParser(description='Generate blocking colliders from trace')
    parser.add_argument('trace', help='Input trace JSON file')
    parser.add_argument('--bounds', nargs=4, type=float, default=None,
                        metavar=('MIN_X', 'MAX_X', 'MIN_Z', 'MAX_Z'),
                        help='Map bounds (default: auto from trace)')
    parser.add_argument('--margin', type=float, default=0.3, help='Trace buffer margin')
    parser.add_argument('--max-distance', type=float, default=2.0,
                        help='Max distance from trace to generate blocks')
    parser.add_argument('--grid-size', type=float, default=0.1, help='Grid cell size')
    parser.add_argument('--min-block-size', type=float, default=0.5,
                        help='Minimum block size (width or height)')
    parser.add_argument('--y-center', type=float, default=0, help='Collider Y center')
    parser.add_argument('--y-size', type=float, default=5, help='Collider Y size')

    args = parser.parse_args()

    # Generate output path
    trace_path = Path(args.trace)
    output_dir = Path('out')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{trace_path.stem}_colliders.json"

    # Generate colliders
    colliders = find_blocking_rectangles(
        args.trace,
        map_bounds=args.bounds,
        margin=args.margin,
        max_distance=args.max_distance,
        grid_size=args.grid_size,
        min_block_size=args.min_block_size,
        y_center=args.y_center,
        y_size=args.y_size
    )

    # Save output
    with open(output_file, 'w') as f:
        json.dump({"colliders": colliders}, f, indent=2)

    print(f"Generated {len(colliders)} colliders -> {output_file}")


if __name__ == '__main__':
    main()