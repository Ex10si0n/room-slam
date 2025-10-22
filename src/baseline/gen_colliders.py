import json
import argparse
from pathlib import Path
import numpy as np
from shapely.geometry import LineString, box, Point


def load_trace_json(filepath):
    """Load trace data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_grid_mask(trace_data, map_bounds, margin=0.3, grid_size=0.1):
    """Create grid mask where True = blocked by trace"""
    min_x, max_x, min_z, max_z = map_bounds

    # Create grid
    x_bins = int((max_x - min_x) / grid_size)
    z_bins = int((max_z - min_z) / grid_size)

    grid = np.zeros((z_bins, x_bins), dtype=bool)

    # Mark cells intersecting with trace
    points = [(p['x'], p['z']) for p in trace_data]
    line = LineString(points)
    buffered = line.buffer(margin)

    for i in range(z_bins):
        for j in range(x_bins):
            cell_x = min_x + (j + 0.5) * grid_size
            cell_z = min_z + (i + 0.5) * grid_size
            if buffered.contains(Point(cell_x, cell_z)):
                grid[i, j] = True

    return grid, (min_x, min_z, grid_size)


def merge_rectangles(grid, grid_info, min_size=0.5):
    """Find rectangles covering False cells (empty areas) with minimum size"""
    min_x, min_z, grid_size = grid_info
    z_bins, x_bins = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    rectangles = []

    min_cells = int(min_size / grid_size)

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
            if width >= min_size or height >= min_size:
                rectangles.append({
                    'x': (x1 + x2) / 2,
                    'z': (z1 + z2) / 2,
                    'width': width,
                    'height': height
                })

    return rectangles


def find_blocking_rectangles(trace_file, map_bounds, margin=0.3, grid_size=0.1,
                             min_block_size=0.5, y_center=0, y_size=5):
    """Generate blocking colliders using grid-based approach"""
    trace_data = load_trace_json(trace_file)

    # Create grid mask
    grid, grid_info = create_grid_mask(trace_data, map_bounds, margin, grid_size)

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
    parser.add_argument('--bounds', nargs=4, type=float, default=[-2, 2, -6, 3],
                        metavar=('MIN_X', 'MAX_X', 'MIN_Z', 'MAX_Z'),
                        help='Map bounds (default: -2 2 -6 3)')
    parser.add_argument('--margin', type=float, default=0.3, help='Trace buffer margin')
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