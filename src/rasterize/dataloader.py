import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple
from scipy.ndimage import distance_transform_edt


class TraceColliderDataset(Dataset):
    """
    Dataset for trace sequences and collider ground truth.

    Data format (shared-collider style):
      - Trace files: agent_data_*.json / human_data_*.json (list of 3D points)
      - One colliders.json in the same folder (list of collider dicts)

    This dataset:
      - Applies geometric augmentations to both traces and colliders
      - Computes kinematic features on traces (velocity, acceleration, speed)
      - Optionally rasterizes traces into a top-down grid (Walk2Map-style)
    """

    def __init__(
            self,
            data_dir: str,
            max_trace_len: int = 3000,
            max_colliders: int = 50,
            augment_rotation: bool = True,
            augment_translation: bool = True,
            augment_scale: bool = True,
            augment_collider_dropout: bool = True,
            rotation_angles: list = [0, 90, 180, 270],
            scale_range: tuple = (0.8, 1.2),
            translation_range: float = 1.0,
            collider_dropout_prob: float = 0.2,
            use_grid: bool = True,
            grid_size: int = 64,
    ):
        """
        Args:
            data_dir: Directory containing trace JSON files + colliders.json
            max_trace_len: Maximum number of trace points per sample
            max_colliders: Maximum number of colliders per scene (padding)
            augment_rotation: If True, apply rotation augmentation
            augment_translation: If True, apply random translation
            augment_scale: If True, apply random scaling
            augment_collider_dropout: If True, randomly drop some colliders
            rotation_angles: List of rotation angles in degrees
            scale_range: (min_scale, max_scale) for random scaling
            translation_range: Maximum translation magnitude (meters)
            collider_dropout_prob: Probability of dropping a non-wall collider
            use_grid: If True, rasterize traces into a 2D grid
            grid_size: Size of square grid (H = W = grid_size)
        """
        # Resolve data directory path
        script_dir = Path(__file__).parent
        data_path = Path(data_dir)

        if data_path.is_absolute():
            self.data_dir = data_path
        else:
            # Resolve relative to the script directory
            self.data_dir = (script_dir / data_path).resolve()

        if not self.data_dir.exists():
            raise ValueError(
                f"Data directory does not exist: {self.data_dir}\n"
                f"Original path: {data_dir}\n"
                f"Script directory: {script_dir}\n"
                f"Current working directory: {Path.cwd()}"
            )
        if not self.data_dir.is_dir():
            raise ValueError(f"Data directory is not a directory: {self.data_dir}")

        print(f"Data directory resolved: {data_dir} -> {self.data_dir}")

        self.max_trace_len = max_trace_len
        self.max_colliders = max_colliders

        # Augmentation settings
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation
        self.augment_scale = augment_scale
        self.augment_collider_dropout = augment_collider_dropout
        self.rotation_angles = rotation_angles
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.collider_dropout_prob = collider_dropout_prob

        # Grid (Walk2Map-style) settings
        self.use_grid = use_grid
        self.grid_size = grid_size

        # Label mapping (3 collider classes)
        self.label_to_id = {
            'BLOCK': 0,
            'LOW': 1,
            'MID': 2,
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        # Load (trace, collider) file pairs
        self.base_data_pairs = self._load_data_pairs()
        print(f"Found {len(self.base_data_pairs)} base samples in {data_dir}")

        # Expand dataset by rotation augmentation (duplicate references with angle tag)
        if self.augment_rotation:
            self.data_pairs = []
            for pair in self.base_data_pairs:
                for angle in self.rotation_angles:
                    self.data_pairs.append({
                        'trace': pair['trace'],
                        'collider': pair['collider'],
                        'rotation': angle
                    })
            print(f"Augmented to {len(self.data_pairs)} samples with rotations: {rotation_angles}°")
        else:
            self.data_pairs = [
                {'trace': p['trace'], 'collider': p['collider'], 'rotation': 0}
                for p in self.base_data_pairs
            ]

        if len(self.data_pairs) == 0:
            raise ValueError(f"No valid data files found in {data_dir}")

    def _rasterize_traces_to_grid(self, traces: List[Dict]) -> torch.Tensor:
        """
        Rasterize 3D traces into a top-down 2D grid.

        This follows the spirit of Walk2Map:
          - Use (x, z) coordinates of the trajectory
          - Normalize to [0, 1] and map to a fixed-size grid
          - Compute distance-transform to the trace
          - Invert distances so that cells closer to the trace have higher values

        Returns:
            Tensor of shape [1, H, W] with values in [0, 1]
        """
        H = W = self.grid_size

        if len(traces) == 0:
            return torch.zeros((1, H, W), dtype=torch.float32)

        xs = np.array([p['x'] for p in traces], dtype=np.float32)
        zs = np.array([p['z'] for p in traces], dtype=np.float32)

        # Avoid degenerate width/height
        eps = 1e-3
        min_x, max_x = xs.min(), xs.max()
        min_z, max_z = zs.min(), zs.max()
        width = max(max_x - min_x, eps)
        height = max(max_z - min_z, eps)

        # Normalize trajectory into a unit square (same aspect ratio)
        scale = max(width, height)
        nx = (xs - min_x) / scale  # 0 ~ 1
        nz = (zs - min_z) / scale  # 0 ~ 1

        # Map to discrete grid indices
        ix = np.clip((nx * (W - 1)).astype(np.int32), 0, W - 1)
        iz = np.clip((nz * (H - 1)).astype(np.int32), 0, H - 1)

        # "free" map: True = no trace, False = trajectory cell
        free = np.ones((H, W), dtype=bool)
        free[iz, ix] = False

        # Distance transform: each cell gets distance to nearest trace cell
        dist = distance_transform_edt(free)
        if dist.max() > 0:
            dist = dist / dist.max()

        # Invert distance: closer to trace = higher value
        inv = 1.0 - dist  # [H, W]

        return torch.from_numpy(inv.astype(np.float32)).unsqueeze(0)  # [1, H, W]

    def _load_data_pairs(self) -> List[Dict[str, Path]]:
        """
        Load and pair trace and collider files.

        Supported pattern:
          - shared colliders.json in the same directory
          - multiple trace files:
              agent_data_*.json
              human_data_*.json

        Returns:
            List of dicts with keys:
              - 'trace': Path to trace file
              - 'collider': Path to shared colliders.json
        """
        pairs = []

        shared_collider = self.data_dir / "colliders.json"

        if shared_collider.exists():
            # All agent/human trajectories share the same collider file
            agent_files = sorted(self.data_dir.glob("agent_data_*.json"))
            human_files = sorted(self.data_dir.glob("human_data_*.json"))

            all_trace_files = agent_files + human_files

            for trace_file in all_trace_files:
                pairs.append({
                    'trace': trace_file,
                    'collider': shared_collider
                })

            if len(all_trace_files) > 0:
                print(f"Using shared colliders.json for {len(all_trace_files)} trace files")
            else:
                print(
                    f"Warning: Found colliders.json but no agent_data_*.json or human_data_*.json files in {self.data_dir}"
                )
        else:
            print(f"Warning: No colliders.json found in {self.data_dir}")
            all_json_files = list(self.data_dir.glob("*.json"))
            if all_json_files:
                print(f"Found {len(all_json_files)} JSON files: {[f.name for f in all_json_files[:5]]}")
            else:
                print(f"No JSON files found in {self.data_dir}")

        return pairs

    def _rotate_traces(self, traces: List[Dict], angle_degrees: float) -> List[Dict]:
        """
        Rotate traces around Y-axis (vertical axis) by the given angle in degrees.
        Only (x, z) are rotated, y is kept as-is.
        """
        angle_rad = np.radians(angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        rotated_traces = []
        for point in traces:
            x, y, z = point['x'], point['y'], point['z']

            x_new = cos_a * x + sin_a * z
            z_new = -sin_a * x + cos_a * z

            rotated_traces.append({
                'x': x_new,
                'y': y,
                'z': z_new,
                'timestamp': point['timestamp']
            })

        return rotated_traces

    def _rotate_colliders(self, colliders: List[Dict], angle_degrees: float) -> List[Dict]:
        """
        Rotate axis-aligned box colliders around Y-axis by the given angle.

        For 90°/270° rotations, we swap the x/z sizes to keep boxes aligned.
        """
        angle_rad = np.radians(angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        rotated_colliders = []
        for col in colliders:
            center = col['center']
            size = col['size']

            cx, cy, cz = center['x'], center['y'], center['z']
            sx, sy, sz = size['x'], size['y'], size['z']

            # Rotate center position
            cx_new = cos_a * cx + sin_a * cz
            cz_new = -sin_a * cx + cos_a * cz

            # Rotate box dimensions (swap x/z for 90° and 270°)
            if angle_degrees == 90 or angle_degrees == 270:
                sx_new, sz_new = sz, sx
            else:
                sx_new, sz_new = sx, sz

            rotated_colliders.append({
                'type': col.get('type', 'BoxCollider'),
                'label': col.get('label', 'BLOCK'),
                'center': {
                    'x': float(cx_new),
                    'y': float(cy),
                    'z': float(cz_new)
                },
                'size': {
                    'x': float(sx_new),
                    'y': float(sy),
                    'z': float(sz_new)
                },
                'radius': col.get('radius', 0.0),
                'height': col.get('height', 0.0)
            })

        return rotated_colliders

    def _translate_traces(self, traces: List[Dict], tx: float, tz: float) -> List[Dict]:
        """Translate traces in the X-Z plane by (tx, tz)."""
        return [{
            'x': p['x'] + tx,
            'y': p['y'],
            'z': p['z'] + tz,
            'timestamp': p['timestamp']
        } for p in traces]

    def _translate_colliders(self, colliders: List[Dict], tx: float, tz: float) -> List[Dict]:
        """Translate colliders in the X-Z plane by (tx, tz)."""
        translated = []
        for col in colliders:
            new_col = col.copy()
            new_col['center'] = {
                'x': col['center']['x'] + tx,
                'y': col['center']['y'],
                'z': col['center']['z'] + tz
            }
            translated.append(new_col)
        return translated

    def _scale_traces(self, traces: List[Dict], scale: float) -> List[Dict]:
        """Uniformly scale traces (all coordinates) by 'scale'."""
        return [{
            'x': p['x'] * scale,
            'y': p['y'] * scale,
            'z': p['z'] * scale,
            'timestamp': p['timestamp']
        } for p in traces]

    def _scale_colliders(self, colliders: List[Dict], scale: float) -> List[Dict]:
        """Uniformly scale colliders by 'scale'."""
        scaled = []
        for col in colliders:
            scaled.append({
                'type': col.get('type', 'BoxCollider'),
                'label': col.get('label', 'BLOCK'),
                'center': {
                    'x': col['center']['x'] * scale,
                    'y': col['center']['y'] * scale,
                    'z': col['center']['z'] * scale
                },
                'size': {
                    'x': col['size']['x'] * scale,
                    'y': col['size']['y'] * scale,
                    'z': col['size']['z'] * scale
                },
                'radius': col.get('radius', 0.0) * scale,
                'height': col.get('height', 0.0) * scale
            })
        return scaled

    def _dropout_colliders(self, colliders: List[Dict], dropout_prob: float) -> List[Dict]:
        """
        Randomly drop colliders to force the model to rely on the trace.

        We never drop "wall-like" colliders (large BLOCKs), so that room
        structure remains roughly intact.
        """
        kept_colliders = []
        for col in colliders:
            # Heuristic: treat large BLOCKs as walls
            is_wall = (
                col.get('label') == 'BLOCK' and
                (col['size'].get('x', 0) > 5.0 or
                 col['size'].get('z', 0) > 5.0)
            )

            if is_wall or np.random.rand() > dropout_prob:
                kept_colliders.append(col)

        # Always keep at least one collider to avoid degenerate supervision
        return kept_colliders if kept_colliders else colliders

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with geometric/temporal augmentations.

        Returns a dict with:
          - traces: [N, 11]     (position, time, kinematics)
          - trace_mask: [N]     (all True, padded later in collate_fn)
          - boxes: [M, 6]       (cx, cy, cz, sx, sy, sz)
          - labels: [M]         (class IDs)
          - valid_mask: [M]     (True for valid colliders, False for padding)
          - num_traces: scalar
          - num_colliders: scalar
          - filename: str       (original trace filename, for debugging)
          - rotation: scalar    (applied rotation angle)
          - grid_map: [1, H, W] (Walk2Map-style rasterized trace)
        """
        pair = self.data_pairs[idx]
        rotation_angle = pair['rotation']

        # Load trace data
        with open(pair['trace'], 'r') as f:
            trace_data = json.load(f)

        # Load collider data
        with open(pair['collider'], 'r') as f:
            collider_data = json.load(f)

        # Extract traces and colliders from raw JSON
        traces = trace_data if isinstance(trace_data, list) else []
        colliders = collider_data.get('colliders', [])

        # Geometric augmentation parameters
        tx, tz = 0.0, 0.0
        scale = 1.0

        # 1) Rotation augmentation
        if rotation_angle != 0:
            traces = self._rotate_traces(traces, rotation_angle)
            colliders = self._rotate_colliders(colliders, rotation_angle)

        # 2) Random translation
        if self.augment_translation:
            tx = np.random.uniform(-self.translation_range, self.translation_range)
            tz = np.random.uniform(-self.translation_range, self.translation_range)
            traces = self._translate_traces(traces, tx, tz)
            colliders = self._translate_colliders(colliders, tx, tz)

        # 3) Random scaling
        if self.augment_scale:
            scale = np.random.uniform(*self.scale_range)
            traces = self._scale_traces(traces, scale)
            colliders = self._scale_colliders(colliders, scale)

        # 4) Random sequence reversal
        if np.random.rand() < 0.5:
            traces = list(reversed(traces))

        # 5) Add small Gaussian noise to positions
        if np.random.rand() < 0.8:
            for p in traces:
                p['x'] += np.random.normal(0, 0.02)
                p['y'] += np.random.normal(0, 0.01)
                p['z'] += np.random.normal(0, 0.02)

        # 6) Random subsequence cropping
        if len(traces) > 100 and np.random.rand() < 0.5:
            start = np.random.randint(0, int(0.2 * len(traces)))
            end = np.random.randint(int(0.8 * len(traces)), len(traces))
            traces = traces[start:end]

        # 7) Simple time-warping (two-piece scaling on timestamps)
        if np.random.rand() < 0.5 and len(traces) > 0:
            t = np.array([p['timestamp'] for p in traces], dtype=np.float32)
            t = t - t.min()
            k = np.random.uniform(0.4, 0.6)
            s1, s2 = np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.5)
            t_max = t.max() + 1e-6
            mask_t = (t / t_max < k)
            t[mask_t] *= s1
            t[~mask_t] = k * s1 + (t[~mask_t] - k * t_max) * s2
            for i, p in enumerate(traces):
                p['timestamp'] = float(t[i])

        # 8) Random collider dropout (simulate missing labels)
        if self.augment_collider_dropout and np.random.rand() < 0.5:
            colliders = self._dropout_colliders(colliders, self.collider_dropout_prob)

        # Convert traces and colliders to tensors
        trace_array = self._process_traces(traces)
        collider_boxes, collider_labels, collider_valid = self._process_colliders(colliders)

        # Rasterize traces into a Walk2Map-style grid
        if self.use_grid:
            grid_map = self._rasterize_traces_to_grid(traces)  # [1, H, W]
        else:
            grid_map = torch.zeros((1, self.grid_size, self.grid_size), dtype=torch.float32)

        return {
            'traces': trace_array,
            'trace_mask': torch.ones(len(trace_array), dtype=torch.bool),
            'boxes': collider_boxes,
            'labels': collider_labels,
            'valid_mask': collider_valid,
            'num_traces': torch.tensor(len(traces), dtype=torch.long),
            'num_colliders': torch.tensor(len(colliders), dtype=torch.long),
            'filename': f"{pair['trace'].name}_rot{rotation_angle}",
            'rotation': torch.tensor(rotation_angle, dtype=torch.float32),
            'grid_map': grid_map,
        }

    def _process_traces(self, traces: List[Dict]) -> torch.Tensor:
        """
        Convert raw trace list into a tensor of shape [N, 7]:
          - (x, z, t)  # 2D position + time
          - (vx, vz)   # 2D velocity
          - (ax, az)   # 2D acceleration
          - speed (2D)
        """
        FEAT_DIM = 7

        if len(traces) == 0:
            return torch.zeros((1, FEAT_DIM), dtype=torch.float32)

        # Collect x, z, timestamp
        trace_list = []
        for p in traces:
            trace_list.append([
                p.get('x', 0.0),
                p.get('z', 0.0),
                p.get('timestamp', 0.0)
            ])

        trace_array = np.array(trace_list, dtype=np.float32)  # [N, 3]

        # Sort by timestamp
        if trace_array.shape[0] > 1:
            order = np.argsort(trace_array[:, 2])  # timestamp at index 2
            trace_array = trace_array[order]

        # Normalize time
        if trace_array.shape[0] > 0:
            trace_array[:, 2] -= trace_array[0, 2]

        # Compute 2D kinematics (only x, z)
        diffs = np.diff(trace_array, axis=0, prepend=trace_array[[0], :])  # [N, 3]
        dt = np.clip(diffs[:, 2], 1e-3, None)  # [N]

        # Velocity: only x and z components
        vel = diffs[:, :2] / dt[:, None]  # [N, 2]
        vel = np.clip(vel, -10.0, 10.0)

        # Acceleration: derivative of velocity
        vel_diffs = np.diff(vel, axis=0, prepend=vel[[0], :])  # [N, 2]
        acc = vel_diffs / dt[:, None]  # [N, 2]
        acc = np.clip(acc, -20.0, 20.0)

        # 2D speed (magnitude of 2D velocity)
        speed = np.linalg.norm(vel, axis=1, keepdims=True)  # [N, 1]

        kin = np.concatenate([vel, acc, speed], axis=1)  # [N, 5]
        # Only keep x, z (not timestamp) + kinematics
        trace_array = np.concatenate([trace_array[:, :2], kin], axis=1)  # [N, 2+5=7]

        # Downsample if too long
        if len(trace_array) > self.max_trace_len:
            indices = np.linspace(0, len(trace_array) - 1, self.max_trace_len, dtype=int)
            trace_array = trace_array[indices]

        return torch.from_numpy(trace_array.astype(np.float32))

    def _process_colliders(
            self,
            colliders: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert collider list into padded tensors.

        Returns:
            boxes: [M, 4]  (cx, cz, sx, sz) - 2D boxes
            labels: [M]    (class IDs, -1 for padding)
            valid_mask: [M] (True for valid colliders, False for padding)
        """
        boxes = torch.zeros((self.max_colliders, 4), dtype=torch.float32)
        labels = torch.full((self.max_colliders,), -1, dtype=torch.long)
        valid_mask = torch.zeros(self.max_colliders, dtype=torch.bool)

        if len(colliders) == 0:
            return boxes, labels, valid_mask

        num_valid = min(len(colliders), self.max_colliders)

        for i, col in enumerate(colliders[:num_valid]):
            center = col.get('center', {})
            size = col.get('size', {})

            cx = center.get('x', 0.0)
            cz = center.get('z', 0.0)

            sx = size.get('x', 0.0)
            sz = size.get('z', 0.0)

            label_str = col.get('label', 'BLOCK')
            label_id = self.label_to_id.get(label_str, 0)

            boxes[i] = torch.tensor([cx, cz, sx, sz], dtype=torch.float32)
            labels[i] = label_id
            valid_mask[i] = True

        return boxes, labels, valid_mask


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length traces.

    - Pads traces in time dimension to the max length in the batch
    - Stacks colliders and grid maps
    """
    # Max time length in this batch
    max_len = max(item['traces'].shape[0] for item in batch)

    traces_padded = []
    masks = []
    grids = []

    for item in batch:
        trace = item['traces']          # [N, F]
        feat_dim = trace.shape[1]
        pad_len = max_len - trace.shape[0]

        grids.append(item['grid_map'])  # [1, H, W]

        if pad_len > 0:
            trace = torch.cat([
                trace,
                torch.zeros((pad_len, feat_dim), dtype=torch.float32)
            ], dim=0)

            mask = torch.cat([
                torch.ones(item['traces'].shape[0], dtype=torch.bool),
                torch.zeros(pad_len, dtype=torch.bool)
            ])
        else:
            mask = torch.ones(trace.shape[0], dtype=torch.bool)

        traces_padded.append(trace)
        masks.append(mask)

    batched = {
        'traces': torch.stack(traces_padded),           # [B, T, F]
        'trace_mask': torch.stack(masks),               # [B, T]
        'boxes': torch.stack([item['boxes'] for item in batch]),          # [B, M, 6]
        'labels': torch.stack([item['labels'] for item in batch]),        # [B, M]
        'valid_mask': torch.stack([item['valid_mask'] for item in batch]),# [B, M]
        'num_traces': torch.stack([item['num_traces'] for item in batch]),
        'num_colliders': torch.stack([item['num_colliders'] for item in batch]),
        # Batched grid maps: [B, 1, H, W]
        'grid_maps': torch.stack(grids),
    }
    return batched


def create_dataloader(
        data_dir: str,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
        max_trace_len: int = 3000,
        max_colliders: int = 50,
        augment_rotation: bool = True,
        augment_translation: bool = True,
        augment_scale: bool = True,
        augment_collider_dropout: bool = True,
        rotation_angles: list = [0, 90, 180, 270],
        scale_range: tuple = (0.8, 1.2),
        translation_range: float = 1.0,
        collider_dropout_prob: float = 0.2,
        use_grid: bool = True,
        grid_size: int = 64,
) -> DataLoader:
    """
    Factory function to create a DataLoader for the TraceColliderDataset.
    """
    dataset = TraceColliderDataset(
        data_dir=data_dir,
        max_trace_len=max_trace_len,
        max_colliders=max_colliders,
        augment_rotation=augment_rotation,
        augment_translation=augment_translation,
        augment_scale=augment_scale,
        augment_collider_dropout=augment_collider_dropout,
        rotation_angles=rotation_angles,
        scale_range=scale_range,
        translation_range=translation_range,
        collider_dropout_prob=collider_dropout_prob,
        use_grid=use_grid,
        grid_size=grid_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return loader


def print_dataset_statistics(data_dir: str, augment_rotation: bool = True):
    """
    Simple utility to print dataset statistics (trace length, collider count, label distribution).
    """
    dataset = TraceColliderDataset(
        data_dir,
        augment_rotation=augment_rotation,
        rotation_angles=[0, 90, 180, 270] if augment_rotation else [0]
    )

    print(f"\n{'=' * 50}")
    print(f"Dataset Statistics")
    print(f"{'=' * 50}")
    print(f"Base samples: {len(dataset.base_data_pairs)}")
    if augment_rotation:
        print(f"Total samples (with augmentation): {len(dataset)}")
        print(f"Augmentation: {dataset.rotation_angles}°")
    else:
        print(f"Total samples: {len(dataset)}")

    num_samples_to_analyze = min(len(dataset), 100)
    num_traces_list = []
    num_colliders_list = []
    label_counts = {label: 0 for label in dataset.label_to_id.keys()}
    rotation_counts = {}

    for i in range(num_samples_to_analyze):
        sample = dataset[i]
        num_traces_list.append(sample['num_traces'].item())
        num_colliders = sample['valid_mask'].sum().item()
        num_colliders_list.append(num_colliders)

        rot = sample['rotation'].item()
        rotation_counts[rot] = rotation_counts.get(rot, 0) + 1

        valid_labels = sample['labels'][sample['valid_mask']]
        for label_id in valid_labels:
            label_name = dataset.id_to_label[label_id.item()]
            label_counts[label_name] += 1

    print(f"\nTrace statistics (first {num_samples_to_analyze} samples):")
    print(f"  Min traces: {min(num_traces_list)}")
    print(f"  Max traces: {max(num_traces_list)}")
    print(f"  Avg traces: {np.mean(num_traces_list):.1f}")

    print(f"\nCollider statistics (first {num_samples_to_analyze} samples):")
    print(f"  Min colliders: {min(num_colliders_list)}")
    print(f"  Max colliders: {max(num_colliders_list)}")
    print(f"  Avg colliders: {np.mean(num_colliders_list):.1f}")

    if augment_rotation:
        print(f"\nRotation distribution (first {num_samples_to_analyze} samples):")
        for angle in sorted(rotation_counts.keys()):
            print(f"  {int(angle)}°: {rotation_counts[angle]} samples")

    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    import sys

    data_dir = "../../dataset/train"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print(f"Testing dataloader with data from: {data_dir}")

    try:
        print("\n=== WITH Rotation Augmentation ===")
        print_dataset_statistics(data_dir, augment_rotation=True)
    except Exception as e:
        print(f"Error in statistics: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\n=== WITHOUT Rotation Augmentation ===")
        print_dataset_statistics(data_dir, augment_rotation=False)
    except Exception as e:
        print(f"Error in statistics: {e}")

    try:
        print("\n=== Testing Dataloader ===")
        loader = create_dataloader(
            data_dir,
            batch_size=2,
            shuffle=False,
            augment_rotation=True
        )

        print(f"Created dataloader with {len(loader)} batches")

        for batch_idx, batch in enumerate(loader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Traces shape: {batch['traces'].shape}")
            print(f"  Trace mask shape: {batch['trace_mask'].shape}")
            print(f"  Boxes shape: {batch['boxes'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            print(f"  Valid mask shape: {batch['valid_mask'].shape}")
            print(f"  Grid maps shape: {batch['grid_maps'].shape}")
            print(f"  Num traces: {batch['num_traces'].tolist()}")
            print(f"  Num colliders: {batch['num_colliders'].tolist()}")
            print(f"  Valid colliders per sample: {batch['valid_mask'].sum(dim=1).tolist()}")

            if batch_idx >= 2:
                break

        print("\n✓ Dataloader test passed!")

    except Exception as e:
        print(f"Error testing dataloader: {e}")
        import traceback
        traceback.print_exc()