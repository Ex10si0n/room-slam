import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple


class TraceColliderDataset(Dataset):
    """
    Dataset for trace sequences and collider predictions.

    Expects dual-file format:
    - Trace files: *_trace.json (contains list of trace points)
    - Collider files: *_collider.json (contains collider ground truth)

    Supports aggressive data augmentation:
    - Rotation (0°, 90°, 180°, 270°)
    - Translation (random shifts)
    - Scaling (simulate different room sizes)
    - Collider dropout (remove random colliders)
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
            collider_dropout_prob: float = 0.2
    ):
        """
        Args:
            data_dir: Directory containing trace and collider JSON files
            max_trace_len: Maximum number of trace points
            max_colliders: Maximum number of colliders per scene
            augment_rotation: If True, apply rotation augmentation
            augment_translation: If True, apply random translation
            augment_scale: If True, apply random scaling
            augment_collider_dropout: If True, randomly drop colliders
            rotation_angles: List of rotation angles in degrees
            scale_range: (min_scale, max_scale) for random scaling
            translation_range: Maximum translation in meters
            collider_dropout_prob: Probability of dropping each collider
        """
        self.data_dir = Path(data_dir)
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

        # Label mapping
        self.label_to_id = {
            'BLOCK': 0,
            'LOW': 1,
            'MID': 2,
            'HIGH': 3
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        # Load file pairs
        self.base_data_pairs = self._load_data_pairs()
        print(f"Found {len(self.base_data_pairs)} base samples in {data_dir}")

        # Expand dataset with rotations
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
            self.data_pairs = [{'trace': p['trace'], 'collider': p['collider'], 'rotation': 0}
                               for p in self.base_data_pairs]

        if len(self.data_pairs) == 0:
            raise ValueError(f"No valid data files found in {data_dir}")

    def _load_data_pairs(self) -> List[Dict[str, Path]]:
        """
        Load and pair trace and collider files.

        Supports multiple naming patterns:
        1. *_trace.json + *_collider.json (paired files)
        2. agent_data_*.json + colliders.json (agent traces with shared colliders)
        3. human_data_*.json + colliders.json (human traces with shared colliders)

        Returns:
            List of dicts with 'trace' and 'collider' file paths
        """
        pairs = []

        # Pattern 1: Standard *_trace.json + *_collider.json pairs
        trace_files = sorted(self.data_dir.glob("*_trace.json"))

        for trace_file in trace_files:
            # Find corresponding collider file
            base_name = trace_file.stem.replace('_trace', '')
            collider_file = self.data_dir / f"{base_name}_collider.json"

            if collider_file.exists():
                pairs.append({
                    'trace': trace_file,
                    'collider': collider_file
                })
            else:
                print(f"Warning: No collider file for {trace_file}")

        # Pattern 2 & 3: agent_data_* / human_data_* with shared colliders.json
        if len(pairs) == 0:
            # Look for shared colliders.json
            shared_collider = self.data_dir / "colliders.json"

            if shared_collider.exists():
                # Find all agent and human data files
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
                print("Warning: No colliders.json found")

        return pairs

    def _rotate_traces(self, traces: List[Dict], angle_degrees: float) -> List[Dict]:
        """
        Rotate traces around Y-axis (vertical axis).

        Args:
            traces: List of trace points with x, y, z, timestamp
            angle_degrees: Rotation angle in degrees (90, 180, 270)

        Returns:
            Rotated traces
        """
        angle_rad = np.radians(angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Rotation matrix around Y-axis
        # [x']   [cos  0  sin] [x]
        # [y'] = [ 0   1   0 ] [y]
        # [z']   [-sin 0  cos] [z]

        rotated_traces = []
        for point in traces:
            x, y, z = point['x'], point['y'], point['z']

            # Apply rotation
            x_new = cos_a * x + sin_a * z
            z_new = -sin_a * x + cos_a * z

            rotated_traces.append({
                'x': x_new,
                'y': y,  # Y unchanged (vertical axis)
                'z': z_new,
                'timestamp': point['timestamp']
            })

        return rotated_traces

    def _rotate_colliders(self, colliders: List[Dict], angle_degrees: float) -> List[Dict]:
        """
        Rotate colliders around Y-axis.

        Args:
            colliders: List of collider dicts with center and size
            angle_degrees: Rotation angle in degrees

        Returns:
            Rotated colliders
        """
        angle_rad = np.radians(angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        rotated_colliders = []
        for col in colliders:
            center = col['center']
            size = col['size']

            # Rotate center position
            cx, cy, cz = center['x'], center['y'], center['z']
            cx_new = cos_a * cx + sin_a * cz
            cz_new = -sin_a * cx + cos_a * cz

            # Rotate size (swap x and z for 90° and 270°)
            sx, sy, sz = size['x'], size['y'], size['z']

            if angle_degrees == 90 or angle_degrees == 270:
                # Swap X and Z dimensions
                sx_new, sz_new = sz, sx
            else:
                # 0° or 180°: keep dimensions
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
        """Translate traces in X-Z plane"""
        return [{
            'x': p['x'] + tx,
            'y': p['y'],
            'z': p['z'] + tz,
            'timestamp': p['timestamp']
        } for p in traces]

    def _translate_colliders(self, colliders: List[Dict], tx: float, tz: float) -> List[Dict]:
        """Translate colliders in X-Z plane"""
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
        """Scale traces (simulate different room sizes)"""
        return [{
            'x': p['x'] * scale,
            'y': p['y'] * scale,
            'z': p['z'] * scale,
            'timestamp': p['timestamp']
        } for p in traces]

    def _scale_colliders(self, colliders: List[Dict], scale: float) -> List[Dict]:
        """Scale colliders"""
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
        Randomly drop colliders to force model to learn from traces.
        Never drop walls (BLOCK with large size).
        """
        kept_colliders = []
        for col in colliders:
            # Keep walls (large BLOCK colliders)
            is_wall = (col.get('label') == 'BLOCK' and
                       (col['size'].get('x', 0) > 5.0 or
                        col['size'].get('z', 0) > 5.0))

            # Keep with probability or if it's a wall
            if is_wall or np.random.rand() > dropout_prob:
                kept_colliders.append(col)

        return kept_colliders if kept_colliders else colliders  # Keep at least something

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with aggressive augmentation.

        Returns:
            Dict containing augmented traces and colliders
        """
        pair = self.data_pairs[idx]
        rotation_angle = pair['rotation']

        # Load trace data
        with open(pair['trace'], 'r') as f:
            trace_data = json.load(f)

        # Load collider data
        with open(pair['collider'], 'r') as f:
            collider_data = json.load(f)

        # Extract traces and colliders
        traces = trace_data if isinstance(trace_data, list) else []
        colliders = collider_data.get('colliders', [])

        # Apply rotation augmentation
        if rotation_angle != 0:
            traces = self._rotate_traces(traces, rotation_angle)
            colliders = self._rotate_colliders(colliders, rotation_angle)

        # Apply random translation (simulates different room positions)
        if self.augment_translation:
            tx = np.random.uniform(-self.translation_range, self.translation_range)
            tz = np.random.uniform(-self.translation_range, self.translation_range)
            traces = self._translate_traces(traces, tx, tz)
            colliders = self._translate_colliders(colliders, tx, tz)

        # Apply random scaling (simulates different room sizes)
        if self.augment_scale:
            scale = np.random.uniform(*self.scale_range)
            traces = self._scale_traces(traces, scale)
            colliders = self._scale_colliders(colliders, scale)

        # Apply collider dropout (simulates missing/partial observations)
        if self.augment_collider_dropout and np.random.rand() < 0.5:
            colliders = self._dropout_colliders(colliders, self.collider_dropout_prob)

        # Process data
        trace_array = self._process_traces(traces)
        collider_boxes, collider_labels, collider_valid = self._process_colliders(colliders)

        return {
            'traces': trace_array,
            'trace_mask': torch.ones(len(trace_array), dtype=torch.bool),
            'boxes': collider_boxes,
            'labels': collider_labels,
            'valid_mask': collider_valid,
            'num_traces': torch.tensor(len(traces), dtype=torch.long),
            'num_colliders': torch.tensor(len(colliders), dtype=torch.long),
            'filename': f"{pair['trace'].name}_rot{rotation_angle}",
            'rotation': torch.tensor(rotation_angle, dtype=torch.float32)
        }

    def _process_traces(self, traces: List[Dict]) -> torch.Tensor:
        """
        Convert trace list to tensor [N, 4] (x, y, z, t).

        Args:
            traces: List of dicts with keys 'x', 'y', 'z', 'timestamp'

        Returns:
            Tensor of shape [N, 4]
        """
        if len(traces) == 0:
            return torch.zeros((1, 4), dtype=torch.float32)

        # Extract coordinates
        trace_list = []
        for p in traces:
            trace_list.append([
                p.get('x', 0.0),
                p.get('y', 0.0),
                p.get('z', 0.0),
                p.get('timestamp', 0.0)
            ])

        trace_array = np.array(trace_list, dtype=np.float32)

        # Normalize timestamp to start from 0
        if trace_array.shape[0] > 0 and trace_array[:, 3].max() > 0:
            trace_array[:, 3] -= trace_array[:, 3].min()

        # Downsample if too long
        if len(trace_array) > self.max_trace_len:
            indices = np.linspace(0, len(trace_array) - 1, self.max_trace_len, dtype=int)
            trace_array = trace_array[indices]

        return torch.from_numpy(trace_array)

    def _process_colliders(
            self,
            colliders: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert colliders to tensors.

        Args:
            colliders: List of collider dicts with 'center', 'size', 'label'

        Returns:
            boxes: [M, 6] tensor of (cx, cy, cz, sx, sy, sz)
            labels: [M] tensor of label IDs
            valid_mask: [M] bool tensor (True for valid colliders, False for padding)
        """
        # Initialize with padding
        boxes = torch.zeros((self.max_colliders, 6), dtype=torch.float32)
        labels = torch.full((self.max_colliders,), -1, dtype=torch.long)
        valid_mask = torch.zeros(self.max_colliders, dtype=torch.bool)

        if len(colliders) == 0:
            return boxes, labels, valid_mask

        # Fill in actual colliders
        num_valid = min(len(colliders), self.max_colliders)

        for i, col in enumerate(colliders[:num_valid]):
            # Extract center
            center = col.get('center', {})
            cx = center.get('x', 0.0)
            cy = center.get('y', 0.0)
            cz = center.get('z', 0.0)

            # Extract size
            size = col.get('size', {})
            sx = size.get('x', 0.0)
            sy = size.get('y', 0.0)
            sz = size.get('z', 0.0)

            # Extract label
            label_str = col.get('label', 'BLOCK')
            label_id = self.label_to_id.get(label_str, 0)

            # Fill tensors
            boxes[i] = torch.tensor([cx, cy, cz, sx, sy, sz], dtype=torch.float32)
            labels[i] = label_id
            valid_mask[i] = True

        return boxes, labels, valid_mask


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable length traces.
    Pads traces to the maximum length in the batch.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched tensors with proper padding
    """
    # Find max trace length in batch
    max_len = max(item['traces'].shape[0] for item in batch)

    # Pad traces
    traces_padded = []
    masks = []

    for item in batch:
        trace = item['traces']  # [N, 4]
        pad_len = max_len - len(trace)

        if pad_len > 0:
            # Pad with zeros
            trace = torch.cat([
                trace,
                torch.zeros((pad_len, 4), dtype=torch.float32)
            ], dim=0)

            # Create mask: True for valid positions, False for padding
            mask = torch.cat([
                torch.ones(len(item['traces']), dtype=torch.bool),
                torch.zeros(pad_len, dtype=torch.bool)
            ])
        else:
            mask = torch.ones(len(trace), dtype=torch.bool)

        traces_padded.append(trace)
        masks.append(mask)

    # Stack everything
    batched = {
        'traces': torch.stack(traces_padded),  # [B, N, 4]
        'trace_mask': torch.stack(masks),  # [B, N]
        'boxes': torch.stack([item['boxes'] for item in batch]),  # [B, M, 6]
        'labels': torch.stack([item['labels'] for item in batch]),  # [B, M]
        'valid_mask': torch.stack([item['valid_mask'] for item in batch]),  # [B, M]
        'num_traces': torch.stack([item['num_traces'] for item in batch]),
        'num_colliders': torch.stack([item['num_colliders'] for item in batch]),
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
        collider_dropout_prob: float = 0.2
) -> DataLoader:
    """
    Create dataloader with aggressive augmentation.

    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        max_trace_len: Maximum trace length
        max_colliders: Maximum number of colliders
        augment_rotation: Enable rotation augmentation
        augment_translation: Enable random translation
        augment_scale: Enable random scaling
        augment_collider_dropout: Enable random collider dropout
        rotation_angles: List of rotation angles (e.g., [0, 90, 180, 270])
        scale_range: (min_scale, max_scale) for random scaling
        translation_range: Maximum translation in meters
        collider_dropout_prob: Probability of dropping each collider

    Returns:
        DataLoader instance
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
        collider_dropout_prob=collider_dropout_prob
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
    """Print statistics about the dataset."""
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

    # Analyze first few samples (or all if dataset is small)
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

        # Count rotations
        rot = sample['rotation'].item()
        rotation_counts[rot] = rotation_counts.get(rot, 0) + 1

        # Count labels
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

    # Test dataloader
    data_dir = "../../dataset/train"

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print(f"Testing dataloader with data from: {data_dir}")

    # Print statistics WITH augmentation
    try:
        print("\n=== WITH Rotation Augmentation ===")
        print_dataset_statistics(data_dir, augment_rotation=True)
    except Exception as e:
        print(f"Error in statistics: {e}")
        import traceback

        traceback.print_exc()

    # Print statistics WITHOUT augmentation
    try:
        print("\n=== WITHOUT Rotation Augmentation ===")
        print_dataset_statistics(data_dir, augment_rotation=False)
    except Exception as e:
        print(f"Error in statistics: {e}")

    # Test dataloader
    try:
        print("\n=== Testing Dataloader ===")
        loader = create_dataloader(
            data_dir,
            batch_size=2,
            shuffle=False,
            augment_rotation=True
        )

        print(f"Created dataloader with {len(loader)} batches")

        # Test first few batches
        for batch_idx, batch in enumerate(loader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Traces shape: {batch['traces'].shape}")
            print(f"  Trace mask shape: {batch['trace_mask'].shape}")
            print(f"  Boxes shape: {batch['boxes'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            print(f"  Valid mask shape: {batch['valid_mask'].shape}")
            print(f"  Num traces: {batch['num_traces'].tolist()}")
            print(f"  Num colliders: {batch['num_colliders'].tolist()}")
            print(f"  Valid colliders per sample: {batch['valid_mask'].sum(dim=1).tolist()}")

            # Show sample data
            print(f"\n  First sample in batch:")
            print(f"    First 3 trace points:")
            print(f"      {batch['traces'][0, :3]}")

            # Show first valid collider
            valid_indices = batch['valid_mask'][0].nonzero(as_tuple=False)
            if len(valid_indices) > 0:
                valid_idx = valid_indices[0].item()
                print(f"    First valid collider:")
                print(f"      Box: {batch['boxes'][0, valid_idx]}")
                print(f"      Label ID: {batch['labels'][0, valid_idx].item()}")
                label_name = loader.dataset.id_to_label[batch['labels'][0, valid_idx].item()]
                print(f"      Label name: {label_name}")

            if batch_idx >= 2:  # Only show first 3 batches
                break

        print("\n✓ Dataloader test passed!")

    except Exception as e:
        print(f"Error testing dataloader: {e}")
        import traceback

        traceback.print_exc()