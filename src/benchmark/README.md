# Trace to Collider Transformer

Deep learning model for predicting 3D colliders from agent trajectory sequences.

## Overview

This project trains a Transformer-based model to reconstruct room geometry (3D bounding boxes) from agent movement traces. The model learns spatial constraints by analyzing how agents navigate around obstacles.

## Project Structure

```
room-slam/
├── dataset/                  # Dataset directory
│   ├── agent_data_*_trace.json
│   ├── human_data_*_trace.json
│   └── colliders.json        # Ground truth colliders
└── src/
    └── benchmark/            # Training scripts
        ├── dataloader.py     # Data loading
        ├── model.py          # Transformer model
        ├── train.py          # Training script
        ├── inference.py      # Inference with NMS
        ├── visualize.py      # Visualization (top view only)
        └── README.md
```

## Data Format

### Input: Trace Files
```json
[
  {"timestamp": 15.97091, "x": 1.209547, "y": -0.4596849, "z": 2.122062},
  {"timestamp": 15.98762, "x": 1.208123, "y": -0.4598234, "z": 2.120456},
  ...
]
```

### Output: Collider Predictions
```json
{
  "colliders": [
    {
      "type": "BoxCollider",
      "label": "BLOCK",
      "confidence": 0.96,
      "center": {"x": 1.889, "y": 1.341, "z": -1.256},
      "size": {"x": 0.133, "y": 3.600, "z": 9.075}
    }
  ]
}
```

**Label Types:**
- `BLOCK`: Walls and large obstacles
- `LOW`: Low furniture (chairs, stools)
- `MID`: Medium-height furniture (tables, desks)
- `HIGH`: High obstacles

## Installation

```bash
# Clone repository
git clone <repository-url>
cd room-slam/src/benchmark

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- scipy, numpy, matplotlib, tqdm

## Usage

### 1. Test Data Loader

```bash
# Test dataloader and view dataset statistics
python dataloader.py ../../dataset
```

**Output:**
```
Found 7 samples in ../../dataset
Dataset Statistics:
  Total samples: 7
  Avg traces per sample: 8,500.5
  Avg colliders per sample: 11.2
  Label distribution:
    BLOCK: 450
    LOW: 120
    MID: 80
```

### 2. Train Model

```bash
python train.py
```

**Training Configuration:**
- Batch size: 4
- Epochs: 200
- Learning rate: 2e-4 with warmup
- Model: Lightweight Transformer (3M parameters)
- Loss: Classification + L1 + GIoU
- Optimizer: AdamW with cosine LR decay
- Device: Automatic CUDA detection

**Training Progress:**
```
Epoch 0: loss=5.234, cls=1.234, l1=2.000, giou=2.000, LR=0.000020
Epoch 10: loss=3.456, cls=0.789, l1=1.567, giou=1.100, LR=0.000200
...
Saved best model with loss 1.234
```

Models are saved in `./checkpoints/`:
- `best_model.pth`: Best validation loss
- `checkpoint_epoch_X.pth`: Periodic checkpoints

### 3. Run Inference

```bash
python inference.py \
  --checkpoint ./checkpoints/best_model.pth \
  --input ../../dataset/human_data_20251015_181004.json \
  --output predictions.json \
  --threshold 0.7 \
  --nms 0.3
```

**Arguments:**
- `--checkpoint`: Path to trained model
- `--input`: Input trace file (JSON)
- `--output`: Output prediction file
- `--threshold`: Confidence threshold (default: 0.7)
- `--nms`: NMS IoU threshold for duplicate removal (default: 0.3)

**Output:**
```
Using device: cuda (NVIDIA Tesla T4)
Loading model from ./checkpoints/best_model.pth
Processing ../../dataset/human_data_20251015_181004.json
Downsampling traces from 14829 to 3000 points

Found 8 colliders:
  1. BLOCK at (1.89, 1.34, -1.26) - confidence: 0.963
  2. BLOCK at (-1.82, 1.51, -1.26) - confidence: 0.956
  3. MID at (-0.00, -0.07, -1.72) - confidence: 0.872
  ...

Results saved to predictions.json
```

### 4. Visualize Results

```bash
# Visualize traces with predictions
python visualize.py \
  --input ../../dataset/human_data_20251015_181004.json \
  --predictions predictions.json \
  --output trace_with_pred.png

# Include ground truth colliders
python visualize.py \
  --input ../../dataset/human_data_20251015_181004.json \
  --colliders ../../dataset/colliders.json \
  --predictions predictions.json \
  --output full_comparison.png
```

**Arguments:**
- `--input`: Trace file (JSON)
- `--colliders`: (Optional) Ground truth colliders file
- `--predictions`: (Optional) Prediction results file
- `--output`: Output image path (PNG)

**Visualization Legend:**
- 🔵 Blue line: Agent trajectory
- 🟩 Green dot: Start position
- 🟥 Red dot: End position
- 🔴 Red filled boxes: Ground truth colliders
- 🔵 Blue dashed boxes: Predictions (with confidence scores)

## Model Architecture

```
Input Traces [B, N, 4] (x, y, z, timestamp)
    ↓
[Positional Encoding] - 3D + temporal encoding
    ↓
[Transformer Encoder] - 3 layers, 4 heads, d=128
    ↓
Encoded Features [B, N, 128]
    ↓
[Transformer Decoder] - 3 layers, 4 heads
    ↓ (cross-attention with learnable queries)
Object Queries [B, 30, 128]
    ↓
[Prediction Heads]
    ├─→ Box Head: [B, 30, 6] (cx, cy, cz, sx, sy, sz)
    └─→ Class Head: [B, 30, 4] (BLOCK/LOW/MID/HIGH)
```

**Key Features:**
- DETR-style architecture with learnable object queries
- Hungarian matching for training
- Dynamic positional encoding (supports up to 20K points)
- Automatic downsampling for long traces (>3000 points)

## Loss Function

```python
Total Loss = 2.0 × Classification Loss 
           + 5.0 × L1 Loss 
           + 2.0 × GIoU Loss
```

- **Classification Loss**: Cross-entropy for label prediction
- **L1 Loss**: Absolute error for box centers and sizes
- **GIoU Loss**: Generalized IoU for better localization

## Post-Processing

**Non-Maximum Suppression (NMS):**
1. Filter predictions by confidence threshold (default: 0.7)
2. Sort by confidence scores
3. For each class separately:
   - Keep highest scoring box
   - Remove boxes with IoU > 0.3 overlap
4. Return deduplicated predictions

This eliminates duplicate predictions and improves precision.

## Training Tips

### Hyperparameter Tuning

| Parameter | Default | Adjust If... |
|-----------|---------|--------------|
| `batch_size` | 4 | Increase if GPU has memory / Decrease if OOM |
| `lr` | 2e-4 | Decrease if loss oscillates |
| `num_epochs` | 200 | Increase for better convergence |
| `threshold` | 0.7 | Lower for more detections / Higher for precision |
| `nms` | 0.3 | Lower for aggressive deduplication |

### Common Issues

**Q: Training loss not decreasing?**
- Check data loading: `python dataloader.py`
- Verify ground truth quality
- Lower learning rate to 1e-4
- Increase batch size

**Q: Too many overlapping predictions?**
- Lower `--nms` threshold (0.3 → 0.2)
- Increase `--threshold` (0.7 → 0.8)
- Train for more epochs

**Q: Missing small objects (LOW/MID)?**
- Check label distribution in dataset
- Increase classification loss weight
- Add data augmentation

**Q: Out of memory during training?**
- Reduce `batch_size` (4 → 2)
- Reduce `max_trace_len` in dataloader (3000 → 2000)
- Reduce `d_model` (128 → 64)

## Performance Metrics

**Model Size:**
- Parameters: ~3M trainable
- Model file: ~12 MB (FP32)
- GPU memory: ~2-4 GB during training

**Inference Speed (NVIDIA T4):**
- Single trace: ~100ms
- Batch of 4: ~300ms

**Memory Usage:**
- Training: ~4 GB GPU memory
- Inference: ~1 GB GPU memory

## Advanced Features

### Custom Data Augmentation

Add to `dataloader.py`:
```python
# Random rotation around Y-axis
angle = np.random.uniform(-np.pi/6, np.pi/6)
cos_a, sin_a = np.cos(angle), np.sin(angle)
trace_array[:, [0, 2]] = trace_array[:, [0, 2]] @ np.array([
    [cos_a, sin_a],
    [-sin_a, cos_a]
])

# Random translation
trace_array[:, :3] += np.random.uniform(-0.1, 0.1, 3)
```

### Multi-Scale Training

Train with different trace lengths:
```python
max_trace_lens = [1000, 2000, 3000, 5000]
for epoch in range(num_epochs):
    max_len = np.random.choice(max_trace_lens)
    # Update dataloader with new max_len
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{trace2collider2025,
  title={Trace to Collider: Learning 3D Scene Geometry from Agent Trajectories},
  author={Your Name},
  year={2025}
}
```

## License

MIT License

## Troubleshooting

### CUDA Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"

# Install correct PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Data Format Issues

```bash
# Validate trace file format
python -c "
import json
with open('dataset/agent_data_xxx_trace.json') as f:
    data = json.load(f)
    print(f'Traces: {len(data)}')
    print(f'Keys: {data[0].keys()}')
"

# Validate collider file format
python -c "
import json
with open('dataset/colliders.json') as f:
    data = json.load(f)
    print(f'Colliders: {len(data[\"colliders\"])}')
    print(f'Keys: {data[\"colliders\"][0].keys()}')
"
```

## Contact

For questions or issues, please open a GitHub issue or contact [your-email].