# Room-SLAM: Semantic Layout Acquisition from Movement

A deep learning approach to infer indoor room layouts from human movement traces using GRU-based sequence-to-object prediction.

## Overview

This project explores a novel approach to generating semantic maps of indoor environments by analyzing human movement patterns. The core hypothesis is that human movement is constrained and guided by physical objects in a space - we walk around tables, stop at chairs, and follow paths through doorways. By analyzing these behavioral patterns, we can reconstruct the semantic layout of the environment.

## Key Features

- **2D Movement Analysis**: Processes time-series of (x, y) coordinates from AprilTag tracking
- **GRU-based Architecture**: Uses Gated Recurrent Units to encode temporal movement patterns
- **Structured Output**: Predicts object classes, positions, sizes, and orientations
- **Multi-task Learning**: Combines classification and regression losses
- **Baseline Comparison**: Includes rule-based occupancy heatmap baseline
- **Comprehensive Evaluation**: Implements mAP, IoU, and other object detection metrics

## Object Classes

- **GROUND**: Floor, accessible space
- **LOW**: Chairs, beds, low-height storage (sittable surfaces)
- **MID**: Desks, tables, mid-height surfaces (not typically sittable)
- **BLOCK**: Walls, never accessible

## Project Structure

```
room-slam/
├── src/
│   ├── data/
│   │   ├── constants.py      # Configuration and class definitions
│   │   └── dataset.py        # Data loading and preprocessing
│   ├── models/
│   │   ├── baseline.py       # Rule-based occupancy heatmap baseline
│   │   └── room_slam.py      # Main GRU-based model
│   └── utils/                # Utility functions
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── demo.py                   # Demo with synthetic data
└── requirements.txt          # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ex10si0n/Room-SLAM.git
cd Room-SLAM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run the Demo
Test the baseline model with synthetic data:
```bash
python demo.py
```

### 2. Create Sample Data
Generate sample data for testing:
```bash
python train.py --create_sample_data
```

### 3. Train the Model
Train the GRU-based model:
```bash
python train.py --data_dir data/sample --epochs 50
```

### 4. Evaluate the Model
Evaluate trained model performance:
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --compare_baseline --visualize
```

## Data Format

### Input: Movement Traces
CSV files with columns: `timestamp, x, y`
```csv
0.0, 1.2, 2.3
0.1, 1.25, 2.31
0.2, 1.3, 2.35
...
```

### Output: Object Predictions
JSON format with structured object descriptions:
```json
{
  "objects": [
    {
      "class_id": 1,
      "class_name": "LOW",
      "position": [2.5, 1.8],
      "size": [0.6, 0.6],
      "orientation": 0.0,
      "confidence": 0.85
    }
  ]
}
```

## Model Architecture

### GRU Encoder
- Processes movement sequences (batch_size, seq_len, 2)
- Bidirectional GRU with dropout
- Outputs fixed-size latent representation

### MLP Decoder
- Multi-layer perceptron with object-specific heads
- Predicts: classes, positions, sizes, orientations, validity
- Structured output for maximum of N objects

### Multi-task Loss
- Classification loss (CrossEntropy)
- Regression losses (L1 for position, size, orientation)
- Validity loss (BCE for object existence)

## Evaluation Metrics

- **Mean Average Precision (mAP)**: Primary metric for object detection
- **IoU**: Intersection over Union for bounding box accuracy
- **Precision/Recall**: Per-class performance metrics
- **Baseline Comparison**: Rule-based occupancy heatmap

## Data Collection

The project uses a hybrid data collection approach:

1. **AprilTag Tracking**: iOS app with CSV export for precise 2D coordinates
2. **3D Room Mapping**: iPhone LiDAR for environmental context
3. **Manual Labeling**: Ground-truth semantic object annotations

### AprilTag Setup
- Modified iOS app: https://github.com/Ex10si0n/google-apriltags-ios
- Overhead camera positioning for 2D tracking
- 10Hz sampling rate recommended

## Training Configuration

```python
# Default hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
SEQUENCE_LENGTH = 500
MAX_OBJECTS = 10
NUM_EPOCHS = 100
```

## Results and Visualization

The model generates:
- Movement trace plots
- Occupancy heatmaps
- Stationary time analysis
- Predicted object layouts
- Training curves and metrics

## Future Extensions

- **3D Support**: Extend to (x, y, z) coordinates
- **Real-time Inference**: Mobile device deployment
- **Unsupervised Learning**: GAN-based layout generation
- **Multi-room Support**: Handle connected spaces

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{room-slam2024,
  title={Room-SLAM: Semantic Layout Acquisition from Movement},
  author={Zhongbo Yan and Yiming Yin},
  year={2024},
  howpublished={GitHub repository},
  url={https://github.com/Ex10si0n/Room-SLAM}
}
```

## Acknowledgments

- AprilTag visual fiducial system
- PyTorch deep learning framework
- OpenCV for computer vision utilities
