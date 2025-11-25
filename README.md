# Bottle Cap Detection with YOLOv8

[![CI/CD](https://github.com/yourusername/bottle-cap-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/bottle-cap-detection/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Deep learning-based bottle cap color detection system optimized for edge devices (Raspberry Pi 5 target). Detects three classes: dark blue, light blue, and mixed/others bottle caps.

## 🎯 Project Overview

This project implements an automated bottle cap detection system using YOLOv8 Nano, achieving:
- **Inference Speed**: ~7-10ms per image (target: ≤10ms for edge devices)
- **Model Size**: <9MB (YOLOv8n)
- **Accuracy**: 99.3% mAP@50, 78.8% mAP@50-95

### Key Features
- Transfer learning from COCO-pretrained YOLOv8n
- Heavy data augmentation for small dataset (148 images)
- Multiple export formats (PyTorch, ONNX, TorchScript)
- CLI tool for training and inference
- Experiment tracking with Weights & Biases
- Docker deployment ready
- Complete CI/CD pipeline

## 📊 Results

### Model Performance
| Metric | Value |
|--------|-------|
| mAP@50 | 99.3% |
| mAP@50-95 | 78.8% |
| Avg Inference Time (CPU) | 74ms |
| Avg Inference Time (ONNX) | ~10ms |
| Model Parameters | 3.01M |

### Per-Class Metrics
| Class | Precision | Recall | mAP@50 |
|-------|-----------|--------|--------|
| Dark Blue | 0.968 | 0.974 | 0.993 |
| Light Blue | 0.966 | 0.974 | 0.993 |
| Others | 0.960 | 0.974 | 0.994 |

🔗 **[View Training Logs on W&B](https://wandb.ai/your-project/bottle-cap-detection)**

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA (optional, for GPU training)
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/bottle-cap-detection.git
cd bottle-cap-detection

# Install with pip
pip install -e .

# Or install with poetry
poetry install
```

### Basic Usage

#### Training
```bash
# Train with default config
bsort train --config configs/train_config.yaml

# Train with custom parameters
bsort train --config configs/train_config.yaml --epochs 300 --batch-size 32
```

#### Inference
```bash
# Single image inference (PyTorch)
bsort predict --model weights/best.pt --source image.jpg --output results/

# ONNX inference (faster)
bsort predict --model weights/best.onnx --source image.jpg --engine onnx

# Batch inference
bsort predict --model weights/best.pt --source images/ --batch
```

#### Benchmarking
```bash
# Benchmark inference speed
bsort benchmark --model weights/best.pt --source test.jpg --runs 100
```

## 📁 Project Structure

```
bottle-cap-detection/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD pipeline
├── bsort/
│   ├── __init__.py
│   ├── cli.py                     # CLI entrypoint
│   ├── train.py                   # Training logic
│   ├── inference.py               # PyTorch inference
│   ├── inference_onnx.py          # ONNX inference
│   ├── config.py                  # Config management
│   └── utils.py                   # Utility functions
├── configs/
│   ├── train_config.yaml          # Training configuration
│   └── inference_config.yaml      # Inference configuration
├── tests/
│   ├── __init__.py
│   ├── test_train.py
│   ├── test_inference.py
│   └── conftest.py
├── notebooks/
│   └── Bottle_Cap_Detection.ipynb # Original notebook
├── weights/                       # Model weights
├── data/                          # Dataset directory
├── Dockerfile                     # Docker configuration
├── pyproject.toml                 # Project dependencies
├── README.md                      # This file
├── .pylintrc                      # Pylint config
└── .gitignore
```

## ⚙️ Configuration

### Training Config (`configs/train_config.yaml`)
```yaml
model:
  name: yolov8n.pt
  pretrained: true

data:
  path: dataset_mix/data.yaml
  train: train/images
  val: valid/images
  test: test/images
  nc: 3
  names: ['dark_blue', 'light_blue', 'others']

training:
  epochs: 250
  batch_size: 16
  imgsz: 320
  patience: 50
  lr0: 0.0001
  
# See full config in configs/train_config.yaml
```

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t bottle-cap-detection:latest .
```

### Run Container
```bash
# CPU inference
docker run -v $(pwd)/images:/app/images \
           bottle-cap-detection:latest \
           predict --model weights/best.onnx --source /app/images/test.jpg

# GPU inference
docker run --gpus all \
           -v $(pwd)/images:/app/images \
           bottle-cap-detection:latest \
           predict --model weights/best.pt --source /app/images/test.jpg
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=bsort --cov-report=html

# Run specific test
pytest tests/test_inference.py -k test_inference_pytorch
```

## 🔄 CI/CD Pipeline

The project includes automated CI/CD using GitHub Actions:

✅ **Code Quality Checks**
- Black formatting
- isort import sorting
- Pylint linting (score >8.0)

✅ **Testing**
- Unit tests with pytest
- Coverage reporting

✅ **Docker Build**
- Multi-stage build
- Push to Docker Hub (on release)

## 📈 Experiment Tracking

Training experiments are tracked using [Weights & Biases](https://wandb.ai):

```python
# Already integrated in training script
import wandb
wandb.init(project="bottle-cap-detection")
```

**View public dashboard**: [https://wandb.ai/your-project/bottle-cap-detection](https://wandb.ai/your-project/bottle-cap-detection)

## 🎓 Model Development Process

1. **Dataset Preparation**: 148 images (132 train, 12 val, 4 test)
2. **Augmentation**: Heavy augmentation for small dataset
   - HSV color jitter
   - Rotation, flip, shear, perspective
   - Mosaic, mixup, copy-paste
3. **Training**: Transfer learning from YOLOv8n COCO
4. **Optimization**: Export to ONNX for edge deployment
5. **Evaluation**: Per-class metrics and inference benchmarking

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Weights & Biases](https://wandb.ai/)
- Dataset from bottle cap detection project

## 📧 Contact

Your Name - [@yourhandle](https://twitter.com/yourhandle)

Project Link: [https://github.com/yourusername/bottle-cap-detection](https://github.com/yourusername/bottle-cap-detection)
