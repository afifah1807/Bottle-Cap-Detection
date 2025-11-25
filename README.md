# Bottle Cap Detection 🎯

[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/bottle-cap-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/bottle-cap-detection/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Deep learning-based bottle cap color detection system optimized for edge devices (Raspberry Pi 5 target: ≤10ms inference).

## 📋 Project Summary

This project implements a YOLOv8 Nano-based object detection model to classify bottle caps into three categories:
- **Dark Blue** 🔵
- **Light Blue** 🟦
- **Others** ⚪

### Key Features
- ✅ Optimized for small datasets (100-150 samples) with heavy augmentation
- ✅ Edge-device ready (320x320 input, <10ms inference target)
- ✅ ONNX export for faster inference
- ✅ CLI tool for training and inference
- ✅ Docker containerized deployment
- ✅ CI/CD with automated testing and linting
- ✅ Weights & Biases integration for experiment tracking

### Model Performance
| Metric | Value |
|--------|-------|
| mAP@50 | 99.3% |
| mAP@50-95 | 78.8% |
| Inference Time (PyTorch) | ~70ms (CPU) |
| Inference Time (ONNX) | ~8-15ms (optimized) |
| Model Size | <9 MB |

## 🚀 Quick Start

### Installation

#### Using pip
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/bottle-cap-detection.git
cd bottle-cap-detection

# Install dependencies
pip install -e .
```

#### Using Docker
```bash
# Build image
docker build -t bottle-cap-detection .

# Run container
docker run -it --rm bottle-cap-detection bsort --help
```

## 📦 Usage

### CLI Tool: `bsort`

#### Training
```bash
# Basic training
bsort train --config configs/train_config.yaml

# Custom parameters
bsort train --config configs/train_config.yaml --epochs 300 --batch-size 16
```

#### Inference (PyTorch)
```bash
# Single image
bsort infer --image path/to/image.jpg --model runs/best.pt

# With visualization
bsort infer --image sample.jpg --model best.pt --output result.jpg --visualize
```

#### Inference (ONNX - Fast)
```bash
# ONNX inference
bsort infer-onnx --image sample.jpg --model model.onnx --output result.jpg

# With benchmark
bsort infer-onnx --image sample.jpg --model model.onnx --benchmark --runs 100
```

### Python API
```python
from bsort.detector import BottleCapDetector

# Initialize detector
detector = BottleCapDetector(model_path="best.pt", imgsz=320, conf=0.5)

# Run detection
result = detector.detect("image.jpg", visualize=True)

print(f"Found {len(result['detections'])} caps")
print(f"Inference time: {result['inference_time_ms']:.2f}ms")
```

## 📁 Project Structure

```
bottle-cap-detection/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── bsort/                      # Main package
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point
│   ├── train.py                # Training script
│   ├── detector.py             # PyTorch detector
│   ├── detector_onnx.py        # ONNX detector
│   └── utils.py                # Utility functions
├── configs/
│   └── train_config.yaml       # Training configuration
├── tests/
│   ├── test_detector.py
│   └── test_train.py
├── notebooks/
│   └── Bottle_Cap_Detection.ipynb
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Configuration

Edit `configs/train_config.yaml`:

```yaml
# Model settings
model: yolov8n.pt
epochs: 250
imgsz: 320
batch_size: 16
patience: 50

# Dataset
data_yaml: dataset/data.yaml

# Augmentation
hsv_h: 0.015
hsv_s: 0.7
degrees: 15.0
mosaic: 0.5
mixup: 0.1

# Optimization
lr0: 0.0001
optimizer: Adam
```

## 🧪 Development

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=bsort --cov-report=html
```

### Code Quality
```bash
# Format code
black bsort/ tests/

# Sort imports
isort bsort/ tests/

# Lint
pylint bsort/
```

## 🐳 Docker

### Build
```bash
docker build -t bottle-cap-detection:latest .
```

### Run Training
```bash
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/runs:/app/runs \
           bottle-cap-detection:latest \
           bsort train --config /app/configs/train_config.yaml
```

### Run Inference
```bash
docker run -v $(pwd)/images:/app/images \
           bottle-cap-detection:latest \
           bsort infer --image /app/images/test.jpg --model /app/best.pt
```

## 📊 Results

### Dataset Statistics
- **Total Images**: 148
- **Train**: 132 images
- **Validation**: 12 images
- **Test**: 4 images

### Training Results
- **Best mAP@50**: 99.3%
- **Training Time**: ~2-3 hours (CPU)
- **Final Loss**: <1.0

### Inference Benchmarks
| Device | Model | Avg Time | FPS |
|--------|-------|----------|-----|
| CPU (Colab) | PyTorch | 74ms | 13.5 |
| CPU (Colab) | ONNX | 8-15ms | 62-125 |
| Raspberry Pi 5* | ONNX | <10ms* | >100* |

*Projected performance

## 🔗 Weights & Biases

View training metrics and experiments:
[https://wandb.ai/afifahapriliani2000-mecata-foundation/bottle-cap-detection/)


