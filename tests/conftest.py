"""
Pytest configuration and shared fixtures.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image."""
    import cv2

    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    img_path = temp_dir / "test_image.jpg"
    cv2.imwrite(str(img_path), img)
    return str(img_path)


@pytest.fixture
def sample_config_dict():
    """Return a sample configuration dictionary."""
    return {
        "model": {"name": "yolov8n.pt", "input_size": 320},
        "dataset": {"data_yaml": "data.yaml"},
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "patience": 1,
            "device": "cpu",
        },
        "optimization": {"optimizer": "Adam", "lr0": 0.0001, "lrf": 0.01},
        "augmentation": {
            "hsv_h": 0.015,
            "mosaic": 0.5,
        },
        "output": {"project": "runs/test", "plots": False},
        "wandb": {"enabled": False},
        "export": {"formats": []},
    }


@pytest.fixture
def sample_config_file(temp_dir, sample_config_dict):
    """Create a sample configuration file."""
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    return str(config_path)


@pytest.fixture
def mock_detection_result():
    """Return a mock detection result."""
    return {
        "detections": [
            {
                "class": "dark_blue",
                "confidence": 0.95,
                "bbox": [100, 100, 200, 200],
            },
            {
                "class": "light_blue",
                "confidence": 0.87,
                "bbox": [300, 300, 400, 400],
            },
        ],
        "inference_time_ms": 15.5,
        "image": np.zeros((640, 640, 3), dtype=np.uint8),
    }