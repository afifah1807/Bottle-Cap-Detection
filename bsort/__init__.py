"""
Bottle Sort (bsort) - Bottle cap detection and sorting package.

This package provides tools for training and running inference
on bottle cap detection models optimized for edge devices.
"""

__version__ = "1.0.0"
__author__ = "Afifah Apriliani"
__email__ = "afifahapriliani2000@gmail.com"

from bsort.detector import BottleCapDetector
from bsort.detector_onnx import FastBottleCapDetector
from bsort.train import train_model

__all__ = [
    "BottleCapDetector",
    "FastBottleCapDetector",
    "train_model",
    "__version__",
]