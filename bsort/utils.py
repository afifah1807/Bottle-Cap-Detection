"""
Utility functions for bottle cap detection.

This module provides helper functions for data processing,
visualization, and performance metrics.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def check_file_exists(file_path: str) -> bool:
    """Check if file exists.

    Args:
        file_path: Path to file

    Returns:
        True if file exists, False otherwise
    """
    return Path(file_path).exists()


def create_output_dir(output_path: str) -> None:
    """Create output directory if it doesn't exist.

    Args:
        output_path: Path to output directory
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)


def load_image(image_path: str) -> np.ndarray:
    """Load image from file.

    Args:
        image_path: Path to image file

    Returns:
        Loaded image as numpy array

    Raises:
        ValueError: If image cannot be loaded
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    return img


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save image to file.

    Args:
        image: Image as numpy array
        output_path: Path to save image
    """
    create_output_dir(str(Path(output_path).parent))
    cv2.imwrite(output_path, image)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) of two boxes.

    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]

    Returns:
        IoU score
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def format_time(milliseconds: float) -> str:
    """Format time in milliseconds to human-readable string.

    Args:
        milliseconds: Time in milliseconds

    Returns:
        Formatted time string
    """
    if milliseconds < 1:
        return f"{milliseconds * 1000:.2f} µs"
    elif milliseconds < 1000:
        return f"{milliseconds:.2f} ms"
    else:
        return f"{milliseconds / 1000:.2f} s"


def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """Get BGR color for class name.

    Args:
        class_name: Name of class

    Returns:
        BGR color tuple
    """
    colors = {
        "dark_blue": (139, 0, 0),
        "light_blue": (255, 200, 100),
        "others": (128, 128, 128),
    }
    return colors.get(class_name, (255, 255, 255))


def calculate_fps(inference_time_ms: float) -> float:
    """Calculate FPS from inference time.

    Args:
        inference_time_ms: Inference time in milliseconds

    Returns:
        Frames per second
    """
    if inference_time_ms == 0:
        return 0.0
    return 1000.0 / inference_time_ms


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: int,
    pad_value: int = 114,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image maintaining aspect ratio with padding.

    Args:
        image: Input image
        target_size: Target size (square)
        pad_value: Padding value

    Returns:
        Tuple of (resized_image, scale, (pad_w, pad_h))
    """
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image
    padded = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)

    # Calculate padding
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    return padded, scale, (pad_w, pad_h)


def count_detections_by_class(detections: List[Dict]) -> Dict[str, int]:
    """Count number of detections per class.

    Args:
        detections: List of detection dictionaries

    Returns:
        Dictionary mapping class names to counts
    """
    counts = {}
    for det in detections:
        class_name = det["class"]
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts