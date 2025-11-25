"""
PyTorch-based bottle cap detector.

This module provides a detector class for running inference
using PyTorch YOLOv8 models.
"""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class BottleCapDetector:
    """Bottle cap detector with color classification using PyTorch."""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 320,
        conf: float = 0.5,
    ) -> None:
        """Initialize detector.

        Args:
            model_path: Path to trained YOLO model (.pt file)
            imgsz: Input image size for inference
            conf: Confidence threshold for detections

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = ["dark_blue", "light_blue", "others"]
        self.colors = {
            0: (139, 0, 0),  # Dark blue in BGR
            1: (255, 200, 100),  # Light blue in BGR
            2: (128, 128, 128),  # Gray for others
        }

    def detect(
        self,
        image_path: str,
        visualize: bool = True,
    ) -> Dict[str, any]:
        """Run detection on image.

        Args:
            image_path: Path to input image
            visualize: Whether to visualize results on image

        Returns:
            Dictionary containing:
                - detections: List of detection dictionaries
                - inference_time_ms: Inference time in milliseconds
                - image: Annotated image (if visualize=True)

        Raises:
            ValueError: If image cannot be loaded
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # Run inference
        start_time = time.time()
        results = self.model(img, imgsz=self.imgsz, conf=self.conf, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Parse results
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                detections.append(
                    {
                        "class": self.class_names[cls],
                        "confidence": conf,
                        "bbox": xyxy.tolist(),
                    }
                )

        # Visualize if requested
        output_img = None
        if visualize:
            output_img = self._visualize(img.copy(), detections)

        return {
            "detections": detections,
            "inference_time_ms": inference_time,
            "image": output_img,
        }

    def _visualize(
        self,
        img: np.ndarray,
        detections: List[Dict[str, any]],
    ) -> np.ndarray:
        """Draw bounding boxes on image.

        Args:
            img: Input image
            detections: List of detection dictionaries

        Returns:
            Annotated image
        """
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            class_name = det["class"]
            conf = det["confidence"]

            # Get color based on class
            color_idx = self.class_names.index(class_name)
            color = self.colors[color_idx]

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # Background for text
            cv2.rectangle(
                img,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )

            # Text
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        return img

    def benchmark(
        self,
        image_path: str,
        num_runs: int = 100,
    ) -> Dict[str, float]:
        """Benchmark inference speed.

        Args:
            image_path: Path to test image
            num_runs: Number of runs for averaging

        Returns:
            Dictionary containing benchmark statistics

        Raises:
            ValueError: If image cannot be loaded
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")

        print(f"Running {num_runs} inference iterations...")

        # Warm-up
        for _ in range(10):
            _ = self.model(img, imgsz=self.imgsz, verbose=False)

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.model(img, imgsz=self.imgsz, verbose=False)
            end = time.time()
            times.append((end - start) * 1000)

        stats = {
            "avg_ms": np.mean(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "std_ms": np.std(times),
            "fps": 1000 / np.mean(times),
        }

        return stats