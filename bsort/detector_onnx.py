"""
ONNX Runtime-based fast bottle cap detector.

This module provides an optimized detector using ONNX Runtime
for faster inference on edge devices.
"""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


class FastBottleCapDetector:
    """Optimized bottle cap detector using ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 320,
        conf: float = 0.5,
    ) -> None:
        """Initialize detector with ONNX model.

        Args:
            model_path: Path to ONNX model (.onnx file)
            imgsz: Input image size
            conf: Confidence threshold

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = ["dark_blue", "light_blue", "others"]
        self.colors = {
            0: (139, 0, 0),
            1: (255, 200, 100),
            2: (128, 128, 128),
        }

        # Setup ONNX Runtime with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        providers = ["CPUExecutionProvider"]

        # Try CUDA if available
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
            print(" Using CUDA acceleration")
        else:
            print(" Using CPU")

        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape

        print(f"Model loaded: {model_path}")
        print(f"Input shape: {input_shape}")
        print(f"Output shape: {output_shape}")

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for YOLO - optimized version.

        Args:
            img: Input image (BGR format)

        Returns:
            Tuple of (preprocessed_tensor, scale, padding)
        """
        # Resize with padding to maintain aspect ratio
        img_h, img_w = img.shape[:2]
        scale = min(self.imgsz / img_w, self.imgsz / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        img_padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)

        # Calculate padding
        pad_w = (self.imgsz - new_w) // 2
        pad_h = (self.imgsz - new_h) // 2

        img_padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = img_resized

        # Convert BGR to RGB and normalize
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0

        # CHW format
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0).astype(np.float32)

        return img_batch, scale, (pad_w, pad_h)

    def postprocess(
        self,
        outputs: List[np.ndarray],
        orig_shape: Tuple[int, int],
        scale: float,
        padding: Tuple[int, int],
    ) -> List[Dict[str, any]]:
        """Process model outputs.

        Output shape: (1, 7, 2100)
        Format: [x, y, w, h, conf_class0, conf_class1, conf_class2]

        Args:
            outputs: Model output tensors
            orig_shape: Original image shape (H, W)
            scale: Scaling factor used in preprocessing
            padding: Padding values (pad_w, pad_h)

        Returns:
            List of detection dictionaries
        """
        output = outputs[0]  # (1, 7, 2100)

        # Transpose to (2100, 7)
        predictions = output[0].T  # (2100, 7)

        # Extract components
        boxes_xywh = predictions[:, :4]  # x, y, w, h
        scores = predictions[:, 4:]  # (2100, 3) - class scores

        # Get max class score and index
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence
        mask = confidences > self.conf
        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes_xywh) == 0:
            return []

        # Convert xywh to xyxy
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # Remove padding
        pad_w, pad_h = padding
        boxes_xyxy[:, [0, 2]] -= pad_w
        boxes_xyxy[:, [1, 3]] -= pad_h

        # Scale back to original size
        boxes_xyxy /= scale

        # Clip to image bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_shape[1])
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_shape[0])

        # Apply NMS
        indices = self.nms(boxes_xyxy, confidences)

        detections = []
        for idx in indices:
            detections.append(
                {
                    "class": self.class_names[class_ids[idx]],
                    "confidence": float(confidences[idx]),
                    "bbox": boxes_xyxy[idx].tolist(),
                }
            )

        return detections

    def nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float = 0.45,
    ) -> List[int]:
        """Fast Non-maximum suppression.

        Args:
            boxes: Bounding boxes (N, 4) in xyxy format
            scores: Confidence scores (N,)
            iou_threshold: IoU threshold for NMS

        Returns:
            List of indices to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(
        self,
        image_path: str,
        visualize: bool = True,
    ) -> Dict[str, any]:
        """Run detection on image.

        Args:
            image_path: Path to input image
            visualize: Whether to visualize results

        Returns:
            Dictionary containing detection results

        Raises:
            ValueError: If image cannot be loaded
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")

        orig_shape = img.shape[:2]

        # Preprocess
        input_tensor, scale, padding = self.preprocess(img)

        # Run inference
        start_time = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        inference_time = (time.perf_counter() - start_time) * 1000

        # Postprocess
        detections = self.postprocess(outputs, orig_shape, scale, padding)

        # Visualize
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
        """Draw bounding boxes.

        Args:
            img: Input image
            detections: List of detections

        Returns:
            Annotated image
        """
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            class_name = det["class"]
            conf = det["confidence"]

            color_idx = self.class_names.index(class_name)
            color = self.colors[color_idx]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            cv2.rectangle(
                img,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )

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
        """Benchmark pure inference speed.

        Args:
            image_path: Path to test image
            num_runs: Number of benchmark iterations

        Returns:
            Dictionary containing benchmark statistics

        Raises:
            ValueError: If image cannot be loaded
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")

        print(f"Running {num_runs} inference iterations...")

        # Preprocess once
        input_tensor, _, _ = self.preprocess(img)

        # Warm-up
        print("Warming up...")
        for _ in range(20):
            _ = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Benchmark
        print("Benchmarking...")
        times = []
        for i in range(num_runs):
            start = time.perf_counter()
            _ = self.session.run(self.output_names, {self.input_name: input_tensor})
            end = time.perf_counter()
            times.append((end - start) * 1000)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{num_runs}")

        times = np.array(times)

        stats = {
            "avg_ms": np.mean(times),
            "median_ms": np.median(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "std_ms": np.std(times),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
            "fps": 1000 / np.mean(times),
        }

        return stats