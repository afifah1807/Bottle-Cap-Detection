"""
Training module for bottle cap detection model.

This module handles model training with YOLOv8, including configuration loading,
training execution, and model export.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import wandb
import yaml
from ultralytics import YOLO


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def train_model(
    config_path: str, **overrides: Any
) -> Tuple[YOLO, Any, str]:
    """Train YOLOv8 model for bottle cap detection.

    Args:
        config_path: Path to configuration YAML file
        **overrides: Optional parameters to override config values

    Returns:
        Tuple of (trained_model, results, best_model_path)

    Raises:
        Exception: If training fails
    """
    # Load configuration
    config = load_config(config_path)

    # Extract configuration sections
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    optimization_cfg = config.get("optimization", {})
    augmentation_cfg = config.get("augmentation", {})
    output_cfg = config.get("output", {})
    wandb_cfg = config.get("wandb", {})

    # Apply overrides
    if "epochs" in overrides:
        training_cfg["epochs"] = overrides["epochs"]
    if "batch_size" in overrides:
        training_cfg["batch_size"] = overrides["batch_size"]
    if "device" in overrides:
        training_cfg["device"] = overrides["device"]

    # Auto-detect device if not specified
    if "device" not in training_cfg or training_cfg["device"] == "cuda":
        training_cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = output_cfg.get("name") or f"yolov8n_cap_{timestamp}"

    # Print training info
    print("=" * 50)
    print(" Bottle Cap Detection Training (YOLOv8n)")
    print(f"Dataset Size: Small. Augmentation is CRUCIAL.")
    print("=" * 50)
    print(f"Device: {training_cfg['device']}")
    print(f"Base Model: {model_cfg['name']} (Pre-trained)")
    print(f"Fine-Tuning LR: {optimization_cfg['lr0']}")
    print(f"Patience (Early Stop): {training_cfg['patience']}")
    print("=" * 50)

    # Initialize Weights & Biases
    if wandb_cfg.get("enabled", True):
        wandb.init(
            project=wandb_cfg.get("project", "bottle-cap-detection"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name") or run_name,
            config={
                "model": model_cfg,
                "training": training_cfg,
                "optimization": optimization_cfg,
                "augmentation": augmentation_cfg,
            },
            tags=wandb_cfg.get("tags", []),
        )

    # Load pre-trained model
    model = YOLO(model_cfg["name"])

    # Prepare training arguments
    train_args = {
        # Dataset
        "data": dataset_cfg["data_yaml"],
        # Training parameters
        "epochs": training_cfg["epochs"],
        "imgsz": model_cfg["input_size"],
        "batch": training_cfg["batch_size"],
        "device": training_cfg["device"],
        "patience": training_cfg["patience"],
        "save_period": training_cfg.get("save_period", 50),
        "workers": training_cfg.get("workers", 8),
        # Output
        "project": output_cfg.get("project", "runs/train"),
        "name": run_name,
        "plots": output_cfg.get("plots", True),
        "verbose": output_cfg.get("verbose", False),
        # Optimization
        "optimizer": optimization_cfg.get("optimizer", "Adam"),
        "lr0": optimization_cfg["lr0"],
        "lrf": optimization_cfg.get("lrf", 0.01),
        "momentum": optimization_cfg.get("momentum", 0.937),
        "weight_decay": optimization_cfg.get("weight_decay", 0.0005),
        "warmup_epochs": optimization_cfg.get("warmup_epochs", 3.0),
        "warmup_momentum": optimization_cfg.get("warmup_momentum", 0.8),
        "warmup_bias_lr": optimization_cfg.get("warmup_bias_lr", 0.1),
    }

    # Add augmentation parameters
    train_args.update(augmentation_cfg)

    # Train model
    print("\n Starting training...\n")
    results = model.train(**train_args)

    # Validation after training
    print("\n" + "=" * 50)
    print(" Validating final model...")
    print("=" * 50)

    final_val_results = model.val()

    # Log final metrics to W&B
    if wandb_cfg.get("enabled", True) and wandb.run:
        wandb.log(
            {
                "final_mAP50": final_val_results.box.map50,
                "final_mAP50-95": final_val_results.box.map,
            }
        )

    # Get best model path
    best_model_path = os.path.join(
        output_cfg.get("project", "runs/train"),
        run_name,
        "weights",
        "best.pt",
    )

    print(f"\n Best model saved: {best_model_path}")

    # Export models if specified
    export_formats = config.get("export", {}).get("formats", [])
    if export_formats:
        print("\n" + "=" * 50)
        print(" Exporting optimized models...")
        print("=" * 50)

        best_model = YOLO(best_model_path)

        if "onnx" in export_formats:
            print("\n Exporting to ONNX...")
            onnx_cfg = config.get("export", {}).get("onnx", {})
            onnx_path = best_model.export(
                format="onnx",
                imgsz=model_cfg["input_size"],
                simplify=onnx_cfg.get("simplify", True),
                opset=onnx_cfg.get("opset", 11),
            )
            print(f" ONNX model saved: {onnx_path}")

        if "torchscript" in export_formats:
            print("\n Exporting to TorchScript...")
            ts_path = best_model.export(
                format="torchscript",
                imgsz=model_cfg["input_size"],
            )
            print(f" TorchScript model saved: {ts_path}")

        if "engine" in export_formats:
            try:
                print("\n Exporting to TensorRT...")
                trt_cfg = config.get("export", {}).get("tensorrt", {})
                trt_path = best_model.export(
                    format="engine",
                    imgsz=model_cfg["input_size"],
                    half=trt_cfg.get("half", True),
                )
                print(f" TensorRT model saved: {trt_path}")
            except Exception as e:
                print(f" TensorRT export failed: {e}")
                print("   (This is normal if TensorRT is not installed)")

    # Finish W&B run
    if wandb_cfg.get("enabled", True) and wandb.run:
        wandb.finish()

    print("\n" + "=" * 50)
    print(" Training Complete!")
    print("=" * 50)
    print(f"Best model: {best_model_path}")
    print("\nNext steps:")
    print("1. Check W&B dashboard for detailed metrics")
    print("2. Test inference on sample images")
    print("3. Deploy to edge device for real-world testing")

    return model, results, best_model_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)

    config_file = sys.argv[1]
    train_model(config_file)