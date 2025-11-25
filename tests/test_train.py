"""
Unit tests for training module.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from bsort.train import load_config, train_model


class TestTrainingModule:
    """Test suite for training module."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration dictionary."""
        return {
            "model": {"name": "yolov8n.pt", "input_size": 320},
            "dataset": {"data_yaml": "data.yaml"},
            "training": {
                "epochs": 10,
                "batch_size": 8,
                "patience": 5,
                "device": "cpu",
            },
            "optimization": {
                "optimizer": "Adam",
                "lr0": 0.0001,
                "lrf": 0.01,
            },
            "augmentation": {
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "degrees": 15.0,
                "mosaic": 0.5,
            },
            "output": {"project": "runs/test", "plots": True},
            "wandb": {"enabled": False},
            "export": {"formats": []},
        }

    @pytest.fixture
    def config_file(self, sample_config, tmp_path):
        """Create a temporary config file."""
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)
        return str(config_path)

    def test_load_config_valid(self, config_file):
        """Test loading valid configuration file."""
        config = load_config(config_file)

        assert isinstance(config, dict)
        assert "model" in config
        assert "training" in config
        assert config["model"]["name"] == "yolov8n.pt"

    def test_load_config_not_found(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_load_config_structure(self, config_file):
        """Test configuration has expected structure."""
        config = load_config(config_file)

        required_sections = [
            "model",
            "dataset",
            "training",
            "optimization",
            "augmentation",
        ]

        for section in required_sections:
            assert section in config

    @patch("bsort.train.YOLO")
    @patch("bsort.train.wandb")
    def test_train_model_with_overrides(
        self, mock_wandb, mock_yolo_class, config_file
    ):
        """Test training with parameter overrides."""
        # Mock YOLO model
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Mock training results
        mock_results = MagicMock()
        mock_model.train.return_value = mock_results

        # Mock validation results
        mock_val_results = MagicMock()
        mock_val_results.box.map50 = 0.95
        mock_val_results.box.map = 0.75
        mock_model.val.return_value = mock_val_results

        # Mock wandb
        mock_wandb.run = None

        # Train with overrides
        with patch("bsort.train.os.path.join") as mock_join:
            mock_join.return_value = "test_model.pt"

            model, results, best_path = train_model(
                config_file, epochs=5, batch_size=4
            )

            # Verify YOLO was initialized
            mock_yolo_class.assert_called()

            # Verify training was called
            mock_model.train.assert_called_once()

            # Verify overrides were applied
            train_call_kwargs = mock_model.train.call_args[1]
            assert train_call_kwargs["epochs"] == 5
            assert train_call_kwargs["batch"] == 4

    def test_config_augmentation_parameters(self, sample_config):
        """Test augmentation parameters are correctly defined."""
        aug_config = sample_config["augmentation"]

        required_params = ["hsv_h", "hsv_s", "degrees", "mosaic"]

        for param in required_params:
            assert param in aug_config
            assert isinstance(aug_config[param], (int, float))

    def test_config_optimization_parameters(self, sample_config):
        """Test optimization parameters are correctly defined."""
        opt_config = sample_config["optimization"]

        assert "optimizer" in opt_config
        assert "lr0" in opt_config
        assert "lrf" in opt_config
        assert opt_config["optimizer"] in ["Adam", "SGD", "AdamW"]