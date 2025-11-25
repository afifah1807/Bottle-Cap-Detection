"""
CLI interface for bottle cap detection and sorting.

This module provides command-line interface for training models
and running inference on bottle cap images.
"""

import click

from bsort import __version__
from bsort.detector import BottleCapDetector
from bsort.detector_onnx import FastBottleCapDetector
from bsort.train import train_model


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Bottle Sort (bsort) - Bottle cap detection CLI tool."""
    pass


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to training configuration YAML file",
)
@click.option("--epochs", type=int, help="Override number of epochs")
@click.option("--batch-size", type=int, help="Override batch size")
@click.option("--device", type=str, help="Override device (cuda/cpu)")
def train(config: str, epochs: int, batch_size: int, device: str) -> None:
    """Train a bottle cap detection model.

    Args:
        config: Path to configuration YAML file
        epochs: Number of training epochs (overrides config)
        batch_size: Batch size (overrides config)
        device: Device to use for training (overrides config)
    """
    click.echo(" Starting training...")

    overrides = {}
    if epochs:
        overrides["epochs"] = epochs
    if batch_size:
        overrides["batch_size"] = batch_size
    if device:
        overrides["device"] = device

    try:
        model, results, best_path = train_model(config, **overrides)
        click.echo(f"\n Training complete! Best model: {best_path}")
    except Exception as e:
        click.echo(f"\n Training failed: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option(
    "--image", type=click.Path(exists=True), required=True, help="Path to input image"
)
@click.option(
    "--model", type=click.Path(exists=True), required=True, help="Path to model weights"
)
@click.option("--imgsz", type=int, default=320, help="Input image size")
@click.option("--conf", type=float, default=0.5, help="Confidence threshold")
@click.option("--output", type=str, default="output.jpg", help="Output image path")
@click.option("--visualize/--no-visualize", default=True, help="Visualize results")
@click.option("--benchmark", is_flag=True, help="Run inference benchmark")
@click.option("--runs", type=int, default=100, help="Number of benchmark runs")
def infer(
    image: str,
    model: str,
    imgsz: int,
    conf: float,
    output: str,
    visualize: bool,
    benchmark: bool,
    runs: int,
) -> None:
    """Run inference using PyTorch model.

    Args:
        image: Path to input image
        model: Path to model weights (.pt file)
        imgsz: Input image size
        conf: Confidence threshold
        output: Output image path
        visualize: Whether to visualize results
        benchmark: Whether to run benchmark
        runs: Number of benchmark iterations
    """
    click.echo("=" * 60)
    click.echo(" Bottle Cap Detector (PyTorch)")
    click.echo("=" * 60)

    try:
        detector = BottleCapDetector(model, imgsz, conf)

        click.echo(f"Image: {image}")
        click.echo(f"Model: {model}")
        click.echo("\n Running detection...")

        result = detector.detect(image, visualize=visualize)

        click.echo(f"\n Results:")
        click.echo(f"Inference time: {result['inference_time_ms']:.2f} ms")
        click.echo(f"Detections: {len(result['detections'])}")

        if result["detections"]:
            click.echo("\n Detected objects:")
            for i, det in enumerate(result["detections"], 1):
                click.echo(f"  {i}. {det['class']}: {det['confidence']:.3f}")
        else:
            click.echo("\n  No objects detected")

        if result["image"] is not None and visualize:
            import cv2

            cv2.imwrite(output, result["image"])
            click.echo(f"\n Output saved: {output}")

        if benchmark:
            click.echo(f"\n Running benchmark ({runs} iterations)...")
            stats = detector.benchmark(image, runs)

            click.echo("\n Benchmark Results:")
            click.echo(f"  Average: {stats['avg_ms']:.2f} ms")
            click.echo(f"  Min: {stats['min_ms']:.2f} ms")
            click.echo(f"  Max: {stats['max_ms']:.2f} ms")
            click.echo(f"  FPS: {stats['fps']:.2f}")

    except Exception as e:
        click.echo(f"\n Inference failed: {str(e)}", err=True)
        raise click.Abort()


@cli.command("infer-onnx")
@click.option(
    "--image", type=click.Path(exists=True), required=True, help="Path to input image"
)
@click.option(
    "--model", type=click.Path(exists=True), required=True, help="Path to ONNX model"
)
@click.option("--imgsz", type=int, default=320, help="Input image size")
@click.option("--conf", type=float, default=0.5, help="Confidence threshold")
@click.option("--output", type=str, default="output_onnx.jpg", help="Output image path")
@click.option("--visualize/--no-visualize", default=True, help="Visualize results")
@click.option("--benchmark", is_flag=True, help="Run inference benchmark")
@click.option("--runs", type=int, default=100, help="Number of benchmark runs")
def infer_onnx(
    image: str,
    model: str,
    imgsz: int,
    conf: float,
    output: str,
    visualize: bool,
    benchmark: bool,
    runs: int,
) -> None:
    """Run fast inference using ONNX model.

    Args:
        image: Path to input image
        model: Path to ONNX model (.onnx file)
        imgsz: Input image size
        conf: Confidence threshold
        output: Output image path
        visualize: Whether to visualize results
        benchmark: Whether to run benchmark
        runs: Number of benchmark iterations
    """
    click.echo("=" * 70)
    click.echo(" Fast Bottle Cap Detector (ONNX Runtime)")
    click.echo("=" * 70)

    try:
        detector = FastBottleCapDetector(model, imgsz, conf)

        click.echo(f"Image: {image}")
        click.echo(f"Model: {model}")
        click.echo("\n Running detection...")

        result = detector.detect(image, visualize=visualize)

        click.echo(f"\n Results:")
        click.echo(f" Inference time: {result['inference_time_ms']:.2f} ms")
        click.echo(f" Detections: {len(result['detections'])}")

        if result["detections"]:
            click.echo("\n Detected objects:")
            for i, det in enumerate(result["detections"], 1):
                click.echo(f"  {i}. {det['class']}: {det['confidence']:.3f}")
        else:
            click.echo("\n No objects detected")

        if result["image"] is not None and visualize:
            import cv2

            cv2.imwrite(output, result["image"])
            click.echo(f"\n Output saved: {output}")

        if benchmark:
            click.echo(f"\n{'='*70}")
            click.echo(f" BENCHMARK MODE - {runs} iterations")
            click.echo(f"{'='*70}")

            stats = detector.benchmark(image, runs)

            click.echo(f"\n BENCHMARK RESULTS:")
            click.echo(f"{'─'*70}")
            click.echo(f"  Average:    {stats['avg_ms']:>8.2f} ms")
            click.echo(f"  Median:     {stats.get('median_ms', 0):>8.2f} ms")
            click.echo(f"  Min:        {stats['min_ms']:>8.2f} ms")
            click.echo(f"  Max:        {stats['max_ms']:>8.2f} ms")
            click.echo(f"  Std Dev:    {stats['std_ms']:>8.2f} ms")
            click.echo(f"  FPS:        {stats['fps']:>8.2f}")
            click.echo(f"{'─'*70}")

            # Performance evaluation
            if stats["avg_ms"] <= 10:
                click.echo(f"\n EXCELLENT! {stats['avg_ms']:.2f}ms ≤ 10ms target")
            elif stats["avg_ms"] <= 15:
                click.echo(f"\n GOOD! {stats['avg_ms']:.2f}ms is close to target")
            else:
                click.echo(f"\n  NEEDS OPTIMIZATION: {stats['avg_ms']:.2f}ms")

    except Exception as e:
        click.echo(f"\n Inference failed: {str(e)}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli