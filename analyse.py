"""
analyse.py
-----------
Benchmark Visualization Script for Optimizer Runs

This script scans the `runs/` directory for subfolders named after optimizers (e.g., ADAM, L_BFGS, TRUST_REGION_NEWTON, INTERIOR_POINT, GRADIENT_DESCENT). For each optimizer, it loads the saved parameter snapshots and generates visualizations based on the command-line options:

  • 2-D PCA trajectory plots (--pca2d)
  • 3-D PCA trajectory plots (--pca3d)
  • Animated PCA trajectory GIFs (--gif)
  • Loss surface contour in PCA plane (--surface)

Usage:
    python analyse.py [OPTIONS]

Options:
    --pca2d           Generate static 2-D PCA trajectory for each optimizer.
    --pca3d           Generate static 3-D PCA trajectory for each optimizer.
    --gif             Create animated GIF of the PCA trajectory for each optimizer.
    --surface         Draw loss contour in the PCA plane for each optimizer.
    --grid N          Grid resolution (default: 35) for loss surface computation.
    --gpu             Use GPU (CUDA) for loss surface forward passes and contour evaluation.

Recommendations:
    • To speed up loss surface computation, run with `--gpu` on a machine with a CUDA-capable GPU.
    • Increase `--grid` for finer contours (more compute), decrease for faster runs.
    • Ensure each optimizer has a corresponding folder in `runs/` containing `snapshots.pkl`.

How It Works:
    1. Locate all subdirectories under `runs/` (optimizer names).
    2. For each directory, load `snapshots.pkl` as a NumPy array of shape (checkpoints, parameters).
    3. Call visualization utilities from `visualiser_utils/` based on the selected options.
    4. Save output files (PNG or GIF) back into the optimizer’s run directory with descriptive filenames.
"""

import argparse
import pathlib
import pickle
import numpy as np

from visualiser_utils.pca_2d_visualiser import plot_2d_pca_trajectory
from visualiser_utils.pca_3d_visualiser import plot_3d_pca_trajectory
from visualiser_utils.pca_gif_visualiser import create_pca_gif
from visualiser_utils.loss_surface_visualiser import draw_pca_loss_surface

from data import load_dataset
from models import build_model
from losses import get_loss
from config import DatasetName, ModelName, LossName


def load_snapshot_matrix(run_directory: pathlib.Path) -> np.ndarray:
    """Reads the pickled snapshot list and stacks into (T, D) NumPy array."""
    with open(run_directory / "snapshots.pkl", "rb") as file:
        return np.vstack(pickle.load(file))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise training runs for all optimizers.")
    parser.add_argument("--pca2d", action="store_true",
                        help="plot 2-D PCA PNG for each optimizer")
    parser.add_argument("--pca3d", action="store_true",
                        help="plot 3-D PCA PNG for each optimizer")
    parser.add_argument("--gif", action="store_true",
                        help="create animated GIF for each optimizer")
    parser.add_argument("--surface", action="store_true",
                        help="draw loss contour in PCA plane for each optimizer")
    parser.add_argument("--grid", type=int, default=35,
                        help="grid resolution for loss surface")
    parser.add_argument("--gpu", action="store_true",
                        help="use GPU for loss-surface evaluation")
    args = parser.parse_args()

    base_runs_dir = pathlib.Path("runs")
    if not base_runs_dir.is_dir():
        print(f"Error: Base runs directory '{base_runs_dir}' not found.")
        return

    optimizer_dirs = [d for d in base_runs_dir.iterdir() if d.is_dir()]

    if not optimizer_dirs:
        print(f"No optimizer run directories found in '{base_runs_dir}'.")
        return

    for optimizer_run_dir in optimizer_dirs:
        print(f"\n--- Analysing optimizer: {optimizer_run_dir.name} ---")
        snapshot_file = optimizer_run_dir / "snapshots.pkl"
        if not snapshot_file.exists():
            print(
                f"Snapshots file not found in {optimizer_run_dir}, skipping.")
            continue

        snapshot_matrix = load_snapshot_matrix(optimizer_run_dir)

        if args.pca2d:
            print(f"  Generating 2D PCA plot...")
            plot_2d_pca_trajectory(
                snapshot_matrix=snapshot_matrix,
                output_path=optimizer_run_dir /
                f"pca_path_2d_{optimizer_run_dir.name}.png",
            )
            print(
                f"  Saved 2D PCA plot to {optimizer_run_dir / f'pca_path_2d_{optimizer_run_dir.name}.png'}")

        if args.pca3d:
            print(f"  Generating 3D PCA plot...")
            plot_3d_pca_trajectory(
                snapshot_matrix=snapshot_matrix,
                output_path=optimizer_run_dir /
                f"pca_path_3d_{optimizer_run_dir.name}.png",
            )
            print(
                f"  Saved 3D PCA plot to {optimizer_run_dir / f'pca_path_3d_{optimizer_run_dir.name}.png'}")

        if args.gif:
            print(f"  Generating PCA GIF...")
            create_pca_gif(
                snapshot_matrix=snapshot_matrix,
                output_path=optimizer_run_dir /
                f"pca_path_{optimizer_run_dir.name}.gif",
                frames_per_second=8,
                trail_length_in_frames=4,
            )
            print(
                f"  Saved PCA GIF to {optimizer_run_dir / f'pca_path_{optimizer_run_dir.name}.gif'}")

        if args.surface:
            print(f"  Generating loss surface plot...")
            # Load dataset once for loss surface - assuming MNIST for all now
            # TODO: This might need to be more flexible if different optimizers used different datasets/models
            features, labels = load_dataset(
                DatasetName.MNIST_DIGITS)
            draw_pca_loss_surface(
                snapshot_matrix=snapshot_matrix,
                output_path=optimizer_run_dir /
                f"pca_loss_surface_{optimizer_run_dir.name}.png",
                build_model_fn=lambda: build_model(ModelName.MNIST_MLP),
                loss_fn=get_loss(LossName.CROSS_ENTROPY),
                full_feature_array=features,
                full_label_array=labels,
                grid_points=args.grid,
                use_cuda=args.gpu,
            )
            print(
                f"  Saved loss surface plot to {optimizer_run_dir / f'pca_loss_surface_{optimizer_run_dir.name}.png'}")


if __name__ == "__main__":
    main()
