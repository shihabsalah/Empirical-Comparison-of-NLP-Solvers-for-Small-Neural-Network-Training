"""
analyse.py
-----------
Thin wrapper: loads snapshot matrix, then delegates to visualiser utilities
based on CLI flags.
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
    parser = argparse.ArgumentParser(description="Visualise training run.")
    parser.add_argument("run_dir", type=pathlib.Path,
                        help="folder containing snapshots.pkl")
    parser.add_argument("--pca2d", action="store_true",
                        help="plot 2-D PCA PNG")
    parser.add_argument("--pca3d", action="store_true",
                        help="plot 3-D PCA PNG")
    parser.add_argument("--gif", action="store_true",
                        help="create animated GIF")
    parser.add_argument("--surface", action="store_true",
                        help="draw loss contour in PCA plane")
    parser.add_argument("--grid", type=int, default=35,
                        help="grid resolution for loss surface")
    parser.add_argument("--gpu", action="store_true",
                        help="use GPU for loss-surface evaluation")
    args = parser.parse_args()

    snapshot_matrix = load_snapshot_matrix(args.run_dir)

    if args.pca2d:
        plot_2d_pca_trajectory(
            snapshot_matrix=snapshot_matrix,
            output_path=args.run_dir / "pca_path.png",
        )

    if args.pca3d:
        plot_3d_pca_trajectory(
            snapshot_matrix=snapshot_matrix,
            output_path=args.run_dir / "pca_path_3d.png",
        )

    if args.gif:
        create_pca_gif(
            snapshot_matrix=snapshot_matrix,
            output_path=args.run_dir / "pca_path.gif",
            frames_per_second=8,
            trail_length_in_frames=4,
        )

    if args.surface:
        # Load dataset once for loss surface
        features, labels = load_dataset(
            DatasetName.MNIST_DIGITS_CLASSIFICATION)
        draw_pca_loss_surface(
            snapshot_matrix=snapshot_matrix,
            output_path=args.run_dir / "pca_loss_surface.png",
            build_model_fn=lambda: build_model(ModelName.MNIST_MLP),
            loss_fn=get_loss(LossName.CROSS_ENTROPY),
            full_feature_array=features,
            full_label_array=labels,
            grid_points=args.grid,
            use_cuda=args.gpu,
        )


if __name__ == "__main__":
    main()
