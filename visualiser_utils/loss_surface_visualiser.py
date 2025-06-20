"""
Compute and plot a loss contour in the PCA(PC-1, PC-2) plane.
Heavy forward-passes can run on GPU.
"""

from pathlib import Path
from typing import Callable, Dict, Tuple, Union
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from utils import Utils                # choose_device, assign_flat_to_params


def draw_pca_loss_surface(
    snapshot_matrix: np.ndarray,
    output_path: Union[str, Path],
    build_model_fn: Callable[[], torch.nn.Module],
    loss_fn: torch.nn.Module,
    full_feature_array: np.ndarray,
    full_label_array: np.ndarray,
    grid_points: int = 35,
    use_cuda: bool = True,
) -> None:
    """
    Parameters
    ----------
    snapshot_matrix
        (T, D) NumPy array of checkpoints.
    output_path
        PNG filename for the contour plot.
    build_model_fn
        Function that returns a *fresh* model instance.
    loss_fn
        Instantiated loss function (will be copied to device inside).
    full_feature_array, full_label_array
        Entire dataset as NumPy arrays.
    grid_points
        Resolution of α/β grid per axis.
    use_cuda
        If True and CUDA available → run forward passes on GPU.
    """
    # First check if we have pre-computed data
    run_dir = Path(output_path).parent
    precomputed_file = run_dir / "loss_surface_data.pkl"

    if precomputed_file.exists():
        print(
            f"[loss-surface] Using pre-computed data from {precomputed_file}")
        with open(precomputed_file, "rb") as handle:
            precomputed_data = pickle.load(handle)
        _plot_loss_surface(
            alpha_grid=precomputed_data['alpha_grid'],
            beta_grid=precomputed_data['beta_grid'],
            loss_matrix=precomputed_data['loss_matrix'],
            alpha_vals=precomputed_data['alpha_vals'],
            beta_vals=precomputed_data['beta_vals'],
            output_path=output_path
        )
        return

    # Check if we have enough snapshots for PCA
    if len(snapshot_matrix) < 2:
        print(
            f"[loss-surface] Not enough snapshots for PCA (need 2+, got {len(snapshot_matrix)})")

        # Create a placeholder image instead
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "Not enough snapshots for loss surface\n(need 2+ parameter states for PCA)",
                ha='center', va='center', fontsize=12)
        ax.set_xlabel("PC-1")
        ax.set_ylabel("PC-2")
        ax.set_title("Loss Surface (insufficient data)")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    # Otherwise compute from scratch
    print(f"[loss-surface] No pre-computed data found, computing from scratch...")
    device = Utils.choose_device(request_cuda=use_cuda)
    print(f"[loss-surface] using device: {device}")

    # --- PCA basis ------------------------------------------------------------
    pca = PCA(n_components=2)
    projection = pca.fit_transform(snapshot_matrix)
    alpha_vals, beta_vals = projection.T
    dir1, dir2 = pca.components_
    theta_center = pca.mean_

    # --- grid extents slightly beyond trajectory ------------------------------
    padding = 0.5
    alpha_min, alpha_max = alpha_vals.min() - padding, alpha_vals.max() + padding
    beta_min, beta_max = beta_vals.min() - padding, beta_vals.max() + padding
    alpha_grid = np.linspace(alpha_min, alpha_max, grid_points)
    beta_grid = np.linspace(beta_min, beta_max, grid_points)
    loss_matrix = np.zeros((grid_points, grid_points))

    # --- data loader on device -------------------------------------------------
    features_tensor = torch.tensor(full_feature_array,
                                   dtype=torch.float32,
                                   device=device)
    labels_tensor = torch.tensor(full_label_array,
                                 dtype=torch.long,
                                 device=device)
    data_loader = DataLoader(
        TensorDataset(features_tensor, labels_tensor),
        batch_size=4096,
        shuffle=False,
    )

    # --- build model once, reuse variable -------------------------------------
    model = build_model_fn().to(device)
    loss_fn = loss_fn.to(device)

    # --- loop over grid --------------------------------------------------------
    for i, alpha in enumerate(alpha_grid):
        for j, beta in enumerate(beta_grid):
            theta = theta_center + alpha * dir1 + beta * dir2
            Utils.assign_flat_to_params(theta, model.parameters())

            running_loss, total = 0.0, 0
            with torch.no_grad():
                for xb, yb in data_loader:
                    outputs = model(xb)
                    batch_loss = loss_fn(outputs, yb)
                    running_loss += batch_loss.item() * xb.size(0)
                    total += xb.size(0)
            loss_matrix[j, i] = running_loss / total

    # Plot the surface
    _plot_loss_surface(alpha_grid, beta_grid, loss_matrix,
                       alpha_vals, beta_vals, output_path)


def _plot_loss_surface(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    loss_matrix: np.ndarray,
    alpha_vals: np.ndarray,
    beta_vals: np.ndarray,
    output_path: Union[str, Path]
) -> None:
    """Create the contour plot using pre-computed or newly generated data."""
    # --- contour plot ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    contour = ax.contour(alpha_grid, beta_grid, loss_matrix,
                         levels=30, cmap="coolwarm")
    ax.clabel(contour, inline=True, fontsize=6)
    ax.plot(alpha_vals, beta_vals,
            color="black", marker="o", markersize=3, linewidth=1.5)
    ax.scatter(alpha_vals[-1], beta_vals[-1],
               c="red", s=40, label="final")
    ax.set(
        title="Loss surface in PCA plane",
        xlabel="α (PC-1)",
        ylabel="β (PC-2)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
