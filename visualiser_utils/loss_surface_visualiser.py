"""
Compute and plot a loss contour in the PCA(PC-1, PC-2) plane.
Heavy forward-passes can run on GPU.
"""

from pathlib import Path
from typing import Callable, Tuple, Union

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
