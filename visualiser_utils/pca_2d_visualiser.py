"""
Draw a static 2-D PCA trajectory PNG.
"""

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_2d_pca_trajectory(
    snapshot_matrix: np.ndarray,
    output_path: Union[str, Path],
) -> None:
    """
    Parameters
    ----------
    snapshot_matrix
        shape (num_checkpoints, num_parameters)
    output_path
        e.g. run_dir / 'pca_path.png'
    """
    pca = PCA(n_components=2)
    projection = pca.fit_transform(snapshot_matrix)
    x_vals, y_vals = projection.T

    fig, ax = plt.subplots(figsize=(5, 4))
    scatter = ax.scatter(
        x_vals,
        y_vals,
        c=np.arange(len(x_vals)),
        cmap="viridis",
        s=18,
        label="checkpoints",
    )
    ax.plot(x_vals, y_vals, color="gray", linewidth=0.8)
    ax.scatter(x_vals[-1], y_vals[-1], c="red", s=30, label="final")

    ax.set(
        title="Training trajectory in PCA plane",
        xlabel="PC-1",
        ylabel="PC-2",
    )
    fig.colorbar(scatter, ax=ax, label="epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
