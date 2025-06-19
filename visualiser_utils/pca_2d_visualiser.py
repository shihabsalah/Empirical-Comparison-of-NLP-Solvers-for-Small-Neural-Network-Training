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
    title: str | None = None,
) -> None:
    """
    Create a 2-D PCA trajectory plot from model checkpoints.

    Parameters
    ----------
    snapshot_matrix
        (T, D) array of parameter snapshots (rows = epochs).
    output_path
        Image filename (should be .png or .jpg).
    title
        Custom title for plot (None → auto-generate).
    """
    # Check if we have enough snapshots for PCA
    if len(snapshot_matrix) < 2:
        print(
            f"Warning: Not enough snapshots for PCA (need 2+, got {len(snapshot_matrix)})")

        # Create a simple figure with a message
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "Not enough snapshots for PCA\n(only 1 parameter state available)",
                ha='center', va='center', fontsize=12)
        ax.set_xlabel("PC-1")
        ax.set_ylabel("PC-2")
        ax.set_title(title or "PCA Trajectory (insufficient data)")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    # Proceed with normal PCA for 2+ snapshots
    pca = PCA(n_components=2)
    projection = pca.fit_transform(snapshot_matrix)

    # Coordinates for each checkpoint
    x_coords, y_coords = projection.T

    # Ratio of variance explained by these components
    var_explained = pca.explained_variance_ratio_

    # Generate the plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x_coords, y_coords, 'o-', markersize=4)
    ax.grid(True, alpha=0.3)

    # Starting point and ending point
    ax.plot(x_coords[0], y_coords[0], 'go', markersize=7, label="start")
    ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=7, label="end")

    # Add a few epoch numbers as labels
    t_max = len(x_coords)
    for t in range(0, t_max, max(1, t_max // 5)):
        ax.annotate(f"{t}",
                    xy=(x_coords[t], y_coords[t]),
                    xytext=(5, 5),  # small offset
                    textcoords='offset points')

    # Default title explains variance captured by the PCs
    if title is None:
        title = f"PCA Trajectory (PC-1/2 explain {100*sum(var_explained):0.1f}% variance)"
    ax.set_title(title)

    # X and Y axis labels with variance explained
    ax.set_xlabel(f"PC-1 ({100*var_explained[0]:0.1f}%)")
    ax.set_ylabel(f"PC-2 ({100*var_explained[1]:0.1f}%)")

    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
