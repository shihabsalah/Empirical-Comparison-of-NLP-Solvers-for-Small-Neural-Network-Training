"""
utils/pca_3d_visualiser.py
--------------------------------
Create a static PNG (or optional interactive HTML) that shows the optimiser’s
trajectory in the first three principal-component directions.
"""

from pathlib import Path
from typing import Sequence, Union

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required by MPL)
import numpy as np
from sklearn.decomposition import PCA


def plot_3d_pca_trajectory(
    snapshot_matrix: np.ndarray,
    output_path: Union[str, Path],
    colour_by_epoch: bool = True,
) -> None:
    """
    Parameters
    ----------
    snapshot_matrix
        NumPy array of shape (num_checkpoints, num_parameters).
    output_path
        Where to save the PNG ('.png') *or* HTML ('.html').
    colour_by_epoch
        If True, dots are coloured by epoch index.
    """
    # ------------------------------------------------------------------ PCA
    pca = PCA(n_components=3)
    xyz = pca.fit_transform(snapshot_matrix)  # shape (T, 3)
    x_vals, y_vals, z_vals = xyz.T

    # ------------------------------------------------------------------ mpl 3-D
    if str(output_path).lower().endswith(".png"):
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            x_vals, y_vals, z_vals,
            color="slategray", linewidth=1.4, label="optimiser path",
        )

        scatter = ax.scatter(
            x_vals, y_vals, z_vals,
            c=range(len(x_vals)) if colour_by_epoch else "blue",
            cmap="viridis", s=26,
        )
        ax.scatter(
            x_vals[-1], y_vals[-1], z_vals[-1],
            c="red", s=60, label="final checkpoint",
        )

        ax.set(
            title="Trajectory in first three principal components",
            xlabel="PC-1", ylabel="PC-2", zlabel="PC-3",
        )
        if colour_by_epoch:
            fig.colorbar(scatter, ax=ax, label="epoch index")
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------ Plotly
    elif str(output_path).lower().endswith(".html"):
        import plotly.express as px

        fig = px.line_3d(
            x=x_vals, y=y_vals, z=z_vals,
            color=range(len(x_vals)) if colour_by_epoch else None,
            title="Trajectory in first three principal components",
            labels={"x": "PC-1", "y": "PC-2", "z": "PC-3", "color": "epoch"},
        )
        fig.write_html(output_path)

    else:
        raise ValueError("output_path must end with .png or .html")
