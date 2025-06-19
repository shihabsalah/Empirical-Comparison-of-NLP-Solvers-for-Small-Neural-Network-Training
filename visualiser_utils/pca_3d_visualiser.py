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
    use_plotly: bool = False,
) -> None:
    """
    Create a 3-D PCA trajectory plot from model checkpoints.

    Parameters
    ----------
    snapshot_matrix
        (T, D) array of parameter snapshots (rows = epochs).
    output_path
        Image filename (should be .png or .jpg).
    use_plotly
        If True, use Plotly for interactive 3D visualization (HTML output).
    """
    # Check if we have enough snapshots for PCA
    if len(snapshot_matrix) < 3:  # Need at least 3 for 3D PCA
        print(
            f"Warning: Not enough snapshots for 3D PCA (need 3+, got {len(snapshot_matrix)})")

        # Create a simple figure with a message
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Place text in center of plot
        ax.text(0, 0, 0, "Not enough snapshots for 3D PCA\n(need 3+ parameter states)",
                ha='center', va='center', fontsize=12)

        ax.set_xlabel("PC-1")
        ax.set_ylabel("PC-2")
        ax.set_zlabel("PC-3")
        ax.set_title("3D PCA Trajectory (insufficient data)")

        # Set limits to make empty plot look reasonable
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    # From here on, normal behavior for 3+ snapshots
    if use_plotly:
        # If the Plotly backend is requested – use it
        import plotly.express as px

        # Compute the PCA projection of the trajectory to 3-D
        pca = PCA(n_components=3)
        projection = pca.fit_transform(snapshot_matrix)

        # Ratios of variances explained by the first three PCs
        var_explained = pca.explained_variance_ratio_

        # Generate synthetic "epoch" numbers (could be optimization steps)
        epochs = list(range(len(snapshot_matrix)))

        # Create Plotly figure – interactive version for browser
        fig = px.scatter_3d(
            x=projection[:, 0],
            y=projection[:, 1],
            z=projection[:, 2],
            color=epochs,
            color_continuous_scale='viridis',
            title=f"PCA Trajectory (explains {100*sum(var_explained):.1f}% variance)",
        )

        fig.update_layout(
            scene=dict(
                xaxis_title=f"PC-1 ({100*var_explained[0]:.1f}%)",
                yaxis_title=f"PC-2 ({100*var_explained[1]:.1f}%)",
                zaxis_title=f"PC-3 ({100*var_explained[2]:.1f}%)",
            )
        )

        # Add a connected line to show the path
        fig.update_traces(marker={'size': 5})

        # Add the trajectory line
        fig.add_scatter3d(
            x=projection[:, 0],
            y=projection[:, 1],
            z=projection[:, 2],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.5)', width=3),
            showlegend=False,
        )

        # Save to HTML file (interactive)
        html_path = str(output_path).replace('.png', '.html')
        fig.write_html(html_path)

        # Also save a static image for the report
        fig.write_image(str(output_path))

    else:
        # Standard matplotlib static figure
        pca = PCA(n_components=3)
        projection = pca.fit_transform(snapshot_matrix)
        var_explained = pca.explained_variance_ratio_

        # Make the plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the trajectory in 3-D
        ax.plot(
            projection[:, 0],
            projection[:, 1],
            projection[:, 2],
            'o-',
            markersize=4,
            linewidth=1,
        )

        # Label the start and end points
        ax.scatter(
            projection[0, 0],
            projection[0, 1],
            projection[0, 2],
            color='green',
            s=50,
            label='start',
        )
        ax.scatter(
            projection[-1, 0],
            projection[-1, 1],
            projection[-1, 2],
            color='red',
            s=50,
            label='end',
        )

        # Annotate some epoch numbers
        n_annotations = 8   # Max number of annotations to avoid clutter
        step = max(1, len(snapshot_matrix) // n_annotations)
        for t in range(0, len(snapshot_matrix), step):
            ax.text(
                projection[t, 0] + 0.02,
                projection[t, 1] + 0.02,
                projection[t, 2] + 0.02,
                f"{t}",  # epoch number as text label
                fontsize=8,
            )

        # Style the plot with titles and legend
        ax.set_title(
            f"PCA Trajectory (explains {100*sum(var_explained):.1f}% variance)")
        ax.set_xlabel(f"PC-1 ({100*var_explained[0]:.1f}%)")
        ax.set_ylabel(f"PC-2 ({100*var_explained[1]:.1f}%)")
        ax.set_zlabel(f"PC-3 ({100*var_explained[2]:.1f}%)")
        ax.legend()

        # Save the figure
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
