"""
utils/pca_gif_visualiser.py
--------------------------------
Create an animated GIF of the optimizer's trajectory in the PCA space.
"""

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from sklearn.decomposition import PCA


def create_pca_gif(
    snapshot_matrix: np.ndarray,
    output_path: Union[str, Path],
    frames_per_second: int = 4,
    trail_length_in_frames: int = 5,
) -> None:
    """
    Create an animated GIF showing the optimizer's path through the PCA space.

    Parameters
    ----------
    snapshot_matrix
        (T, D) array of parameter snapshots (rows = epochs).
    output_path
        Path to save the output .gif file.
    frames_per_second
        Animation speed (frames per second).
    trail_length_in_frames
        How many previous points to show in the "trail" behind the current point.
    """
    # Check if we have enough snapshots for PCA
    if len(snapshot_matrix) < 2:
        print(
            f"Warning: Not enough snapshots for PCA GIF (need 2+, got {len(snapshot_matrix)})")

        # Create a simple static image instead of a GIF
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "Not enough snapshots for PCA GIF\n(only 1 parameter state available)",
                ha='center', va='center', fontsize=12)
        ax.set_xlabel("PC-1")
        ax.set_ylabel("PC-2")
        ax.set_title("PCA Trajectory Animation (insufficient data)")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()

        # Save as PNG instead of GIF when we don't have enough data
        png_output = str(output_path).replace('.gif', '_static.png')
        fig.savefig(png_output, dpi=150)
        print(
            f"  Saved static placeholder image to {png_output} (GIF not possible with only 1 snapshot)")
        plt.close(fig)
        return

    # Normal behavior for 2+ snapshots - existing code below
    pca = PCA(n_components=2)
    projection = pca.fit_transform(snapshot_matrix)

    # Extract the x/y coordinates
    x_coords, y_coords = projection.T
    n_frames = len(x_coords)

    # Add some padding to the plot limits
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    x_padding = 0.15 * (x_max - x_min)
    y_padding = 0.15 * (y_max - y_min)
    xlim = (x_min - x_padding, x_max + x_padding)
    ylim = (y_min - y_padding, y_max + y_padding)

    # Create the figure and axis only once
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Variance explained by the first two components
    var_explained = pca.explained_variance_ratio_

    ax.set_title(
        f"PCA Trajectory (explains {100*sum(var_explained):.1f}% variance)")
    ax.set_xlabel(f"PC-1 ({100*var_explained[0]:.1f}%)")
    ax.set_ylabel(f"PC-2 ({100*var_explained[1]:.1f}%)")

    # The trail is a line that will get updated in each frame
    trail, = ax.plot([], [], 'o-', markersize=4, color='blue', alpha=0.7)

    # Current position marker
    point, = ax.plot([], [], 'ro', markersize=8, label='current')

    # Show full path in the background
    ax.plot(x_coords, y_coords, '.-', color='gray', alpha=0.3, markersize=3)

    # Start and end markers
    ax.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='start')
    ax.plot(x_coords[-1], y_coords[-1], 'mo', markersize=8, label='end')
    ax.legend()

    # Progress indicator text (epoch number)
    progress_text = ax.text(
        0.02, 0.02, '',
        transform=ax.transAxes,
        backgroundcolor='white',
        alpha=0.7,
    )

    # Animation update function
    def update(t):
        if t >= n_frames:  # t will go from 0 to n_frames-1
            return (trail, point, progress_text)

        # Calculate how many previous points to show
        trail_start = max(0, t - trail_length_in_frames)

        # Update the trail
        trail.set_data(x_coords[trail_start:t+1], y_coords[trail_start:t+1])

        # Update the current point
        point.set_data([x_coords[t]], [y_coords[t]])

        # Update progress text
        progress_text.set_text(f"Step {t}")

        return (trail, point, progress_text)

    # Create animation
    anim = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=n_frames,
        interval=1000 / frames_per_second,  # milliseconds between frames
        blit=True,
    )

    # Save as GIF - requires pillow
    anim.save(output_path, writer='pillow', fps=frames_per_second)
    plt.close(fig)
