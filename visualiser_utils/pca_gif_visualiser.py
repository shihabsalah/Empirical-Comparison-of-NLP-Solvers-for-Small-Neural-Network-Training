"""
utils/pca_gif_visualiser.py
--------------------------------
Create an animated GIF that shows a dot sliding along the 2-D PCA plane.
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
    frames_per_second: int = 8,
    trail_length_in_frames: int = 4,
) -> None:
    """
    Parameters
    ----------
    snapshot_matrix
        NumPy array (checkpoints, parameters).
    output_path
        GIF file name (should end with '.gif').
    frames_per_second
        Playback speed.
    trail_length_in_frames
        How many frames the marker stays on the same checkpoint.
    """
    if not str(output_path).lower().endswith(".gif"):
        raise ValueError("output_path must end with .gif")

    pca = PCA(n_components=2)
    xy = pca.fit_transform(snapshot_matrix)
    x_vals, y_vals = xy.T
    total_frames = len(x_vals) * trail_length_in_frames

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set(
        xlim=(x_vals.min() - 0.5, x_vals.max() + 0.5),
        ylim=(y_vals.min() - 0.5, y_vals.max() + 0.5),
        xlabel="PC-1", ylabel="PC-2",
        title="Optimiser path (animated)",
    )
    path_line, = ax.plot([], [], color="slategray", linewidth=1.0)
    moving_dot, = ax.plot([], [], marker="o", color="red", markersize=6)

    def initialise():
        path_line.set_data([], [])
        moving_dot.set_data([], [])
        return path_line, moving_dot

    def update(frame_index: int):
        checkpoint_index = frame_index // trail_length_in_frames
        path_line.set_data(
            x_vals[: checkpoint_index + 1],
            y_vals[: checkpoint_index + 1],
        )
        # Wrap scalars in a list so `set_data` sees a sequence
        moving_dot.set_data(
            [x_vals[checkpoint_index]],
            [y_vals[checkpoint_index]],
        )
        return path_line, moving_dot

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=initialise,
        frames=total_frames,
        interval=1000 / frames_per_second,
        blit=True,
    )
    anim.save(output_path, writer="pillow", fps=frames_per_second)
    plt.close(fig)
