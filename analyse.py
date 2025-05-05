import pickle
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader
from flatten import assign_flat_to_params
from models import build_model
from losses import get_loss
from config import ModelName, LossName
from data import load_dataset
from config import DatasetName

# ------------------------------------------------------------


def main(run_dir: pathlib.Path,
         draw_loss_surface: bool = False,
         grid_pts: int = 35):

    # 1) load snapshots list
    with open(run_dir / "snapshots.pkl", "rb") as f:
        snaps = np.vstack(pickle.load(f))          # shape (T, D)

    # 2) PCA -> 2-D projection
    pca = PCA(n_components=2)
    proj = pca.fit_transform(snaps)                # shape (T, 2)

    # 3) trajectory plot
    plt.figure(figsize=(5, 4))
    plt.scatter(proj[:, 0], proj[:, 1],
                c=np.arange(len(proj)), cmap="viridis",
                s=18, label="checkpoints")
    plt.plot(proj[:, 0], proj[:, 1], color="gray", linewidth=0.8)
    plt.scatter(proj[-1, 0], proj[-1, 1], c="red", s=30, label="final")
    plt.colorbar(label="epoch")
    plt.title("Training trajectory in PCA plane")
    plt.xlabel("PC-1")
    plt.ylabel("PC-2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "pca_path.png", dpi=150)

    # 4) optional loss-surface grid in the same plane -----------------
    if draw_loss_surface:
        print("Computing loss grid … (this can take a few minutes)")
        # 4a) rebuild model & data once
        model = build_model(ModelName.MNIST_MLP)
        loss_fn = get_loss(LossName.CROSS_ENTROPY)
        X, y = load_dataset(
            DatasetName.MNIST_DIGITS_CLASSIFICATION, limit=None)
        dl = DataLoader(list(zip(torch.from_numpy(X), torch.from_numpy(y))),
                        batch_size=1024)

        theta_mean = snaps[-1]                     # centre grid at final point
        dir1, dir2 = pca.components_              # shape (2, D)

        alphas = np.linspace(-1.2, 1.2, grid_pts)
        betas = np.linspace(-1.2, 1.2, grid_pts)
        loss_mat = np.zeros((grid_pts, grid_pts))

        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                theta = theta_mean + a*dir1 + b*dir2
                assign_flat_to_params(theta, model.parameters())
                # full-batch loss:
                with torch.no_grad():
                    running = 0.0
                    for xb, yb in dl:
                        logits = model(xb)
                        running += loss_fn(logits, yb).item() * xb.size(0)
                loss_mat[j, i] = running / len(dl.dataset)

        # 4b) contour plot under the trajectory
        plt.figure(figsize=(5, 4))
        CS = plt.contour(alphas, betas, loss_mat, levels=30, cmap="coolwarm")
        plt.clabel(CS, fontsize=6, inline=True)
        # re-plot trajectory in αβ coords
        AB = np.stack([np.zeros(len(proj)), np.zeros(len(proj))], axis=1)
        # solve proj = theta_mean + a*dir1 + b*dir2  → coefficients (a,b)
        for k, theta in enumerate(snaps):
            coef = np.linalg.lstsq(
                np.stack([dir1, dir2], axis=1), theta-theta_mean, rcond=None)[0]
            AB[k] = coef
        plt.plot(AB[:, 0], AB[:, 1], c="black", marker='o', markersize=2)
        plt.title("Loss surface in PCA plane")
        plt.xlabel("α (PC-1)")
        plt.ylabel("β (PC-2)")
        plt.tight_layout()
        plt.savefig(run_dir / "pca_loss_surface.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=pathlib.Path,
                    help="folder that contains snapshots.pkl")
    ap.add_argument("--surface", action="store_true",
                    help="also compute loss contour on PCA plane")
    ap.add_argument("--grid", type=int, default=35,
                    help="grid points per axis for loss surface")
    args = ap.parse_args()
    main(args.run_dir, draw_loss_surface=args.surface, grid_pts=args.grid)
