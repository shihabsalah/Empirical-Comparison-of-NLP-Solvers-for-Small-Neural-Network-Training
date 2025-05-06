import pickle
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader
from utils import Utils
from models import build_model
from losses import get_loss
from config import ModelName, LossName
from data import load_dataset
from config import DatasetName

# ------------------------------------------------------------
import torch


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

    # 4) optional loss-surface grid -----------------------------------------
    if draw_loss_surface:
        print("Computing loss grid … this can take a few minutes")
        device = Utils.choose_device(request_cuda=args.gpu)
        print("Using device:", device)

        # -- (re)build model & data once ------------------------------------
        model = build_model(ModelName.MNIST_MLP).to(device)
        loss_fn = get_loss(LossName.CROSS_ENTROPY).to(device)
        X, y = load_dataset(DatasetName.MNIST_DIGITS_CLASSIFICATION)
        X_pt = torch.from_numpy(X).to(device)
        y_pt = torch.from_numpy(y).to(device)
        full_dl = DataLoader(list(zip(X_pt, y_pt)),
                             batch_size=4096,  # big batch, GPU friendly
                             pin_memory=False, shuffle=False)

        # -- α,β coordinates for *all* snapshots ----------------------------
        #   proj = PCA-transform of snaps  → already computed above
        alpha_all, beta_all = proj[:, 0], proj[:, 1]

        # padding so the grid is a bit larger than the trajectory box
        pad = 0.5
        alpha_min, alpha_max = alpha_all.min() - pad, alpha_all.max() + pad
        beta_min,  beta_max = beta_all.min() - pad, beta_all.max() + pad

        grid_pts = grid_pts            # comes from CLI arg
        alphas = np.linspace(alpha_min, alpha_max, grid_pts)
        betas = np.linspace(beta_min,  beta_max,  grid_pts)
        loss_mat = np.zeros((grid_pts, grid_pts))

        dir1, dir2 = pca.components_           # (2, D)
        theta_mean = pca.mean_                 # centre of PCA coords

        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                theta = theta_mean + a*dir1 + b*dir2
                Utils.assign_flat_to_params(theta, model.parameters())

                with torch.no_grad():
                    running = 0.0
                    for xb, yb in full_dl:
                        logits = model(xb)
                        running += loss_fn(logits, yb).item() * xb.size(0)
                loss_mat[j, i] = running / len(full_dl.dataset)

        # -- draw the contour + trajectory ---------------------------------
        plt.figure(figsize=(5, 4))
        CS = plt.contour(alphas, betas, loss_mat, levels=30, cmap="coolwarm")
        plt.clabel(CS, inline=True, fontsize=6)

        plt.plot(alpha_all, beta_all,
                 color="black", marker='o', markersize=3, linewidth=1.5)
        plt.scatter(alpha_all[-1], beta_all[-1],
                    c="red",  s=40, label="final")
        plt.title("Loss surface in PCA plane")
        plt.xlabel("α (PC-1)")
        plt.ylabel("β (PC-2)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "pca_loss_surface.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true",
                    help="evaluate loss grid on the first CUDA GPU")
    ap.add_argument("run_dir", type=pathlib.Path,
                    help="folder that contains snapshots.pkl")
    ap.add_argument("--surface", action="store_true",
                    help="also compute loss contour on PCA plane")
    ap.add_argument("--grid", type=int, default=35,
                    help="grid points per axis for loss surface")
    args = ap.parse_args()
    main(args.run_dir, draw_loss_surface=args.surface, grid_pts=args.grid)
