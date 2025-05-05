from __future__ import annotations
import random
import numpy as np
import torch
from config import ExperimentConfig, ModelName
from data import load_dataset
from models import build_model
from losses import get_loss
from optimizers import get_optimizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from flatten import flatten_params
import pickle
import time
import pathlib


class Trainer:
    def __init__(self, cfg: ExperimentConfig):
        # reproducibility
        random.seed(cfg.hp.seed)
        np.random.seed(cfg.hp.seed)
        torch.manual_seed(cfg.hp.seed)

        X, y = load_dataset(cfg.dataset, limit=None)   # full MNIST by default
        self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(
            X, y, test_size=0.2, random_state=cfg.hp.seed, stratify=y)

        self.model = build_model(cfg.model)
        self.loss_fn = get_loss(cfg.loss)
        self.optim = get_optimizer(cfg.optimizer,
                                   self.model.parameters(),
                                   cfg.hp.learning_rate)
        self.epochs = cfg.hp.epochs
        self.batch_size = cfg.hp.batch_size

        # build PyTorch DataLoaders
        tr_ds = TensorDataset(torch.from_numpy(self.Xtr),
                              torch.from_numpy(self.ytr))
        te_ds = TensorDataset(torch.from_numpy(self.Xte),
                              torch.from_numpy(self.yte))
        self.tr_loader = DataLoader(tr_ds, batch_size=self.batch_size,
                                    shuffle=True)
        self.te_loader = DataLoader(te_ds, batch_size=1024)

        self.snapshots: list[np.ndarray] = []
        self.snapshot_every = cfg.hp.snapshot_every or 1  # default = each epoch

    def _epoch(self, epoch_idx: int):
        self.model.train()
        total, correct, running_loss = 0, 0, 0.0
        for xb, yb in self.tr_loader:
            self.optim.zero_grad()
            logits = self.model(xb)
            loss = self.loss_fn(logits, yb)
            loss.backward()
            self.optim.step()

            running_loss += loss.item() * xb.size(0)
            if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
            total += xb.size(0)

        # -------- snapshot logic ------------
        if epoch_idx % self.snapshot_every == 0:
            self.snapshots.append(flatten_params(self.model.parameters()))
        # ------------------------------------

        return running_loss/total, correct/total

    def _eval(self):
        self.model.eval()
        total, correct, running_loss = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in self.te_loader:
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                running_loss += loss.item() * xb.size(0)
                if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                total += xb.size(0)
        return running_loss/total, correct/total

    # ------------------------------------------------------------------
    def fit(self):
        for ep in range(1, self.epochs + 1):
            train_loss, train_acc = self._epoch(ep)
            test_loss,  test_acc = self._eval()
            print(f"ep {ep:02d} | "
                  f"train loss {train_loss:.4f} acc {train_acc*100:5.1f}%  ||  "
                  f"test loss {test_loss:.4f} acc {test_acc*100:5.1f}%")
        run_dir = pathlib.Path("runs") / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "snapshots.pkl", "wb") as f:
            pickle.dump(self.snapshots, f)
