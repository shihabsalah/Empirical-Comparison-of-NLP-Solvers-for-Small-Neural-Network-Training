from __future__ import annotations
import random
import numpy as np
import torch
from config import ExperimentConfig, ModelName
from data import load_dataset
from models import build_model
from losses import get_loss
from optimizers import get_optimizer


class Trainer:
    def __init__(self, cfg: ExperimentConfig):
        # reproducibility
        random.seed(cfg.hp.seed)
        np.random.seed(cfg.hp.seed)
        torch.manual_seed(cfg.hp.seed)

        # data
        self.X_np, self.y_np = load_dataset(cfg.dataset)

        # model + loss + optim
        self.model = build_model(cfg.model)
        self.loss_fn = get_loss(cfg.loss)
        self.optim = get_optimizer(
            cfg.optimizer, self.model.parameters(), cfg.hp.learning_rate)
        self.epochs = cfg.hp.epochs

    # ------------------------------------------------------------------
    def fit(self):
        losses: list[float] = []
        X_tensor = torch.from_numpy(self.X_np)
        y_tensor = torch.from_numpy(self.y_np)
        for epoch in range(self.epochs):
            self.optim.zero_grad()
            outputs = self.model.forward(X_tensor)
            loss = self.loss_fn(outputs, y_tensor)
            loss.backward()
            self.optim.step()
            losses.append(float(loss.item()))
            if epoch % max(1, self.epochs // 10) == 0:
                print(f"epoch {epoch:4d}  loss {loss.item():.6f}")
        print("Training finished → final loss:", losses[-1])
        return losses
