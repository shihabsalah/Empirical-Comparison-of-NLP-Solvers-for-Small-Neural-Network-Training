from __future__ import annotations
import abc
import numpy as np
import torch
import torch.nn as nn
from config import ModelName


class BaseModel(abc.ABC):
    @abc.abstractmethod
    def forward(self, x): ...

    @abc.abstractmethod
    def parameters(self): ...

    def predict(self, x):
        return self.forward(x)


# ─────────────────────────────────────────────────────────────────────────────
# NumPy linear model (for regression demo)
# ─────────────────────────────────────────────────────────────────────────────
class LinearModel(BaseModel):
    """y = w·x (no bias) – pure NumPy."""

    def __init__(self):
        self.w = np.random.uniform(-1, 1, size=(1,))

    def forward(self, x: np.ndarray):
        return x * self.w

    def parameters(self):
        # wrap in tensor so torch optimisers work
        return [torch.tensor(self.w, requires_grad=True)]


# ─────────────────────────────────────────────────────────────────────────────
# Tiny 2‑5‑1 MLP (for two‑moons binary classification)
# ─────────────────────────────────────────────────────────────────────────────
class TinyMLP(BaseModel, nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Linear(2, 5), nn.Sigmoid(),
            nn.Linear(5, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):  # convert if caller used NumPy
            x = torch.from_numpy(x)
        return self.net(x)

    def parameters(self):  # type: ignore[override]
        return self.net.parameters()


# ─────────────────────────────────────────────────────────────────────────────
# MNIST MLP 784‑128‑10 (no softmax – CrossEntropyLoss expects raw logits)
# ─────────────────────────────────────────────────────────────────────────────
class MnistMLP(BaseModel, nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 10),                # 10 digits
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return self.net(x)

    def parameters(self):  # type: ignore[override]
        return self.net.parameters()


# helper factory --------------------------------------------------------------

def build_model(name: ModelName):
    if name == ModelName.LINEAR:
        return LinearModel()
    if name == ModelName.TINY_MLP:
        return TinyMLP()
    if name == ModelName.MNIST_MLP:
        return MnistMLP()
    raise ValueError(name)
