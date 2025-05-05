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


class LinearModel(BaseModel):
    """y = w·x (no bias). Pure NumPy implementation."""

    def __init__(self):
        self.w = np.random.uniform(-1, 1, (1,))

    def forward(self, x: np.ndarray):
        return x * self.w

    def parameters(self):
        # return a list so torch‑style optimisers still iterate nicely
        return [torch.tensor(self.w, requires_grad=True)]


class TinyMLP(BaseModel, nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Linear(2, 5), nn.Sigmoid(),
            nn.Linear(5, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return self.net(x)

    def parameters(self):  # type: ignore[override]
        return self.net.parameters()

# helper factory -----------------------------------------------------


def build_model(name: ModelName):
    if name == ModelName.LINEAR:
        return LinearModel()
    if name == ModelName.TINY_MLP:
        return TinyMLP()
    raise ValueError(name)
