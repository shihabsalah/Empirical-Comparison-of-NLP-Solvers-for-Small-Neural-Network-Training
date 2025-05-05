from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto


class DatasetName(Enum):
    LINEAR_REGRESSION = auto()
    TWO_MOONS_CLASSIFICATION = auto()


class ModelName(Enum):
    LINEAR = auto()
    TINY_MLP = auto()


class LossName(Enum):
    MSE = auto()
    BCE = auto()


class OptimizerName(Enum):
    SGD = auto()
    ADAM = auto()
    LBFGS = auto()


@dataclass
class HyperParams:
    learning_rate: float = 0.01
    epochs: int = 1000
    batch_size: int | None = None  # None ⇒ full‑batch
    seed: int = 42


@dataclass
class ExperimentConfig:
    dataset: DatasetName
    model: ModelName
    loss: LossName
    optimizer: OptimizerName
    hp: HyperParams = field(default_factory=HyperParams)
