from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto


class DatasetName(Enum):
    LINEAR_REGRESSION = auto()
    TWO_MOONS_CLASSIFICATION = auto()
    MNIST_DIGITS_CLASSIFICATION = auto()


class ModelName(Enum):
    LINEAR = auto()
    TINY_MLP = auto()
    MNIST_MLP = auto()


class LossName(Enum):
    MSE = auto()
    BCE = auto()
    CROSS_ENTROPY = auto()


class OptimizerName(Enum):
    SGD = auto()
    ADAM = auto()
    LBFGS = auto()


@dataclass
class HyperParams:
    learning_rate: float = 0.05
    epochs:        int = 10
    batch_size:    int = 128
    seed:          int = 42
    snapshot_every: int = 1  # default = each epoch


@dataclass
class ExperimentConfig:
    dataset:            DatasetName
    model:              ModelName
    loss:               LossName
    optimizer:          OptimizerName
    hyperParameters:    HyperParams
