from enum import Enum, auto
from dataclasses import dataclass


class DatasetName(Enum):
    MNIST_DIGITS = auto()
    LINEAR_REGRESSION = auto()
    TWO_MOONS_CLASSIFICATION = auto()


class ModelName(Enum):
    MNIST_MLP = auto()
    LINEAR = auto()
    TINY_MLP = auto()  # for two moons classification


class LossName(Enum):
    CROSS_ENTROPY = auto()
    MSE = auto()
    BCE = auto()  # binary cross entropy (for two moons)

# ---------------- new, expressive solver enum ------------------------


class OptimizerName(Enum):
    ADAM = auto()
    L_BFGS = auto()
    TRUST_REGION_NEWTON = auto()
    INTERIOR_POINT = auto()
    GRADIENT_DESCENT = auto()


@dataclass
class HyperParameterSet:
    learning_rate: float = 0.05
    epochs: int = 10
    batch_size: int = 128
    seed: int = 42
    use_cuda: bool = True
    snapshot_every: int = 1
    precompute_surface: bool = False  # Flag to control loss surface precomputation


@dataclass
class ExperimentConfig:
    dataset: DatasetName
    model: ModelName
    loss: LossName
    optimizer_choice: OptimizerName
    hyper_parameters: HyperParameterSet
