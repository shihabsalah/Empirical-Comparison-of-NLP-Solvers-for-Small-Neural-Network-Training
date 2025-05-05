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
    # placeholders – will wire these later
    ADAM = auto()
    LBFGS = auto()
