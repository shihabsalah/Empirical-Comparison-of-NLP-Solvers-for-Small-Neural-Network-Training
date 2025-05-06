from config import (
    ExperimentConfig, HyperParameterSet,
    DatasetName, ModelName, LossName, OptimizerName
)
from trainer import Trainer

cfg = ExperimentConfig(
    dataset=DatasetName.MNIST_DIGITS,
    model=ModelName.MNIST_MLP,
    loss=LossName.CROSS_ENTROPY,
    optimizer_choice=OptimizerName.L_BFGS,
    hyper_parameters=HyperParameterSet(learning_rate=0.8, epochs=15)
)
Trainer(cfg).run()
