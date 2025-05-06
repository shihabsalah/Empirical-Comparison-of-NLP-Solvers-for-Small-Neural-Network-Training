from config import (ExperimentConfig, HyperParams,
                    DatasetName, ModelName, LossName, OptimizerName)
from trainer import Trainer

cfg = ExperimentConfig(
    dataset=DatasetName.MNIST_DIGITS_CLASSIFICATION,
    model=ModelName.MNIST_MLP,
    loss=LossName.CROSS_ENTROPY,
    optimizer=OptimizerName.SGD,
    hyperParameters=HyperParams(learning_rate=0.05,
                                epochs=10,
                                batch_size=128,
                                seed=42),
)
Trainer(cfg).fit()
