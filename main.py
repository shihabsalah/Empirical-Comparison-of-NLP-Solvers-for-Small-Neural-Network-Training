from config import DatasetName, ModelName, LossName, OptimizerName, HyperParams, ExperimentConfig
from trainer import Trainer

if __name__ == "__main__":
    cfg = ExperimentConfig(
        dataset=DatasetName.TWO_MOONS_CLASSIFICATION,
        model=ModelName.TINY_MLP,
        loss=LossName.BCE,
        optimizer=OptimizerName.SGD,
        hp=HyperParams(learning_rate=0.1, epochs=1000),
    )
    Trainer(cfg).fit()
