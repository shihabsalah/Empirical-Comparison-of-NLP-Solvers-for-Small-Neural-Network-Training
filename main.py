from config import (
    ExperimentConfig, HyperParameterSet,
    DatasetName, ModelName, LossName, OptimizerName
)
from trainer import Trainer

# ---------- Fair benchmarking configurations ----------
# Full-batch size and epochs common to epoch-style optimizers
COMMON_BATCH_SIZE = 60000
COMMON_EPOCHS = 20

# Standard Gradient Descent baseline
#   Full-batch Gradient Descent: simple per-epoch update w ← w − lr·∇L.
#   learning_rate=0.01 chosen as a conservative step for stable convergence.
#   epochs=COMMON_EPOCHS and batch_size=COMMON_BATCH_SIZE ensure each epoch sees full data.
cfg_gd = ExperimentConfig(
    dataset=DatasetName.MNIST_DIGITS,
    model=ModelName.MNIST_MLP,
    loss=LossName.CROSS_ENTROPY,
    optimizer_choice=OptimizerName.GRADIENT_DESCENT,
    hyper_parameters=HyperParameterSet(
        learning_rate=0.01,
        epochs=COMMON_EPOCHS,
        seed=42,
        use_cuda=True,
        snapshot_every=1,
        batch_size=COMMON_BATCH_SIZE,
    ),
)
# Adam optimizer
#   Adaptive Moment Estimation: uses running mean/variance of gradients for adaptive steps.
#   learning_rate=0.05 scales to match gradient-descent update size in expectation.
#   Same epoch count and batch size to fairly compare convergence behavior.
cfg_adam = ExperimentConfig(
    dataset=DatasetName.MNIST_DIGITS,
    model=ModelName.MNIST_MLP,
    loss=LossName.CROSS_ENTROPY,
    optimizer_choice=OptimizerName.ADAM,
    hyper_parameters=HyperParameterSet(
        learning_rate=0.05,
        epochs=COMMON_EPOCHS,
        seed=42,
        use_cuda=True,
        snapshot_every=1,
        batch_size=COMMON_BATCH_SIZE,
    ),
)
# L-BFGS optimizer (full-batch)
#   Limited-memory Broyden–Fletcher–Goldfarb–Shanno: quasi-Newton method for optimization.
#   learning_rate=1.0 chosen to match typical scaling for second-order methods.
#   Full-batch ensures deterministic updates for fair comparison.
cfg_lbfgs = ExperimentConfig(
    dataset=DatasetName.MNIST_DIGITS,
    model=ModelName.MNIST_MLP,
    loss=LossName.CROSS_ENTROPY,
    optimizer_choice=OptimizerName.L_BFGS,
    hyper_parameters=HyperParameterSet(
        learning_rate=1.0,
        epochs=COMMON_EPOCHS,
        seed=42,
        use_cuda=True,
        snapshot_every=1,
        batch_size=COMMON_BATCH_SIZE,
    ),
)
# Trust-Region Newton optimizer (full-batch)
#   Trust-region Newton method: uses second-order curvature for optimization.
#   learning_rate=1.0 chosen to match typical scaling for second-order methods.
#   Full-batch ensures deterministic updates for fair comparison.
cfg_trn = ExperimentConfig(
    dataset=DatasetName.MNIST_DIGITS,
    model=ModelName.MNIST_MLP,
    loss=LossName.CROSS_ENTROPY,
    optimizer_choice=OptimizerName.TRUST_REGION_NEWTON,
    hyper_parameters=HyperParameterSet(
        learning_rate=1.0,
        epochs=COMMON_EPOCHS,
        seed=42,
        use_cuda=True,
        snapshot_every=1,
        batch_size=COMMON_BATCH_SIZE,
    ),
)
# Interior-Point solver (one-shot full-batch)
#   Interior-point method: solves optimization as a constrained problem.
#   learning_rate=1.0 chosen to match typical scaling for second-order methods.
#   epochs=1 ensures one-shot optimization for fair comparison.
cfg_ip = ExperimentConfig(
    dataset=DatasetName.MNIST_DIGITS,
    model=ModelName.MNIST_MLP,
    loss=LossName.CROSS_ENTROPY,
    optimizer_choice=OptimizerName.INTERIOR_POINT,
    hyper_parameters=HyperParameterSet(
        learning_rate=1.0,
        epochs=1,
        seed=42,
        use_cuda=True,
        snapshot_every=1,
        batch_size=COMMON_BATCH_SIZE,
    ),
)

# Run all benchmarks
Trainer(cfg_gd).run()
Trainer(cfg_adam).run()
Trainer(cfg_lbfgs).run()
Trainer(cfg_trn).run()
Trainer(cfg_ip).run()
