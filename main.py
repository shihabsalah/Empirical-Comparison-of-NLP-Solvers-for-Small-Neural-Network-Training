from config import (
    ExperimentConfig, HyperParameterSet,
    DatasetName, ModelName, LossName, OptimizerName
)
from trainer import Trainer
import argparse
from tqdm import tqdm
import time

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Run neural network optimizer benchmark experiments")
parser.add_argument("--surface", action="store_true",
                    help="Pre-compute loss surfaces during training (significantly increases runtime)")
args = parser.parse_args()

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
        precompute_surface=args.surface,  # Set from command line
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
        precompute_surface=args.surface,  # Set from command line
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
        precompute_surface=args.surface,  # Set from command line
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
        precompute_surface=args.surface,  # Set from command line
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
        precompute_surface=args.surface,  # Set from command line
    ),
)

# Setup all configurations
configs = [
    ("Gradient Descent", cfg_gd),
    ("Adam", cfg_adam),
    ("L-BFGS", cfg_lbfgs),
    ("Trust-Region Newton", cfg_trn),
    ("Interior Point", cfg_ip)
]

# Estimate total work units (epochs across all optimizers + 1 for interior point)
total_work = sum(cfg.hyper_parameters.epochs for _, cfg in configs)

# Setup progress bar
pbar = tqdm(total=total_work, desc="Overall Progress", unit="epoch")

# Run all benchmarks
for name, config in configs:
    start_time = time.time()
    pbar.set_description(f"Running {name}")

    # Create a trainer with the current config
    trainer = Trainer(config)

    # Run the trainer
    trainer.run()

    # Update progress bar based on epochs
    pbar.update(config.hyper_parameters.epochs if not config.optimizer_choice ==
                OptimizerName.INTERIOR_POINT else 1)

    # Show time taken for this optimizer
    elapsed = time.time() - start_time
    pbar.write(f"✓ Completed {name} in {elapsed:.1f}s")

pbar.close()
print("\nAll experiments completed!")
