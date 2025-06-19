"""
optimizers.__init__
-------------------
Factory that returns an object with an interface similar to PyTorch's
torch.optim.Optimizer:  `.step()` and `.zero_grad()` at minimum.
"""

from typing import Iterator
import torch
from optimizers.base_optimizer import BaseOptimizer
from config import OptimizerName
from optimizers.adam_wrapper import AdamWrapper
from optimizers.lbfgs_wrapper import LimitedMemoryBFGSWrapper
from optimizers.trust_region_newton_wrapper import TrustRegionNewtonWrapper
from optimizers.trust_constr_wrapper import InteriorPointTrustConstrWrapper
from optimizers.gradient_descent_wrapper import GradientDescentWrapper


def get_optimizer(
    choice: OptimizerName,
    model_parameters: Iterator[torch.nn.Parameter],
    learning_rate: float,
) -> BaseOptimizer:
    if choice is OptimizerName.ADAM:
        return AdamWrapper(model_parameters, learning_rate)
    if choice is OptimizerName.L_BFGS:
        return LimitedMemoryBFGSWrapper(model_parameters, learning_rate)
    if choice is OptimizerName.TRUST_REGION_NEWTON:
        return TrustRegionNewtonWrapper(model_parameters)
    if choice is OptimizerName.INTERIOR_POINT:
        return InteriorPointTrustConstrWrapper(model_parameters)
    if choice is OptimizerName.GRADIENT_DESCENT:
        return GradientDescentWrapper(model_parameters, learning_rate)

    raise ValueError(f"Unsupported optimizer {choice}")
