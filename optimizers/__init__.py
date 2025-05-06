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
from optimizers.lbfgs_wrapper import LBFGSWrapper
from optimizers.trust_region_newton_wrapper import TrustRegionNewtonWrapper
from optimizers.slsqp_wrapper import SLSQPWrapper
from optimizers.ipopt_wrapper import IpoptWrapper
from optimizers.admm_wrapper import ADMMWrapper
from optimizers.cmaes_wrapper import CMAESWrapper
from optimizers.kfac_wrapper import KFACWrapper


def get_optimizer(
    choice: OptimizerName,
    model_parameters: Iterator[torch.nn.Parameter],
    learning_rate: float,
) -> BaseOptimizer:
    if choice is OptimizerName.ADAM:
        return AdamWrapper(model_parameters, learning_rate)
    if choice is OptimizerName.L_BFGS:
        return LBFGSWrapper(model_parameters, learning_rate)
    if choice is OptimizerName.TRUST_REGION_NEWTON:
        return TrustRegionNewtonWrapper(model_parameters)
    if choice is OptimizerName.SLSQP:
        return SLSQPWrapper(model_parameters)
    if choice is OptimizerName.IPOPT:
        return IpoptWrapper(model_parameters)
    if choice is OptimizerName.ADMM:
        return ADMMWrapper(model_parameters, learning_rate)
    if choice is OptimizerName.CMA_ES:
        return CMAESWrapper(model_parameters)
    if choice is OptimizerName.KFAC:
        return KFACWrapper(model_parameters, learning_rate)
    raise ValueError(f"Unsupported optimizer {choice}")
