"""
optimizers/trust_constr_wrapper.py
----------------------------------
Interior-point/bound-constrained optimiser using
`scipy.optimize.minimize(method="trust-constr")`.

• Handles smooth objectives, simple bounds, and general equality /
  inequality constraints (we use none here, so it acts like a pure
  interior-point solver).
• One-shot solver  →  is_one_shot_solver = True.
"""

from typing import Iterator, Callable
import numpy as np
import torch
import scipy.optimize as sco
from torch import Tensor

from optimizers.base_optimizer import BaseOptimizer
from utils import Utils


class TrustConstrWrapper(BaseOptimizer):
    is_one_shot_solver: bool = True

    def __init__(self, model_parameters: Iterator[Tensor]) -> None:
        self._parameters = list(model_parameters)

    # ------------------------------------------------------------------
    def run_full_batch_solve(self, full_batch_loss: Callable[[], float]) -> None:
        """Optimise full-batch loss and copy the solution into model."""

        # SciPy objective & gradient over flat NumPy vector
        def objective(x: np.ndarray) -> float:
            Utils.assign_flat_to_params(x, self._parameters)
            return full_batch_loss().item()

        def gradient(x: np.ndarray) -> np.ndarray:
            Utils.assign_flat_to_params(x, self._parameters)
            loss_tensor = full_batch_loss()          # tensor
            for p in self._parameters:
                if p.grad is not None:
                    p.grad.zero_()
            loss_tensor.backward()
            grad_vec = Utils.flatten_params(p.grad for p in self._parameters)
            return grad_vec.astype(np.float64)

        x0 = Utils.flatten_params(self._parameters).astype(np.float64)

        result = sco.minimize(
            fun=objective,
            x0=x0,
            jac=gradient,
            method="trust-constr",
            options={
                "verbose": 0,       # 0 = silent, 3 = full dump
                "gtol": 1e-6,       # gradient-norm tolerance
                "maxiter": 500,
            },
        )

        Utils.assign_flat_to_params(result.x, self._parameters)
        print("[trust-constr] status:", result.message,
              "| iterations:", result.niter)

    # unused for one-shot solvers
    def zero_grad(self) -> None: ...

    def step(self, loss_closure: Callable[[],
             Tensor] | None = None) -> None: ...
