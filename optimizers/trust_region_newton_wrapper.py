"""
optimizers/trust_region_newton_wrapper.py
-----------------------------------------
Trust-Region Newton solver via SciPy.

Trust-Region basics
-------------------
At x_k build quadratic model   m_k(d) = gᵀ d + ½ dᵀ H d
Minimise it **subject to**  ||d|| ≤ Δ_k   (trust-region radius).

▪ If the ratio  ρ = (f(x_k) – f(x_k+d)) /  (m_k(0) – m_k(d))
      is good (ρ > η₁) → accept step and maybe enlarge Δ_k.
▪ Else (ρ small)       → reject step and shrink Δ_k.

SciPy’s `"trust-ncg"` does:
    * Conjugate-gradient on the quadratic sub-problem
    * Automatic Powell–Dog-Leg fallback when H indefinite
"""

from typing import Iterator, Callable
import numpy as np
import torch
from torch import Tensor
from optimizers.base_optimizer import BaseOptimizer
from utils import Utils

import scipy.optimize as sco


class TrustRegionNewtonWrapper(BaseOptimizer):
    """
    One-shot solver: after `run_full_batch_solve` finishes the model parameters
    have been replaced with the minimiser found by SciPy.
    """

    def __init__(self, model_parameters: Iterator[Tensor]) -> None:
        # Store *references* to the original parameter tensors
        self._params = list(model_parameters)

    # This optimiser is one-shot → normal epoch loop will skip over it
    def zero_grad(self) -> None: ...

    def step(self, loss_closure: Callable[[],
             Tensor] | None = None) -> None: ...

    # ------------------------------------------------------------------ core
    def run_full_batch_solve(self, full_batch_loss: Callable[[], float]) -> None:
        """
        full_batch_loss(): forward pass over *entire* training set,
                           returns scalar Python float.
        """

        # -------- helper that SciPy will minimise -------------------------
        def scipy_objective(flat_vector: np.ndarray) -> float:
            Utils.assign_flat_to_params(flat_vector, self._params)
            return full_batch_loss()

        def scipy_gradient(flat_vector: np.ndarray) -> np.ndarray:
            Utils.assign_flat_to_params(flat_vector, self._params)
            # Compute gradients with PyTorch autograd
            loss_tensor = torch.as_tensor(
                full_batch_loss(), dtype=torch.float32)
            # zero grads
            for p in self._params:
                if p.grad is not None:
                    p.grad.zero_()
            loss_tensor.backward()
            grad_vector = Utils.flatten_params(p.grad for p in self._params)
            return grad_vector.astype(np.float64)

        initial_flat = Utils.flatten_params(self._params).astype(np.float64)

        result = sco.minimize(
            fun=scipy_objective,
            x0=initial_flat,
            jac=scipy_gradient,
            method="trust-ncg",
            options={"gtol": 1e-6, "maxiter": 200},
        )

        # copy result back into the model
        Utils.assign_flat_to_params(result.x, self._params)
        print("[TrustRegionNewton] SciPy status:", result.message)
