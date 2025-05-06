"""
optimizers/admm_wrapper.py
--------------------------
Very simple ADMM (Alternating Direction Method of Multipliers) for the
unconstrained case rewritten in a splitting form:

   minimise  f(w)   subject to  w = z

Augmented-Lagrangian:
   Lρ(w, z, u) = f(w) + (ρ/2)||w - z + u||²

Algorithm:
   while not converged:
       1) w-step : minimise f(w) + (ρ/2)||w - z + u||²  (one gradient step)
       2) z-step : z ← w + u                            (closed form)
       3) u-step : u ← u + (w - z)

We treat ADMM as **epoch-style** (is_one_shot_solver = False) so Trainer
calls .step() every mini-batch.  Inside .step() we execute one *outer*
ADMM iteration, which itself performs one gradient step on w.
"""

from typing import Iterator, Callable

import torch
from torch import Tensor
from optimizers.base_optimizer import BaseOptimizer


class ADMMWrapper(BaseOptimizer):
    is_one_shot_solver: bool = False  # called each mini-batch

    def __init__(
        self,
        model_parameters: Iterator[Tensor],
        learning_rate: float = 1e-2,
        rho: float = 1.0,
    ) -> None:
        self._params = list(model_parameters)
        # Shadow copy z and dual variable u (same shapes as parameters)
        self._z_vars = [p.detach().clone() for p in self._params]
        self._u_vars = [torch.zeros_like(p) for p in self._params]
        self._lr = learning_rate
        self._rho = rho

    # ------------------------------------------------------------------ API
    def zero_grad(self) -> None:
        for p in self._params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self, loss_closure: Callable[[], Tensor] | None = None) -> None:
        """
        Executes **one outer ADMM iteration**.
        1) one gradient descent update on w
        2) z update (simple copy)
        3) dual update
        """
        if loss_closure is None:
            raise ValueError("ADMMWrapper.step requires a closure")

        # 1) w-step : grad of f + ρ(w - z + u)
        loss_tensor = loss_closure()         # computes grads ∇f(w)
        with torch.no_grad():
            for p, z, u in zip(self._params, self._z_vars, self._u_vars):
                p.grad = p.grad + self._rho * (p - z + u)   # total gradient
                p -= self._lr * p.grad                      # gradient step

        # 2) z-step : z ← w + u
        with torch.no_grad():
            for z, p, u in zip(self._z_vars, self._params, self._u_vars):
                z.copy_(p + u)

        # 3) u-step : u ← u + (w - z)
        with torch.no_grad():
            for u, p, z in zip(self._u_vars, self._params, self._z_vars):
                u += p - z
