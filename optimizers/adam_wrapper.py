"""
optimizers/adam_wrapper.py
--------------------------
Thin wrapper around `torch.optim.Adam`.

The Adam algorithm
------------------
• Per parameter θ:
    m ← β₁·m + (1–β₁)·g_t        # 1st-moment estimate  (mean of grads)
    v ← β₂·v + (1–β₂)·g_t²       # 2nd-moment estimate (uncentered var)
    m̂ = m / (1–β₁ᵗ),  v̂ = v / (1–β₂ᵗ)   # bias-correction
    θ ← θ – α · m̂ / (√v̂ + ε)
• Default β₁ = 0.9, β₂ = 0.999, ε = 1e-8.

The wrapper only forwards `.zero_grad()` and `.step(closure)`.
"""

from typing import Iterator, Callable
import torch
from optimizers.base_optimizer import BaseOptimizer


class AdamWrapper(BaseOptimizer):
    """A minimal, explicit wrapper so the Trainer sees a uniform API."""

    def __init__(
        self,
        model_parameters: Iterator[torch.nn.Parameter],
        learning_rate: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self._torch_optimizer = torch.optim.Adam(
            params=list(model_parameters),
            lr=learning_rate,
            betas=betas,
            eps=epsilon,
            weight_decay=weight_decay,
        )

    # ------------------------------------------------------------------ API
    def zero_grad(self) -> None:
        self._torch_optimizer.zero_grad(set_to_none=True)

    def step(self, loss_closure: Callable[[], torch.Tensor] | None = None) -> None:
        """
        For Adam the closure is optional (only needed for fancy line-search).
        We call it to stay consistent with LBFGS / Trust-Region wrappers.
        """
        if loss_closure is None:
            self._torch_optimizer.step()
        else:
            loss_value = loss_closure()
            self._torch_optimizer.step()
            return loss_value.item()
