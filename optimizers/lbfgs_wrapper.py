"""
optimizers/lbfgs_wrapper.py
---------------------------
Limited-memory BFGS (secant) optimiser.

Key idea
========
Keep the last *m* (parameter-difference, gradient-difference) pairs
to build a low-rank approximation of the inverse Hessian, so every
step is automatically rescaled by estimated curvature.
"""

from typing import Callable, Iterator
import torch
from torch import Tensor

from optimizers.base_optimizer import BaseOptimizer


class LimitedMemoryBFGSWrapper(BaseOptimizer):
    is_one_shot_solver: bool = False           # called each mini-batch / epoch

    def __init__(
        self,
        model_parameters: Iterator[Tensor],
        learning_rate: float = 1.0,
        history_size: int = 10,
        line_search_max_steps: int = 20,
    ) -> None:
        self._torch_lbfgs = torch.optim.LBFGS(
            params=list(model_parameters),
            lr=learning_rate,
            max_iter=line_search_max_steps,     # inner line-search iters
            history_size=history_size,
            line_search_fn="strong_wolfe",
        )

    # --------------- BaseOptimizer interface -----------------------------
    def zero_grad(self) -> None:
        self._torch_lbfgs.zero_grad(set_to_none=True)

    def step(self, loss_closure: Callable[[], Tensor] | None = None) -> None:
        if loss_closure is None:
            raise ValueError("LimitedMemoryBFGSWrapper.step expects a closure")
        self._torch_lbfgs.step(loss_closure)
