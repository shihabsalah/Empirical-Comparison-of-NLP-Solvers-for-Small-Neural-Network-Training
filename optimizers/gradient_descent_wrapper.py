"""
optimizers/gradient_descent_wrapper.py
--------------------------------------
Plain full-batch Gradient Descent.

update equation
---------------
    parameter ← parameter – learning_rate · parameter.grad
"""

from typing import Iterator, Callable
import torch
from torch import Tensor
from optimizers.base_optimizer import BaseOptimizer


class GradientDescentWrapper(BaseOptimizer):
    is_one_shot_solver: bool = False       # epoch-style

    def __init__(self,
                 model_parameters: Iterator[Tensor],
                 learning_rate: float = 1e-2) -> None:
        self._parameters = list(model_parameters)
        self.learning_rate = learning_rate

    # BaseOptimizer API ---------------------------------------------------
    def zero_grad(self) -> None:
        for parameter in self._parameters:
            if parameter.grad is not None:
                parameter.grad.zero_()

    def step(self, loss_closure: Callable[[], Tensor] | None = None) -> None:
        """
        • Requires a closure that already called .backward()
        • Applies one vanilla GD update over *all* parameters.
        """
        if loss_closure is None:
            raise ValueError("GradientDescentWrapper.step needs a closure")

        # Ensure fresh gradients
        loss_closure()

        with torch.no_grad():
            for parameter in self._parameters:
                if parameter.grad is not None:
                    parameter -= self.learning_rate * parameter.grad
