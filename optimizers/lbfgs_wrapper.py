"""
optimizers/lbfgs_wrapper.py
---------------------------
Limited-memory BFGS quasi-Newton optimiser.

Algorithm idea
--------------
BFGS builds an approximation H_k ≈ (∇²f)⁻¹ using only gradient differences:
    s_k = x_{k+1} – x_k
    y_k = g_{k+1} – g_k
    ρ_k = 1 / (y_kᵀ s_k)
    H_{k+1} = (I – ρ_k s_k y_kᵀ) H_k (I – ρ_k y_k s_kᵀ) + ρ_k s_k s_kᵀ
"Limited-memory" keeps only the last *m* pairs (default m = 10) so we never
store a full dense Hessian.

PyTorch’s `LBFGS` already implements this with a built-in strong-Wolfe
line-search.  The wrapper merely adapts its API to `BaseOptimizer`.
"""

from typing import Iterator, Callable
import torch
from optimizers.base_optimizer import BaseOptimizer


class LBFGSWrapper(BaseOptimizer):
    """Expose zero_grad / step(closure) while hiding LBFGS specifics."""

    def __init__(
        self,
        model_parameters: Iterator[torch.nn.Parameter],
        learning_rate: float = 1.0,
        history_size: int = 10,
        max_line_search_steps: int = 20,
    ) -> None:
        self._inner_opt = torch.optim.LBFGS(
            params=list(model_parameters),
            lr=learning_rate,            # step length multiplier
            max_iter=max_line_search_steps,
            history_size=history_size,
            line_search_fn="strong_wolfe",
        )

    # ------------------------ BaseOptimizer interface ------------------------
    def zero_grad(self) -> None:
        self._inner_opt.zero_grad(set_to_none=True)

    def step(self, loss_closure: Callable[[], torch.Tensor] | None = None) -> None:
        """
        LBFGS **requires** a closure that:
          1. recomputes the forward pass,
          2. calls .backward() to get fresh gradients,
          3. returns the loss tensor.
        """
        if loss_closure is None:
            raise ValueError("LBFGSWrapper.step needs a closure.")
        self._inner_opt.step(loss_closure)   # PyTorch handles line-search
