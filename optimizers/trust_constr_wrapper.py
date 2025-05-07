"""
optimizers/trust_constr_wrapper.py
----------------------------------
Interior-point / trust-region algorithm built into SciPy
(`method="trust-constr"`).

Marked as `is_one_shot_solver = True`, so the Trainer will call the
`run_full_batch_solve` method exactly once.
"""

from typing import Callable, Iterator
import numpy as np
import torch
import scipy.optimize as sco
from torch import Tensor
from torch.amp import autocast

from optimizers.base_optimizer import BaseOptimizer
from utils import Utils


class InteriorPointTrustConstrWrapper(BaseOptimizer):
    is_one_shot_solver: bool = True          # single solve, no epochs

    def __init__(self, model_parameters: Iterator[Tensor]) -> None:
        self._parameters = list(model_parameters)

    # -------------- one-shot implementation ------------------------------
    def run_full_batch_solve(self, full_loss_closure: Callable[[], Tensor]) -> None:

        def objective(flat_weights: np.ndarray) -> float:
            Utils.assign_flat_to_params(flat_weights, self._parameters)
            return full_loss_closure().item()            # scalar float

        def gradient(flat_weights: np.ndarray) -> np.ndarray:
            Utils.assign_flat_to_params(flat_weights, self._parameters)
            # Mixed precision forward pass on GPU (using new API)
            with autocast(device_type='cuda'):
                loss_tensor = full_loss_closure()            # tensor for autograd
            for p in self._parameters:
                if p.grad is not None:
                    p.grad.zero_()
            loss_tensor.backward()
            grad_vec = Utils.flatten_params(p.grad for p in self._parameters)
            return grad_vec.astype(np.float64)

        # ——— Hessian-vector product callback ————————————————
        def scipy_hessian_vector_product(
            flat_weights: np.ndarray,
            v_np: np.ndarray
        ) -> np.ndarray:
            Utils.assign_flat_to_params(flat_weights, self._parameters)
            # Mixed precision forward pass for first-order gradient (new API)
            with autocast(device_type='cuda'):
                loss_tensor = full_loss_closure()
            # first-order grads with graph
            first_grads = torch.autograd.grad(
                outputs=loss_tensor,
                inputs=self._parameters,
                create_graph=True
            )
            flat_first = torch.cat([g.contiguous().view(-1)
                                   for g in first_grads])
            # Direction vector on same device and dtype
            v = torch.from_numpy(v_np).to(
                flat_first.device, dtype=flat_first.dtype).view_as(flat_first)
            # Mixed precision backward pass for Hessian-vector product (new API)
            with autocast(device_type='cuda'):
                hvp = torch.autograd.grad(
                    outputs=torch.dot(flat_first, v),
                    inputs=self._parameters,
                    create_graph=False
                )
            flat_hvp = torch.cat([h.contiguous().view(-1) for h in hvp])
            return flat_hvp.detach().cpu().numpy()

        initial_flat = Utils.flatten_params(
            self._parameters).astype(np.float64)

        # Call SciPy’s trust-constr solver with Hessian-vector products
        result = sco.minimize(
            fun=objective,
            x0=initial_flat,
            jac=gradient,
            hessp=scipy_hessian_vector_product,
            method="trust-constr",
            # reduced maxiter and looser tolerance for faster runtime
            options={"gtol": 1e-3, "verbose": 0, "maxiter": 50},
        )

        Utils.assign_flat_to_params(result.x, self._parameters)
        print("[trust-constr] iterations:",
              result.niter, " message:", result.message)

    # epoch-style API not used but must exist
    def zero_grad(self) -> None: ...

    def step(self, loss_closure: Callable[[],
             Tensor] | None = None) -> None: ...
