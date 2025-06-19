"""
Trust-Region Newton optimiser (“trust-ncg”) wrapper for PyTorch models,
using SciPy’s optimizer interface with Hessian-vector products.
"""

from typing import Callable, Iterator
import numpy as np
import torch
import scipy.optimize as scipy_optimize
from torch import Tensor

from optimizers.base_optimizer import BaseOptimizer
from utils import Utils


class TrustRegionNewtonWrapper(BaseOptimizer):
    """
    Epoch-style optimiser: Trainer calls .step() each epoch with a closure
    that recomputes full-batch loss and gradients.
    """

    is_one_shot_solver: bool = False  # we do per-epoch updates

    def __init__(self, model_parameter_iterator: Iterator[Tensor]) -> None:
        # Keep a list of references to all parameters
        self._model_parameter_list = list(model_parameter_iterator)

    def zero_grad(self) -> None:
        # No-op: training loop will zero gradients around the closure
        pass

    def step(self, full_batch_loss_closure: Callable[[], Tensor] | None) -> None:
        if full_batch_loss_closure is None:
            raise ValueError("TrustRegionNewtonWrapper.step needs a closure")

        # ——— Scalar objective for SciPy ——————————————————————
        def scipy_objective(flat_w: np.ndarray) -> float:
            Utils.assign_flat_to_params(flat_w, self._model_parameter_list)
            # Recompute loss from scratch
            return full_batch_loss_closure().item()

        # ——— Gradient callback: one backward, fresh graph ——————————
        def scipy_gradient(flat_w: np.ndarray) -> np.ndarray:
            Utils.assign_flat_to_params(flat_w, self._model_parameter_list)
            loss = full_batch_loss_closure()
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=self._model_parameter_list,
                create_graph=False  # no higher derivatives needed here
            )
            flat_grad = torch.cat([g.view(-1) for g in grads])
            return flat_grad.detach().cpu().numpy()

        # ——— Hessian-vector product callback: own graph ————————————
        def scipy_hessian_vector_product(
            flat_w: np.ndarray,
            v_np: np.ndarray
        ) -> np.ndarray:
            Utils.assign_flat_to_params(flat_w, self._model_parameter_list)
            loss = full_batch_loss_closure()

            # First derivative with create_graph so we can differentiate again
            first_grads = torch.autograd.grad(
                outputs=loss,
                inputs=self._model_parameter_list,
                create_graph=True
            )
            flat_first_grad = torch.cat([g.contiguous().view(-1)
                                         for g in first_grads])

            # Convert direction to tensor on same device
            v = torch.from_numpy(v_np).to(flat_first_grad)

            # Second derivative: directional derivative of the gradient
            hvp = torch.autograd.grad(
                outputs=torch.dot(flat_first_grad, v),
                inputs=self._model_parameter_list,
                create_graph=False  # no further derivatives needed
            )
            flat_hvp = torch.cat([h.contiguous().view(-1) for h in hvp])
            return flat_hvp.detach().cpu().numpy()

        # Initial weights flattened
        x0 = Utils.flatten_params(
            self._model_parameter_list).astype(np.float64)

        # Call SciPy’s trust-ncg solver
        result = scipy_optimize.minimize(
            fun=scipy_objective,
            x0=x0,
            jac=scipy_gradient,
            hessp=scipy_hessian_vector_product,
            method="trust-ncg",
            options={"gtol": 1e-6, "maxiter": 50, "disp": False},
        )

        # Write the final SciPy solution back into the model
        Utils.assign_flat_to_params(result.x, self._model_parameter_list)
        print(
            f"[trust-ncg] iterations: {result.nit} | status: {result.message}")
