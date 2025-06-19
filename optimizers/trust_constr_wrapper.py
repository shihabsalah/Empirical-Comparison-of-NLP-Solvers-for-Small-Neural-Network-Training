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
        # Track improvement for early stopping
        self._last_loss = float('inf')
        self._converged_count = 0
        self._best_params = None

        # Detect device for better performance hints
        sample_param = next(iter(self._parameters))
        self._device = sample_param.device
        self._is_cuda = self._device.type == 'cuda'

    # -------------- one-shot implementation ------------------------------
    def run_full_batch_solve(self, full_loss_closure: Callable[[], Tensor]) -> None:
        # Pre-warm GPU if available for better performance
        if self._is_cuda:
            torch.cuda.synchronize(self._device)

        # Cache for faster evaluation of already-seen points
        point_cache = {}

        def objective(flat_weights: np.ndarray) -> float:
            # Cache lookup for repeated points
            key = flat_weights.tobytes()
            if key in point_cache:
                return point_cache[key][0]

            Utils.assign_flat_to_params(flat_weights, self._parameters)
            loss_val = full_loss_closure().item()            # scalar float

            # Store in cache
            point_cache[key] = (loss_val, None)  # (loss, gradient)

            # Early stopping check
            if self._last_loss - loss_val < 1e-4 and loss_val < self._last_loss:
                self._converged_count += 1
            else:
                self._converged_count = 0

            if loss_val < self._last_loss:
                self._last_loss = loss_val
                # Save best params
                self._best_params = flat_weights.copy()

            # If no progress for several iterations, signal convergence
            if self._converged_count > 5:
                raise StopIteration("Early convergence detected")

            return loss_val

        def gradient(flat_weights: np.ndarray) -> np.ndarray:
            # Cache lookup
            key = flat_weights.tobytes()
            if key in point_cache and point_cache[key][1] is not None:
                return point_cache[key][1]

            Utils.assign_flat_to_params(flat_weights, self._parameters)

            # Clear gradients
            for p in self._parameters:
                if p.grad is not None:
                    p.grad.zero_()

            # Mixed precision settings based on device
            with autocast(device_type=self._device.type, enabled=self._is_cuda):
                loss_tensor = full_loss_closure()            # tensor for autograd

            # Calculate gradients
            loss_tensor.backward()
            grad_vec = Utils.flatten_params(p.grad for p in self._parameters)

            # Update cache
            if key in point_cache:
                point_cache[key] = (point_cache[key][0], grad_vec)
            else:
                point_cache[key] = (loss_tensor.item(), grad_vec)

            return grad_vec.astype(np.float64)

        # ——— Hessian-vector product callback ————————————————
        def scipy_hessian_vector_product(
            flat_weights: np.ndarray,
            v_np: np.ndarray
        ) -> np.ndarray:
            Utils.assign_flat_to_params(flat_weights, self._parameters)

            # Mixed precision settings
            with autocast(device_type=self._device.type, enabled=self._is_cuda):
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

            # Mixed precision for Hessian-vector product
            with autocast(device_type=self._device.type, enabled=self._is_cuda):
                hvp = torch.autograd.grad(
                    outputs=torch.dot(flat_first, v),
                    inputs=self._parameters,
                    create_graph=False
                )

            flat_hvp = torch.cat([h.contiguous().view(-1) for h in hvp])

            # Ensure we clean up GPU memory
            if self._is_cuda:
                torch.cuda.empty_cache()

            return flat_hvp.detach().cpu().numpy()

        initial_flat = Utils.flatten_params(
            self._parameters).astype(np.float64)

        try:
            # Call SciPy's trust-constr solver with optimized settings
            result = sco.minimize(
                fun=objective,
                x0=initial_flat,
                jac=gradient,
                hessp=scipy_hessian_vector_product,
                method="trust-constr",
                # Adjust parameters for better speed-accuracy tradeoff
                options={
                    "gtol": 1e-2,           # Looser tolerance
                    "xtol": 1e-2,           # Looser x tolerance
                    "verbose": 1,           # Show progress
                    "maxiter": 30,          # Fewer iterations
                    "initial_tr_radius": 1.0  # Larger initial trust region
                },
            )
            Utils.assign_flat_to_params(result.x, self._parameters)
            print("[trust-constr] iterations:",
                  result.niter, " message:", result.message)

        except StopIteration:
            # Early stopping triggered - use best parameters found
            print("[trust-constr] Early stopping triggered")
            if self._best_params is not None:
                Utils.assign_flat_to_params(
                    self._best_params, self._parameters)

        finally:
            # Clean up GPU memory
            if self._is_cuda:
                torch.cuda.empty_cache()

            # Clear cache
            point_cache.clear()

    # epoch-style API not used but must exist
    def zero_grad(self) -> None: ...

    def step(self, loss_closure: Callable[[],
             Tensor] | None = None) -> None: ...
