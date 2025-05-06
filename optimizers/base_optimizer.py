from abc import ABC, abstractmethod
from typing import Callable


class BaseOptimizer(ABC):
    """ This is a common optimizer interface. Every optimiser wrapper inherits from this."""

    @abstractmethod
    def zero_grad(self) -> None: ...

    @abstractmethod
    def step(self, loss_closure: Callable[[], float] | None = None) -> None:
        """
        *For gradient-based methods* the closure returns the loss and computes
        gradients.  Methods that do not need a closure may ignore the argument.
        """

    # --- “one-shot” solvers override this ---------------------------------
    def run_full_batch_solve(self, loss_closure: Callable[[], float]) -> None:
        """
        Solvers like SLSQP, IPOPT, CMA-ES perform the entire optimisation in
        one call; default implementation does nothing.
        """
        raise NotImplementedError
