from abc import ABC
from inspect import getfullargspec
from warnings import warn

import torch
from torch import Tensor


class _EigSolver(ABC):
    """ABC for objects responsible for solving the generalized eigenvalue problem.

    Subclasses solve H|ψ⟩ = E·S|ψ⟩ and return eigenvalues (and optionally
    eigenvectors). The interface mirrors `_SkFeed` for consistency.

    Arguments:
        device: Device on which tensors reside.
        dtype: Floating point dtype used by the solver.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.__device = device
        self.__dtype = dtype

    def __init_subclass__(cls, check_sig: bool = True):
        """Warn if subclasses' `solve` method is missing required arguments."""

        def check(func, required_args):
            sig = getfullargspec(func)
            name = func.__qualname__
            if check_sig:
                missing = ", ".join(required_args - set(sig.args))
                if missing:
                    warn(
                        f'Signature Warning: keyword argument(s) "{missing}"'
                        f' missing from method "{name}"',
                        stacklevel=4,
                    )
            if sig.varkw is None:
                warn(
                    f'Signature Warning: method "{name}" must accept '
                    f"arbitrary keyword arguments, i.e. **kwargs.",
                    stacklevel=4,
                )

        if hasattr(cls, "solve"):
            check(cls.solve, {"H", "S"})

    @property
    def device(self) -> torch.device:
        return self.__device

    @device.setter
    def device(self, value):
        name = self.__class__.__name__
        raise AttributeError(
            f"{name} object's device can only be modified via the '.to' method."
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.__dtype

    def solve(self, H: Tensor, S: Tensor, **kwargs) -> tuple:
        """Solve H|ψ⟩ = E·S|ψ⟩.

        Arguments:
            H: Hamiltonian matrix, shape [..., n, n].
            S: Overlap matrix, shape [..., n, n].

        Returns:
            eigenvalues: Shape [..., n] (or [..., m] for partial solvers).
            eigenvectors: Shape [..., n, n] (or [..., n, m]), or None.
        """
        raise NotImplementedError
