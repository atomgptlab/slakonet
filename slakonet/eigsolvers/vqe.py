import torch
from torch import Tensor

from slakonet.eigsolvers.base import _EigSolver


class VqeSolver(_EigSolver):
    """Variational Quantum Eigensolver — stub pending paper implementation."""

    def __init__(self, cfg, device=None, dtype=None):
        device = device or torch.device("cpu")
        dtype = dtype or torch.float64
        super().__init__(device, dtype)
        self.n_layers = cfg.n_layers
        self.n_shots = cfg.n_shots
        self.eps = cfg.eps

    def solve(self, H: Tensor, S: Tensor, **kwargs) -> tuple:
        raise NotImplementedError(
            "VQE solver is not yet implemented; see paper in development."
        )
