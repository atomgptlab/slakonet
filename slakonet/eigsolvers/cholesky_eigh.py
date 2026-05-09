import torch
from torch import Tensor

from slakonet.eigsolvers.base import _EigSolver


class CholeskyEighSolver(_EigSolver):
    """Generalized eigensolver via Cholesky whitening + torch.linalg.eigh.

    Reduces H|ψ⟩ = E·S|ψ⟩ to a standard symmetric problem by computing
    the Cholesky decomposition of S, transforming H into
    H_tilde = L⁻¹ H L⁻ᵀ, and calling torch.linalg.eigh on H_tilde.
    Falls back to torch.linalg.eig on Cholesky failure.

    This is O(n³) and returns all n eigenvalues/eigenvectors.
    """

    def __init__(self, cfg, device=None, dtype=None):
        device = device or torch.device("cpu")
        dtype = dtype or torch.float64
        super().__init__(device, dtype)
        self.eps = cfg.eps

    def solve(self, H: Tensor, S: Tensor, **kwargs) -> tuple:
        n = H.shape[-1]
        device = H.device
        dtype = H.dtype

        eye = torch.eye(n, device=device, dtype=dtype)
        S_reg = S + self.eps * eye

        try:
            L = torch.linalg.cholesky(S_reg)
            L_inv = torch.linalg.inv(L)
            H_tilde = L_inv @ H @ L_inv.mH
            eigenvals, eigenvecs_tilde = torch.linalg.eigh(H_tilde)
            eigenvecs = L_inv.mH @ eigenvecs_tilde
        except RuntimeError as e:
            print(f"Cholesky failed: {e}, falling back to eig")
            eigenvals, eigenvecs = torch.linalg.eig(
                torch.linalg.solve(S_reg, H)
            )
            eigenvals = eigenvals.real

        return eigenvals, eigenvecs
