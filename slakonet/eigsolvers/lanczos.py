import torch
from torch import Tensor

from slakonet.eigsolvers.base import _EigSolver

_REORTH_TYPES = {"normal_gram_schmidt", "modified_gram_schmidt", "selective"}


class LanczosSolver(_EigSolver):
    """Partial eigensolver using the Lanczos algorithm.

    Finds the m lowest eigenvalues of H|ψ⟩ = E·S|ψ⟩ via a Krylov subspace
    of dimension m. The generalized problem is first reduced to a standard
    symmetric form using the same Cholesky preamble as CholeskyEighSolver.

    All operations use GPU-compatible PyTorch primitives only.

    Gradient flow:
        The seed vector v_0 is detached before the Krylov loop — it is an
        arbitrary random direction with no physical meaning. Gradients flow
        from the Ritz eigenvalues through eigh(T) → tridiagonal entries
        (alpha, beta) → matrix-vector products with H_tilde → H and S.
    """

    def __init__(self, cfg, device=None, dtype=None):
        device = device or torch.device("cpu")
        dtype = dtype or torch.float64
        super().__init__(device, dtype)
        self.m = cfg.m
        self.tol = cfg.tol
        self.max_iter = cfg.max_iter
        self.reorthogonalization = cfg.reorthogonalization
        self.reorthogonalization_type = cfg.reorthogonalization_type
        self.eps = cfg.eps

        if self.reorthogonalization_type not in _REORTH_TYPES:
            raise ValueError(
                f"reorthogonalization_type must be one of {_REORTH_TYPES}, "
                f"got {self.reorthogonalization_type!r}"
            )

    def solve(self, H: Tensor, S: Tensor, **kwargs) -> tuple:
        device = H.device
        dtype = H.dtype
        n = H.shape[-1]
        batch_shape = H.shape[:-2]
        m = min(self.m, n)

        eye = torch.eye(n, device=device, dtype=dtype)
        S_reg = S + self.eps * eye

        L = torch.linalg.cholesky(S_reg)
        L_inv = torch.linalg.inv(L)
        H_tilde = L_inv @ H @ L_inv.mH

        alphas, betas, V = self._krylov(H_tilde, m, device, dtype, batch_shape, n)

        T = self._build_tridiagonal(alphas, betas, m, device, dtype, batch_shape)
        eigenvalues_T, Z = torch.linalg.eigh(T)

        eigenvecs_full = V @ Z
        eigenvecs = L_inv.mH @ eigenvecs_full

        return eigenvalues_T, eigenvecs

    def _krylov(self, A, m, device, dtype, batch_shape, n):
        """Build the Lanczos Krylov basis and collect tridiagonal entries."""
        v = torch.randn(*batch_shape, n, device=device, dtype=dtype)
        v = v.detach()
        v = v / torch.linalg.norm(v, dim=-1, keepdim=True)

        alphas = []
        betas = []
        V = []
        v_prev = None
        beta_prev = None

        for j in range(m):
            V.append(v)

            w = (A @ v.unsqueeze(-1)).squeeze(-1)

            alpha = (v * w).sum(dim=-1)
            alphas.append(alpha)

            w = w - alpha.unsqueeze(-1) * v
            if j > 0:
                w = w - beta_prev.unsqueeze(-1) * v_prev

            if self.reorthogonalization and j > 0:
                w = self._reorthogonalize(w, V)

            beta = torch.linalg.norm(w, dim=-1)
            betas.append(beta)

            v_prev = v
            beta_prev = beta
            v = w / beta.unsqueeze(-1).clamp(min=1e-30)

        return alphas, betas, torch.stack(V, dim=-1)

    def _reorthogonalize(self, w: Tensor, V: list) -> Tensor:
        if self.reorthogonalization_type == "normal_gram_schmidt":
            V_mat = torch.stack(V, dim=-1)
            coeffs = (V_mat.mH @ w.unsqueeze(-1)).squeeze(-1)
            w = w - (V_mat @ coeffs.unsqueeze(-1)).squeeze(-1)

        elif self.reorthogonalization_type == "modified_gram_schmidt":
            for vk in V:
                coeff = (vk * w).sum(dim=-1)
                w = w - coeff.unsqueeze(-1) * vk

        elif self.reorthogonalization_type == "selective":
            # Reorthogonalize only against vectors with small residuals, i.e.
            # those whose contribution to w is above a numerical noise floor.
            eps_mach = torch.finfo(w.dtype).eps
            thresh = eps_mach ** 0.5
            for vk in V:
                coeff = (vk * w).sum(dim=-1)
                mask = coeff.abs() > thresh
                if mask.any():
                    w = w - (coeff * mask.to(coeff.dtype)).unsqueeze(-1) * vk

        return w

    @staticmethod
    def _build_tridiagonal(alphas, betas, m, device, dtype, batch_shape):
        """Assemble the m×m symmetric tridiagonal matrix T."""
        T = torch.zeros(*batch_shape, m, m, device=device, dtype=dtype)
        alpha_stack = torch.stack(alphas, dim=-1)
        diag_idx = torch.arange(m, device=device)
        T[..., diag_idx, diag_idx] = alpha_stack

        beta_stack = torch.stack(betas[:-1], dim=-1) if m > 1 else None
        if beta_stack is not None:
            off_idx = torch.arange(m - 1, device=device)
            T[..., off_idx, off_idx + 1] = beta_stack
            T[..., off_idx + 1, off_idx] = beta_stack

        return T
