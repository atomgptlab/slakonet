"""Sparse-matvec-only interior eigensolver: Lanczos + spectrum folding.

Solves the generalized Hermitian eigenproblem ``H c = E S c`` for the
``k`` eigenvalues nearest a target energy ``sigma`` using **Lanczos
iteration on the spectrum-folded operator** ``A = (H - sigma S) S^-1
(H - sigma S)``. The eigenvalues of ``A c = mu S c`` are ``mu_i =
(lambda_i - sigma)^2`` (smallest -> nearest sigma in the original
problem). The recipe is the same Rodrigues et al. 2015 use in
J Comput Electron 14:593 to scale to ~350k atoms.

Key design points
-----------------
* **No factorization.** Spectrum folding avoids the sparse LU needed
  by shift-invert ARPACK. Each Lanczos iteration applies two sparse
  matvecs of ``H`` and ``S`` plus one inner conjugate-gradient solve
  of ``S y = b`` (preconditioned by ``diag(S)^-1``). For TB systems
  with a normalized basis ``S`` is strongly diagonally dominant and
  CG converges in ~10-30 iterations -- linear in nnz, GPU-friendly.
* **Generalized Lanczos with B = S inner product.** Vectors are
  ``S``-orthonormalized so that the Rayleigh-Ritz step on the small
  tridiagonal yields true generalized Ritz values.
* **Eigenvalue recovery.** The folded eigenvalue gives only
  ``|lambda - sigma|``; we recover the signed ``lambda`` by computing
  the Rayleigh quotient of each recovered eigenvector against ``H``.

This module is real-symmetric only for now. The complex-Hermitian
(periodic) path can be added via a 2n x 2n real-symmetric embedding.

KNOWN LIMITATION (prototype status)
-----------------------------------
Spectrum-folded Lanczos converges accurately when the target window
contains *well-separated* eigenvalues. In the tight-degeneracy regime
typical of TB band edges (many states within 1e-3 Ha of each other),
this v0 prototype loses precision: squaring the spectrum collapses
clusters near the shift to (lambda - sigma)^2 ~ machine_eps_squared,
and the recovered Ritz vectors mix the degenerate manifold. Observed
residuals vs scipy's shift-invert ARPACK are ~1e-2 Ha on small bulk Si
clusters, vs ~1e-7 Ha from ARPACK. Production-grade interior solvers
for this regime need either Jacobi-Davidson (Rodrigues 2015's
preferred method) or Lanczos with implicit restart (Krylov-Schur),
both of which are larger implementation efforts. Treat this module as
the GPU-ready *kernel* on which those upgrades will plug in.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------
def _sparse_matvec_fn(A: Tensor) -> Callable[[Tensor], Tensor]:
    """torch sparse_coo -> callable matvec ``x -> A @ x``."""
    if A.is_sparse and not A.is_coalesced():
        A = A.coalesce()
    return lambda x: torch.sparse.mm(A, x.unsqueeze(-1)).squeeze(-1)


def _diag_of_sparse(A: Tensor) -> Tensor:
    """Extract the diagonal of a coalesced 2-D torch sparse_coo tensor."""
    A = A.coalesce()
    idx = A.indices()
    val = A.values()
    n = A.shape[0]
    diag = torch.zeros(n, dtype=val.dtype, device=val.device)
    mask = idx[0] == idx[1]
    if mask.any():
        diag.scatter_add_(0, idx[0][mask], val[mask])
    return diag


def _pcg_solve(
    Amv: Callable[[Tensor], Tensor],
    b: Tensor,
    M_inv_diag: Tensor,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> Tensor:
    """Jacobi-preconditioned CG for SPD ``A x = b``. Pure torch."""
    x = torch.zeros_like(b)
    r = b - Amv(x)
    z = M_inv_diag * r
    p = z.clone()
    rz_old = torch.dot(r, z)
    r0 = torch.dot(r, r).clamp_min(1e-30)
    for _ in range(max_iter):
        Ap = Amv(p)
        alpha = rz_old / torch.dot(p, Ap).clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        if torch.dot(r, r) <= tol * tol * r0:
            break
        z = M_inv_diag * r
        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old.clamp_min(1e-30)
        p = z + beta * p
        rz_old = rz_new
    return x


# -----------------------------------------------------------------------
# main entry point
# -----------------------------------------------------------------------
def solve_near_gap_lanczos(
    H: Tensor,
    S: Tensor,
    k: int,
    sigma: float,
    n_lanczos: Optional[int] = None,
    cg_tol: float = 1e-9,
    cg_max_iter: int = 200,
    reortho: str = "full",
    return_vectors: bool = False,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """``k`` eigenpairs of ``H c = E S c`` nearest ``sigma`` (real).

    Args:
        H, S: torch sparse_coo (real, symmetric). ``S`` must be SPD.
        k: number of eigenpairs to return.
        sigma: target energy (same units as the diagonal of ``H``).
        n_lanczos: subspace size; default ``max(2*k + 20, 40)``.
        cg_tol, cg_max_iter: inner CG knobs for the ``S^-1`` applies.
        reortho: ``"full"`` (default) or ``"none"``.
        return_vectors: also return the recovered eigenvectors.
        device, dtype: cast everything to these before iterating (e.g.
            ``device="cuda"``); otherwise inferred from ``H``.

    Returns:
        (evals, evecs) -- ``evecs`` is ``None`` unless requested.
    """
    if H.is_complex() or S.is_complex():
        raise NotImplementedError(
            "Complex-Hermitian path not yet implemented; use the "
            "2n x 2n real embedding (planned)."
        )

    if device is not None:
        H = H.to(device)
        S = S.to(device)
    if dtype is not None:
        H = H.to(dtype)
        S = S.to(dtype)
    device = H.device
    dtype = H.values().dtype if H.is_sparse else H.dtype
    n = H.shape[0]
    if n_lanczos is None:
        n_lanczos = max(2 * k + 20, 40)
    n_lanczos = min(n_lanczos, n)

    # --- pre-compute callables and Jacobi preconditioner ----------
    Hmv = _sparse_matvec_fn(H)
    Smv = _sparse_matvec_fn(S)
    S_diag = _diag_of_sparse(S).clamp_min(1e-30) if S.is_sparse else \
        torch.diagonal(S).clamp_min(1e-30)
    M_inv = 1.0 / S_diag

    sigma_t = torch.tensor(float(sigma), dtype=dtype, device=device)

    def A_fold_mv(x: Tensor) -> Tensor:
        """A_fold = (H - sigma S) S^-1 (H - sigma S) applied to x."""
        u = Hmv(x) - sigma_t * Smv(x)
        y = _pcg_solve(Smv, u, M_inv, tol=cg_tol, max_iter=cg_max_iter)
        return Hmv(y) - sigma_t * Smv(y)

    # --- generalized Lanczos with B=S inner product ---------------
    # Self-adjoint operator (in B-IP) is K = B^{-1} A_fold; each iter
    # needs one extra S-inverse solve. Symmetric Lanczos recurrence
    # (Saad, eq. 6.34, adapted for B-orthogonal vectors):
    #   alpha_j = q_j^T A_fold q_j      (Euclidean IP of q and A_fold q)
    #   r_j     = K q_j - alpha_j q_j - beta_{j-1} q_{j-1}
    #   beta_j  = sqrt(r_j^T S r_j)
    #   q_{j+1} = r_j / beta_j
    def b_norm(v: Tensor) -> Tensor:
        return torch.sqrt(torch.dot(v, Smv(v)).clamp_min(1e-30))

    Q = torch.zeros(n_lanczos, n, dtype=dtype, device=device)
    alpha = torch.zeros(n_lanczos, dtype=dtype, device=device)
    beta = torch.zeros(n_lanczos, dtype=dtype, device=device)

    # initial random vector, B-normalised
    v = torch.randn(n, dtype=dtype, device=device)
    v = v / b_norm(v)
    Q[0] = v
    Aq = A_fold_mv(v)
    Kq = _pcg_solve(Smv, Aq, M_inv, tol=cg_tol, max_iter=cg_max_iter)
    alpha[0] = torch.dot(v, Aq)
    r = Kq - alpha[0] * v

    for j in range(1, n_lanczos):
        if reortho == "full":
            r = r - Q[:j].t() @ (Q[:j] @ Smv(r))
        beta_j = b_norm(r)
        if float(beta_j) < 1e-12:
            r = torch.randn(n, dtype=dtype, device=device)
            r = r - Q[:j].t() @ (Q[:j] @ Smv(r))
            beta_j = b_norm(r)
        beta[j - 1] = beta_j
        v = r / beta_j
        Q[j] = v
        Aq = A_fold_mv(v)
        Kq = _pcg_solve(
            Smv, Aq, M_inv, tol=cg_tol, max_iter=cg_max_iter
        )
        alpha[j] = torch.dot(v, Aq)
        r = Kq - alpha[j] * v - beta[j - 1] * Q[j - 1]
        if reortho == "full":
            r = r - Q[: j + 1].t() @ (Q[: j + 1] @ Smv(r))

    # --- subspace Rayleigh-Ritz on the *original* generalized problem
    # H c = E S c, restricted to span(Q). This is much more accurate
    # than per-vector Rayleigh quotients when the interior contains
    # near-degenerate clusters (true for TB band edges): it diagonalises
    # the whole projected operator and respects the degenerate-subspace
    # structure.
    # Build the (n_lanczos, n_lanczos) projected matrices via batched
    # sparse matvecs.
    HQ = torch.stack(
        [Hmv(Q[i]) for i in range(n_lanczos)], dim=0
    )           # (n_lanczos, n)
    SQ = torch.stack(
        [Smv(Q[i]) for i in range(n_lanczos)], dim=0
    )
    H_sub = Q @ HQ.t()
    S_sub = Q @ SQ.t()
    # symmetrise (numerical roundoff)
    H_sub = 0.5 * (H_sub + H_sub.t())
    S_sub = 0.5 * (S_sub + S_sub.t())
    # generalized eigh via Cholesky on the small S_sub (cheap)
    L = torch.linalg.cholesky(S_sub)
    Linv_H = torch.linalg.solve_triangular(L, H_sub, upper=False)
    A_std = torch.linalg.solve_triangular(
        L, Linv_H.t(), upper=False
    ).t()
    A_std = 0.5 * (A_std + A_std.t())
    lam_all, Y_std = torch.linalg.eigh(A_std)
    # Y back-transformed: c = L^{-T} y_std
    Y_gen = torch.linalg.solve_triangular(
        L.t(), Y_std, upper=True
    )

    # pick the k eigenvalues nearest sigma
    order = torch.argsort((lam_all - sigma_t).abs())[:k]
    lam = lam_all[order]
    sort_idx = torch.argsort(lam)
    lam = lam[sort_idx]
    order = order[sort_idx]
    if return_vectors:
        V = Q.t() @ Y_gen[:, order]
        return lam, V
    return lam, None
