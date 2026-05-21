"""Jacobi-Davidson interior eigensolver for ``H c = E S c``.

This is the eigensolver method the Rodrigues 2015 million-atom-TB paper
identifies as the most robust for the tight-degeneracy regime that
defeats Lanczos+spectrum folding (see ``lanczos.py``). The
implementation here is Generalized Davidson + Olsen correction:

* Build an expanding S-orthonormal search subspace ``V``.
* Each iteration: project H, S onto V, solve a small generalized eigh,
  pick the Ritz pair (theta, u=V y) closest to the target ``sigma``.
* Compute the residual ``r = H u - theta * S u`` in S-norm.
* Solve the **correction equation** approximately for the next
  expansion vector ``t``::

      (I - u u^T S)(H - theta S)(I - S u u^T) t = -r,    u^T S t = 0

  using a diagonal preconditioner ``K = diag(H) - theta diag(S)``;
  Olsen's closed-form keeps it pure sparse-matvec + element-wise ops.
* Re-orthogonalize ``t`` against V (S-IP) and against the converged
  eigenvectors (deflation), append, repeat.
* Restart by keeping the best ``k_restart`` Ritz vectors when V fills.

All operators are torch sparse_coo; the algorithm itself is pure
torch sparse matvec + small dense linear algebra, so it runs on GPU.
Complex Hermitian (periodic) goes through the 2n x 2n real embedding
(planned follow-up).

STATUS - prototype only, NOT yet paper-quality
----------------------------------------------
This file contains the *scaffold* of a Generalized Davidson / Olsen-
corrected J-D solver: subspace expansion, RR projection, restart,
deflation, all the bookkeeping. Empirically it fails to converge on
TB band-edge degenerate clusters with the diagonal preconditioner --
``diag(H) - theta * diag(S)`` is near-singular *exactly* where the
target eigenvectors live (TB band edges sit at on-site energies), so
the Olsen correction direction is uninformative. A robust production
J-D for this regime needs:

  * an **inner GMRES/MINRES** solver for the projected operator
    ``(I - u u^T S)(H - theta S)(I - S u u^T) t = -r``  (~100 LOC),
  * a better preconditioner than diagonal -- e.g. ILU with a
    spectral shift, or an approximate-inverse polynomial, or a shifted
    multilevel preconditioner,
  * **harmonic Ritz extraction** to target interior eigenpairs
    (Olsen 1990, Sleijpen-Van der Vorst 1996),
  * adaptive sigma update once the first eigenpair converges.

That work is exactly what production libraries (PRIMME, SLEPc) have
been refining for ~20 years; reproducing it from scratch in a chat-
sized effort is unrealistic. Treat this file as the *interface and
scaffold* for the eventual GPU eigensolver; for production today,
use ``solve_near_gap`` (scipy shift-invert ARPACK) which scales to
~30-50k atoms on a 16 GB host before sparse LU fill-in dominates.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


# -----------------------------------------------------------------------
# helpers (mirrors of those in lanczos.py to keep this file self-contained)
# -----------------------------------------------------------------------
def _matvec_fn(A: Tensor) -> Callable[[Tensor], Tensor]:
    if A.is_sparse and not A.is_coalesced():
        A = A.coalesce()
    return lambda x: torch.sparse.mm(A, x.unsqueeze(-1)).squeeze(-1)


def _diag_sparse(A: Tensor) -> Tensor:
    A = A.coalesce()
    idx, val = A.indices(), A.values()
    diag = torch.zeros(A.shape[0], dtype=val.dtype, device=val.device)
    mask = idx[0] == idx[1]
    if mask.any():
        diag.scatter_add_(0, idx[0][mask], val[mask])
    return diag


# -----------------------------------------------------------------------
# main entry point
# -----------------------------------------------------------------------
def solve_near_gap_jd(
    H: Tensor,
    S: Tensor,
    k: int,
    sigma: float,
    *,
    tol: float = 1e-7,
    max_iter: int = 400,
    v_max: Optional[int] = None,
    k_restart: Optional[int] = None,
    initial_subspace: int = 4,
    return_vectors: bool = False,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    verbose: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """``k`` eigenpairs of ``H c = E S c`` nearest ``sigma`` (real).

    Args:
        H, S: torch sparse_coo (real, symmetric). ``S`` must be SPD.
        k: number of eigenpairs to return.
        sigma: target energy.
        tol: residual S-norm tolerance for convergence.
        max_iter: outer iteration cap.
        v_max: maximum search subspace size before restart
            (default ``max(4*k, 30)``).
        k_restart: vectors kept after restart (default ``k + 5``).
        initial_subspace: number of random vectors to seed V with.
        return_vectors: also return the eigenvectors.
        device, dtype, verbose: usual knobs.

    Returns:
        (evals, evecs)  -- evecs is None unless requested.
    """
    if H.is_complex() or S.is_complex():
        raise NotImplementedError(
            "Complex-Hermitian path not yet implemented; use the "
            "real 2n x 2n embedding (planned)."
        )
    if device is not None:
        H, S = H.to(device), S.to(device)
    if dtype is not None:
        H, S = H.to(dtype), S.to(dtype)
    device = H.device
    dtype = H.values().dtype if H.is_sparse else H.dtype
    n = H.shape[0]

    if v_max is None:
        v_max = max(4 * k, 30)
    if k_restart is None:
        k_restart = k + 5

    Hmv = _matvec_fn(H)
    Smv = _matvec_fn(S)
    diag_H = _diag_sparse(H) if H.is_sparse else torch.diagonal(H)
    diag_S = _diag_sparse(S) if S.is_sparse else torch.diagonal(S)
    sigma_t = torch.tensor(float(sigma), dtype=dtype, device=device)

    # --- containers ----------------------------------------------------
    V = torch.zeros(n, v_max, dtype=dtype, device=device)
    HV = torch.zeros(n, v_max, dtype=dtype, device=device)
    SV = torch.zeros(n, v_max, dtype=dtype, device=device)
    m = 0  # current subspace dimension

    converged_evals = []
    converged_evecs = []  # list of (n,) S-normalised vectors

    # --- helpers -------------------------------------------------------
    def s_orthonormalise(t: Tensor) -> Tensor:
        """S-orthogonalise t against converged set and current V; then
        S-normalise. Returns None if t collapses."""
        # against converged eigenvectors
        for c in converged_evecs:
            t = t - c * float(torch.dot(c, Smv(t)))
        # against current V (twice for numerical safety)
        for _ in range(2):
            if m > 0:
                proj = V[:, :m].t() @ Smv(t)
                t = t - V[:, :m] @ proj
        s_norm = torch.sqrt(torch.dot(t, Smv(t)).clamp_min(0.0))
        if float(s_norm) < 1e-12:
            return None
        return t / s_norm

    def append(t: Tensor):
        nonlocal m
        V[:, m] = t
        HV[:, m] = Hmv(t)
        SV[:, m] = Smv(t)
        m += 1

    # --- seed --------------------------------------------------------
    torch.manual_seed(0)
    for _ in range(initial_subspace):
        t = torch.randn(n, dtype=dtype, device=device)
        t = s_orthonormalise(t)
        if t is None:
            continue
        append(t)
    if m == 0:
        raise RuntimeError("Could not seed the initial subspace")

    # --- main loop ---------------------------------------------------
    n_done = 0
    for it in range(max_iter):
        # projected eigenproblem
        H_sub = V[:, :m].t() @ HV[:, :m]
        S_sub = V[:, :m].t() @ SV[:, :m]
        H_sub = 0.5 * (H_sub + H_sub.t())
        S_sub = 0.5 * (S_sub + S_sub.t())

        try:
            L = torch.linalg.cholesky(S_sub)
        except RuntimeError:
            # near-singular projected S: shrink subspace and continue
            # (rare; just restart)
            if verbose:
                print(f"  [it {it}] S_sub indefinite -> restart")
            m = min(k_restart, m)
            continue
        A_std = torch.linalg.solve_triangular(
            L, torch.linalg.solve_triangular(L, H_sub, upper=False).t(),
            upper=False,
        ).t()
        A_std = 0.5 * (A_std + A_std.t())
        theta_all, Y_std = torch.linalg.eigh(A_std)
        Y = torch.linalg.solve_triangular(L.t(), Y_std, upper=True)

        # pick the Ritz pair closest to sigma that hasn't converged yet.
        # Order by |theta - sigma| ascending; skip ones already accepted.
        order = torch.argsort((theta_all - sigma_t).abs())
        target_idx = int(order[n_done].item())   # next-best Ritz pair
        theta = theta_all[target_idx]
        y = Y[:, target_idx]
        u = V[:, :m] @ y
        Hu = HV[:, :m] @ y
        Su = SV[:, :m] @ y
        r = Hu - theta * Su

        res_norm = torch.sqrt(torch.dot(r, Smv(r)).clamp_min(0.0))
        if verbose:
            print(
                f"  [it {it:3d}] m={m:3d}  done={n_done}  "
                f"θ={float(theta):+.6f}  ||r||_S={float(res_norm):.2e}"
            )

        if float(res_norm) < tol:
            converged_evals.append(theta.detach().clone())
            converged_evecs.append(u.detach().clone())
            n_done += 1
            if n_done >= k:
                break
            # do NOT add to V; just move on to next-best Ritz pair next
            # iteration (target_idx advances via n_done index above).
            continue

        # --- correction equation (Olsen, regularised diag precond) ---
        # For INTERIOR eigenvalues diag(H) - θ diag(S) is near-singular
        # exactly where the eigenvectors live (TB band-edges sit at the
        # on-site energies). Regularise by clamping each entry away from
        # zero by a residual-scaled floor; this keeps the preconditioner
        # bounded and re-introduces a meaningful correction direction.
        K_diag = diag_H - theta * diag_S
        floor = max(float(res_norm) * 0.1, 1e-4)
        K_diag = torch.where(
            K_diag.abs() < floor,
            torch.full_like(K_diag, floor)
            * torch.sign(K_diag + 1e-30),
            K_diag,
        )
        Kinv_r = r / K_diag
        Kinv_Su = Su / K_diag
        num = torch.dot(u, Smv(Kinv_r))
        den = torch.dot(u, Smv(Kinv_Su))
        eps = num / den.clamp_min(1e-30) if float(
            den.abs()
        ) > 1e-30 else torch.zeros((), dtype=dtype, device=device)
        t = -Kinv_r + eps * Kinv_Su

        t = s_orthonormalise(t)
        if t is None:
            # collapsed direction; perturb with random and retry
            t = torch.randn(n, dtype=dtype, device=device)
            t = s_orthonormalise(t)
            if t is None:
                if verbose:
                    print("  ** could not extend subspace; stopping")
                break

        # restart if V is full
        if m >= v_max:
            # thick restart: keep the k_restart Ritz vectors nearest sigma
            order_r = torch.argsort((theta_all - sigma_t).abs())
            keep = order_r[: max(k_restart, n_done + 1)]
            Y_keep = Y[:, keep]
            Vnew = V[:, :m] @ Y_keep
            HVnew = HV[:, :m] @ Y_keep
            SVnew = SV[:, :m] @ Y_keep
            mk = Y_keep.shape[1]
            V[:, :mk] = Vnew
            HV[:, :mk] = HVnew
            SV[:, :mk] = SVnew
            m = mk

        append(t)

    if n_done < k:
        if verbose:
            print(f"  ** converged only {n_done}/{k} after {max_iter}")
    evals = torch.stack(
        converged_evals[:k] if converged_evals else [
            torch.tensor(float("nan"), dtype=dtype, device=device)
        ]
    )
    order_out = torch.argsort(evals)
    evals = evals[order_out]
    if return_vectors and converged_evecs:
        evecs = torch.stack(converged_evecs[:k], dim=-1)[:, order_out]
        return evals, evecs
    return evals, None
