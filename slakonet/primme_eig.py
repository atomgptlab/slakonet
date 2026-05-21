"""PRIMME-backed interior eigensolver for ``H c = E S c``.

PRIMME (`pip install primme <https://github.com/primme/primme>`_) is a
production-grade Krylov / Jacobi-Davidson library with ~20 years of
refinement (the same vintage as SLEPc) and robust handling of the
tight-degeneracy regime that defeats the DIY ``lanczos.py`` /
``jacobi_davidson.py`` prototypes in this package.

This module is a thin wrapper around ``primme.eigsh`` that takes
``torch.sparse_coo_tensor`` inputs (matching the rest of slakonet's
sparse pipeline) and returns torch outputs.

* Real-symmetric (finite / Gamma): drop-in replacement for
  ``solve_near_gap`` -- significantly more robust on TB band-edge
  clusters where ARPACK shift-invert struggles with sparse-LU fill-in
  and our hand-rolled Lanczos / JD prototypes fail to converge.
* Complex Hermitian (periodic H(k)): supported natively by PRIMME -- no
  ``2n x 2n`` real embedding needed.

PRIMME ships a CPU build by default; a CUDA build exists (``primme-gpu``)
and can be slotted in here unchanged because the Python API is the
same.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# defer the import so a missing primme doesn't break ``import slakonet``
try:
    import primme as _primme
    _HAS_PRIMME = True
except ImportError:
    _primme = None
    _HAS_PRIMME = False


def _torch_sparse_to_scipy(t: Tensor):
    """torch sparse_coo (real or complex) -> scipy.sparse.csr_matrix."""
    from scipy.sparse import coo_matrix

    t = t.coalesce().cpu()
    idx = t.indices().numpy()
    val = t.values()
    if val.is_complex():
        np_val = val.to(torch.complex128).numpy()
        np_dtype = np.complex128
    else:
        np_val = val.to(torch.float64).numpy()
        np_dtype = np.float64
    return coo_matrix(
        (np_val, (idx[0], idx[1])),
        shape=tuple(t.shape),
        dtype=np_dtype,
    ).tocsr()


def solve_near_gap_primme(
    H: Tensor,
    S: Tensor,
    k: int,
    sigma: float,
    *,
    tol: float = 1e-9,
    max_iter: Optional[int] = None,
    return_vectors: bool = False,
    method: str = "PRIMME_JDQMR_ETol",
    ncv: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """``k`` eigenpairs of ``H c = E S c`` nearest ``sigma`` via PRIMME.

    Args:
        H, S: torch sparse_coo (real symmetric or complex Hermitian).
            ``S`` must be SPD.
        k: number of eigenpairs to return.
        sigma: target energy.
        tol: residual tolerance; default 1e-9.
        max_iter: PRIMME outer iteration cap (passed as ``maxiter``);
            ``None`` lets PRIMME pick.
        return_vectors: also return the eigenvectors.
        method: PRIMME solver method. Defaults to ``PRIMME_JDQMR_ETol``
            (Jacobi-Davidson with adaptive QMR inner solver and energy-
            based convergence -- robust for interior eigenvalues).
            Other useful choices: ``"PRIMME_DEFAULT_MIN_TIME"``,
            ``"PRIMME_GD_Olsen_plusK"``.
        ncv: max search-subspace size (PRIMME ``ncv``); ``None`` lets
            PRIMME pick.

    Returns:
        (evals, evecs) -- evecs is a torch tensor of shape ``(n, k)``
        if requested, else ``None``.
    """
    if not _HAS_PRIMME:
        raise ImportError(
            "primme is not installed.  Run `pip install primme`."
        )

    A = _torch_sparse_to_scipy(H)
    M = _torch_sparse_to_scipy(S)

    # PRIMME with sigma + which='SM' targets the k smallest-magnitude
    # eigenvalues of the shifted system, i.e. those nearest sigma in
    # the original problem -- this is the standard interior-eigenpair
    # call in PRIMME.
    kw = {}
    if max_iter is not None:
        kw["maxiter"] = int(max_iter)
    if ncv is not None:
        kw["ncv"] = int(ncv)
    if method:
        kw["method"] = method

    if return_vectors:
        w, v = _primme.eigsh(
            A, k=k, M=M, sigma=float(sigma), which="SM",
            tol=tol, return_eigenvectors=True, **kw,
        )
        order = np.argsort(w)
        w = w[order]
        v = v[:, order]
        ev = torch.from_numpy(np.asarray(w))
        vc = torch.from_numpy(np.asarray(v))
        return ev, vc

    w = _primme.eigsh(
        A, k=k, M=M, sigma=float(sigma), which="SM",
        tol=tol, return_eigenvectors=False, **kw,
    )
    w = np.sort(np.asarray(w))
    return torch.from_numpy(w), None
