"""Validate the spectrum-folded Lanczos interior eigensolver.

Build sparse H, S for a small finite Si cluster, get reference
near-sigma eigenvalues via the existing ``solve_near_gap`` (scipy
shift-invert ARPACK -- the proven CPU baseline) and compare to the
new ``solve_near_gap_lanczos`` (pure-torch, sparse-matvec-only, no
factorization -- the GPU-ready path).
"""

import time

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk

from slakonet.atoms import Geometry
from slakonet.basis import Basis
from slakonet.optim import default_model
from slakonet.utils import create_feeds, generate_shell_dict_upto_Z65
from slakonet.sparse_sk import hs_matrix_sparse, solve_near_gap
from slakonet.lanczos import solve_near_gap_lanczos

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def finite_cluster(nrep):
    """Diamond-Si supercell as a finite (no-PBC) cluster."""
    sc = bulk("Si", "diamond", a=5.43) * (nrep, nrep, nrep)
    return Atoms(
        numbers=sc.get_atomic_numbers(), positions=sc.get_positions()
    )


def run_case(nrep, k=8, sigma_override=None):
    atoms = finite_cluster(nrep)
    model = default_model()
    shell_dict = generate_shell_dict_upto_Z65(model=model)
    skfs = model.get_updated_skfs()
    h_feed = create_feeds(skfs, shell_dict, "H")
    s_feed = create_feeds(skfs, shell_dict, "S")

    geo = Geometry.from_ase_atoms([atoms])
    basis = Basis(geo.atomic_numbers, shell_dict)
    Hs = hs_matrix_sparse(geo, basis, h_feed, cutoff=10.0)
    Ss = hs_matrix_sparse(geo, basis, s_feed, cutoff=10.0)
    n_orb = int(basis.n_orbitals)

    # cheap sigma estimate: the median of the diagonal of H (s-orbital
    # on-sites cluster); good enough as a target near the gap region.
    sigma = sigma_override if sigma_override is not None else \
        float(torch.diagonal(Hs.to_dense()).median())

    # reference
    t0 = time.perf_counter()
    ev_ref = solve_near_gap(Hs, Ss, k=k, sigma=sigma)
    t_ref = time.perf_counter() - t0

    # candidate
    t0 = time.perf_counter()
    ev_new, _ = solve_near_gap_lanczos(
        Hs, Ss, k=k, sigma=sigma,
        n_lanczos=max(4 * k, 60),
        reortho="full",
    )
    t_new = time.perf_counter() - t0
    ev_new = ev_new.detach().cpu().numpy()

    # match each new eigenvalue to its nearest reference eigenvalue
    err = np.max([np.min(np.abs(ev_ref - e)) for e in ev_new])
    print(
        f"nrep={nrep}  n_atoms={int(geo.n_atoms)}  n_orb={n_orb}  "
        f"sigma={sigma:+.4f}\n"
        f"  scipy ref : {np.array2string(np.sort(ev_ref), precision=5)}\n"
        f"  lanczos   : {np.array2string(np.sort(ev_new), precision=5)}\n"
        f"  max|Δ eig| = {err:.2e}     ref {t_ref:.2f}s  "
        f"lanczos {t_new:.2f}s"
    )
    return err < 1e-5


if __name__ == "__main__":
    ok = True
    for nrep in (2, 3):     # 16 atoms, 54 atoms
        ok &= run_case(nrep, k=6)
    print("\n" + ("PASS" if ok else "FAIL"))
