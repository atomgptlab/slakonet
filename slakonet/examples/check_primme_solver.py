"""Validate the PRIMME wrapper on the same tight-degeneracy Si test
that broke the DIY Lanczos+folding and J-D prototypes. Reference is
scipy shift-invert ARPACK via ``solve_near_gap``.
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
from slakonet.primme_eig import solve_near_gap_primme

torch.set_default_dtype(torch.float64)


def finite_cluster(nrep):
    sc = bulk("Si", "diamond", a=5.43) * (nrep, nrep, nrep)
    return Atoms(
        numbers=sc.get_atomic_numbers(), positions=sc.get_positions()
    )


def run_case(nrep, k=6):
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
    sigma = float(torch.diagonal(Hs.to_dense()).median())

    t0 = time.perf_counter()
    ev_ref = np.sort(solve_near_gap(Hs, Ss, k=k, sigma=sigma))
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    ev_p, _ = solve_near_gap_primme(Hs, Ss, k=k, sigma=sigma, tol=1e-9)
    t_p = time.perf_counter() - t0
    ev_p = np.sort(ev_p.numpy())

    err = float(np.max([np.min(np.abs(ev_ref - e)) for e in ev_p]))
    print(
        f"nrep={nrep}  n_atoms={int(geo.n_atoms)}  Norb={int(basis.n_orbitals)}  "
        f"sigma={sigma:+.4f}\n"
        f"  scipy ref : {np.array2string(ev_ref, precision=6)}\n"
        f"  primme    : {np.array2string(ev_p, precision=6)}\n"
        f"  max|Δeig| = {err:.2e}    scipy {t_ref:.3f}s  primme {t_p:.3f}s"
    )
    return err < 1e-6


if __name__ == "__main__":
    ok = True
    for nrep in (2, 3, 4):       # 16, 54, 128 atoms
        ok &= run_case(nrep, k=6)
    print("\n" + ("PASS" if ok else "FAIL"))
