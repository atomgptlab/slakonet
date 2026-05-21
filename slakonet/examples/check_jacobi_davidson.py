"""Validate the Jacobi-Davidson interior eigensolver.

Same tight-degeneracy Si test that defeated Lanczos+spectrum folding
(see ``check_lanczos_folded.py``).  Reference is scipy shift-invert
ARPACK via ``solve_near_gap``.
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
from slakonet.jacobi_davidson import solve_near_gap_jd

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


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
    ev_jd, _ = solve_near_gap_jd(
        Hs, Ss, k=k, sigma=sigma,
        tol=1e-7, max_iter=400, verbose=False,
    )
    t_jd = time.perf_counter() - t0
    ev_jd = ev_jd.detach().cpu().numpy()

    err = float(np.max([np.min(np.abs(ev_ref - e)) for e in ev_jd]))
    print(
        f"nrep={nrep}  n={int(geo.n_atoms)}  Norb={int(basis.n_orbitals)}  "
        f"sigma={sigma:+.4f}\n"
        f"  ref(scipy): {np.array2string(ev_ref, precision=5)}\n"
        f"  JD       : {np.array2string(np.sort(ev_jd), precision=5)}\n"
        f"  max|Δeig| = {err:.2e}    ref {t_ref:.2f}s   "
        f"JD {t_jd:.2f}s"
    )
    return err < 1e-5


if __name__ == "__main__":
    ok = True
    for nrep in (2, 3):
        ok &= run_case(nrep, k=6)
    print("\n" + ("PASS" if ok else "FAIL"))
