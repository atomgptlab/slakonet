"""scipy ARPACK shift-invert vs PRIMME J-D: where does the crossover sit?

For a ladder of finite Si clusters, compute the same ``k=8`` near-gap
eigenvalues with two solvers and compare. The expected pattern:

  small N : scipy ARPACK + sparse LU wins  (LU is cheap when small)
  large N : PRIMME wins  (sparse LU fill-in explodes; PRIMME stays
            iterative and memory-bounded)

Assembly is already shared (vectorized direct path), so this isolates
the eigensolver. Logs incrementally to a CSV so partial progress
survives interruption.
"""

import csv
import os
import resource
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

NREPS = [4, 5, 6, 7, 8, 10, 12]   # atoms = 2 * n^3 -> 128..3456
KNEAR = 8
SCIPY_BUDGET_S = 1800
PRIMME_BUDGET_S = 1800
LOG = os.path.join(os.path.dirname(__file__),
                   "bench_solvers_crossover.csv")


def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def make_cluster(nrep):
    sc = bulk("Si", "diamond", a=5.43) * (nrep, nrep, nrep)
    return Atoms(
        numbers=sc.get_atomic_numbers(), positions=sc.get_positions()
    )


def append_row(row):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow([
                "nrep", "n_atoms", "n_orb",
                "asm_s",
                "scipy_solve_s", "primme_solve_s",
                "scipy_status", "primme_status",
                "max_abs_diff", "peak_rss_MB",
            ])
        w.writerow(row)


def main():
    t0 = time.perf_counter()
    model = default_model()
    shell_dict = generate_shell_dict_upto_Z65(model=model)
    skfs = model.get_updated_skfs()
    h_feed = create_feeds(skfs, shell_dict, "H")
    s_feed = create_feeds(skfs, shell_dict, "S")
    print(f"[*] model + feeds ready in {time.perf_counter() - t0:.1f}s")

    scipy_alive = True
    primme_alive = True

    for nrep in NREPS:
        ase_atoms = make_cluster(nrep)
        n_atoms = len(ase_atoms)
        geo = Geometry.from_ase_atoms([ase_atoms])
        basis = Basis(geo.atomic_numbers, shell_dict)
        n_orb = int(basis.n_orbitals)
        print(f"\n=== nrep={nrep}  n_atoms={n_atoms}  n_orb={n_orb} ===",
              flush=True)

        t_asm = time.perf_counter()
        Hs = hs_matrix_sparse(geo, basis, h_feed, cutoff=10.0)
        Ss = hs_matrix_sparse(geo, basis, s_feed, cutoff=10.0)
        asm_s = time.perf_counter() - t_asm
        sigma = float(torch.diagonal(Hs.to_dense()).median())
        print(f"  assemble {asm_s:.2f}s   sigma={sigma:+.4f}",
              flush=True)

        ev_scipy = None
        scipy_s = float("nan")
        scipy_status = "skipped"
        if scipy_alive:
            try:
                t0 = time.perf_counter()
                ev_scipy = np.sort(
                    solve_near_gap(Hs, Ss, k=KNEAR, sigma=sigma)
                )
                scipy_s = time.perf_counter() - t0
                scipy_status = "ok"
                print(f"  scipy   {scipy_s:.2f}s", flush=True)
                if scipy_s > SCIPY_BUDGET_S:
                    scipy_alive = False
                    print("  [scipy disabled: over budget]")
            except Exception as e:
                scipy_status = f"fail:{type(e).__name__}"
                scipy_alive = False
                print(f"  scipy FAILED: {e}", flush=True)

        ev_primme = None
        primme_s = float("nan")
        primme_status = "skipped"
        if primme_alive:
            try:
                t0 = time.perf_counter()
                ev_primme_t, _ = solve_near_gap_primme(
                    Hs, Ss, k=KNEAR, sigma=sigma, tol=1e-9,
                )
                ev_primme = np.sort(ev_primme_t.numpy())
                primme_s = time.perf_counter() - t0
                primme_status = "ok"
                print(f"  primme  {primme_s:.2f}s", flush=True)
                if primme_s > PRIMME_BUDGET_S:
                    primme_alive = False
                    print("  [primme disabled: over budget]")
            except Exception as e:
                primme_status = f"fail:{type(e).__name__}"
                primme_alive = False
                print(f"  primme FAILED: {e}", flush=True)

        diff = float("nan")
        if ev_scipy is not None and ev_primme is not None:
            diff = float(np.max(np.abs(ev_scipy - ev_primme)))
            print(f"  max|Δeig| = {diff:.2e}", flush=True)

        append_row([
            nrep, n_atoms, n_orb, asm_s,
            scipy_s, primme_s, scipy_status, primme_status,
            diff, _peak_rss_mb(),
        ])

        if not scipy_alive and not primme_alive:
            print("[STOP] both solvers exhausted")
            break

    print(f"\nLog: {LOG}")


if __name__ == "__main__":
    main()
