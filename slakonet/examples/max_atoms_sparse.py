"""Stress test: how big a finite system can the slakonet sparse pipeline
handle on this machine?

Strategy
--------
* Load the model once.
* Ramp finite-Si cluster size (diamond supercell, periodicity stripped
  so we exercise the validated Γ finite path), timing
  ``hs_matrix_sparse`` (H and S) and ``solve_near_gap`` (k=8 interior
  eigenpairs near the centre of the spectrum) at each size.
* Append a CSV row per successful size to ``max_atoms_sparse_log.csv``
  *as it completes* so partial progress survives interruption.
* Stop when any of these is true:
    - per-size assembly time > MAX_ASM_SEC
    - process available RAM < SAFE_MEM_MB
    - an exception is raised (OOM, etc.)

Run in the background:
    nohup python max_atoms_sparse.py > max_atoms_sparse.out 2>&1 &
"""

import csv
import gc
import os
import resource
import time
import traceback

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk

import psutil

from slakonet.atoms import Geometry
from slakonet.basis import Basis
from slakonet.optim import default_model
from slakonet.utils import create_feeds, generate_shell_dict_upto_Z65
from slakonet.sparse_sk import hs_matrix_sparse, solve_near_gap

# ---- limits -------------------------------------------------------------
NREPS = [4, 5, 6, 7, 8, 10, 12, 14, 16]   # atoms = 2 * nrep**3
MAX_ASM_SEC = 3600.0        # stop if a single sparse assembly exceeds 1h
SAFE_MEM_MB = 1500          # stop if available system RAM drops below this
KNEAR = 8
SIGMA_HA = 0.0              # interior shift (TB on-site centre ~ 0 Ha)
CUTOFF = 10.0
DEVICE = "cpu"
LOG = os.path.join(os.path.dirname(__file__), "max_atoms_sparse_log.csv")

torch.set_default_dtype(torch.float64)


def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _avail_mb():
    return psutil.virtual_memory().available / 1024 / 1024


def make_cluster(nrep):
    """Finite (no PBC) diamond-Si supercell."""
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
                "asm_s", "solve_s", "total_s",
                "nnz_H", "nnz_S",
                "peak_rss_MB", "avail_MB_after",
            ])
        w.writerow(row)


def main():
    print("[*] machine: total RAM %.1f GB, avail %.1f GB, cpus=%d"
          % (psutil.virtual_memory().total / 1e9,
             psutil.virtual_memory().available / 1e9,
             psutil.cpu_count()))
    t0 = time.perf_counter()
    model = default_model()
    shell_dict = generate_shell_dict_upto_Z65(model=model)
    skfs = model.get_updated_skfs()
    h_feed = create_feeds(skfs, shell_dict, "H")
    s_feed = create_feeds(skfs, shell_dict, "S")
    print("[*] model + feeds ready in %.1fs" % (time.perf_counter() - t0))

    last_ok = None
    for nrep in NREPS:
        avail = _avail_mb()
        if avail < SAFE_MEM_MB:
            print(f"[STOP] only {avail:.0f} MB free, below SAFE_MEM_MB="
                  f"{SAFE_MEM_MB}")
            break

        ase_atoms = make_cluster(nrep)
        n_atoms = len(ase_atoms)
        try:
            geo = Geometry.from_ase_atoms([ase_atoms])
            basis = Basis(geo.atomic_numbers, shell_dict)
            n_orb = int(basis.n_orbitals)
            print(
                f"\n=== nrep={nrep}  n_atoms={n_atoms}  n_orb={n_orb}  "
                f"avail={avail:.0f} MB ===",
                flush=True,
            )

            t_asm = time.perf_counter()
            Hs = hs_matrix_sparse(geo, basis, h_feed, cutoff=CUTOFF)
            Ss = hs_matrix_sparse(geo, basis, s_feed, cutoff=CUTOFF)
            asm_s = time.perf_counter() - t_asm
            print(f"  assemble   {asm_s:.2f} s   "
                  f"nnz(H)={Hs._nnz()}  nnz(S)={Ss._nnz()}",
                  flush=True)

            if asm_s > MAX_ASM_SEC:
                print(f"[STOP] assembly {asm_s:.0f}s exceeded "
                      f"MAX_ASM_SEC={MAX_ASM_SEC}s")
                append_row([nrep, n_atoms, n_orb, asm_s, float("nan"),
                            float("nan"), Hs._nnz(), Ss._nnz(),
                            _peak_rss_mb(), _avail_mb()])
                break

            t_slv = time.perf_counter()
            evs = solve_near_gap(Hs, Ss, k=KNEAR, sigma=SIGMA_HA)
            slv_s = time.perf_counter() - t_slv
            print(f"  solve_near {slv_s:.2f} s   "
                  f"e[0]={evs[0]:+.4f}  e[-1]={evs[-1]:+.4f} Ha",
                  flush=True)

            append_row([
                nrep, n_atoms, n_orb,
                asm_s, slv_s, asm_s + slv_s,
                Hs._nnz(), Ss._nnz(),
                _peak_rss_mb(), _avail_mb(),
            ])
            last_ok = (nrep, n_atoms, n_orb, asm_s + slv_s)
            print(f"  peak RSS   {_peak_rss_mb():.0f} MB   "
                  f"avail now  {_avail_mb():.0f} MB", flush=True)

            del Hs, Ss, geo, basis
            gc.collect()

        except Exception as e:
            print(f"[STOP] exception at nrep={nrep}: "
                  f"{type(e).__name__}: {str(e)[:200]}", flush=True)
            traceback.print_exc()
            break

    print("\n========== RESULT ==========")
    if last_ok is None:
        print("No successful run.")
    else:
        nrep, na, no, tot = last_ok
        print(f"Largest successful size: nrep={nrep}  atoms={na}  "
              f"orbitals={no}  wall={tot:.1f}s")
    print(f"Log: {LOG}")


if __name__ == "__main__":
    main()
