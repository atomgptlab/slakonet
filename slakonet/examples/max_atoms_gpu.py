"""Max-atom stress test of the sparse SlaKoNet pipeline on a GPU.

How it works
------------
* Auto-selects the least-busy CUDA device (override with --gpu).
* Loads the model on CPU once, then runs sparse ``hs_matrix_sparse``
  assembly on the GPU for each cluster size.
* The interior eigensolver still runs on CPU (scipy ARPACK shift-
  invert via ``solve_near_gap``) -- there is no validated GPU interior
  solver in this stack yet. H and S are moved to CPU for the solve.
* Ramps finite Si clusters (no PBC), logs an incremental CSV, and
  stops on OOM / over-budget / failure.

Run
---
    python max_atoms_gpu.py                       # auto-pick GPU
    python max_atoms_gpu.py --gpu 2               # force GPU 2
    python max_atoms_gpu.py --device cpu          # CPU only
    python max_atoms_gpu.py --no-solve            # only time assembly
    python max_atoms_gpu.py --max-nrep 18         # push the ladder

Output: ``max_atoms_gpu_log.csv`` and ``max_atoms_gpu.out``.
"""

import argparse
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

try:
    import psutil
except ImportError:
    psutil = None

from slakonet.atoms import Geometry
from slakonet.basis import Basis
from slakonet.optim import default_model
from slakonet.utils import create_feeds, generate_shell_dict_upto_Z65
from slakonet.sparse_sk import hs_matrix_sparse, solve_near_gap


# ---- defaults ----------------------------------------------------------
DEFAULT_NREPS = [4, 5, 6, 7, 8, 10, 12, 14, 16, 18]   # 2 * nrep^3 atoms
KNEAR = 8
MAX_ASM_SEC = 1800.0       # cap per-size assembly wall (30 min)
MAX_SOLVE_SEC = 3600.0     # cap per-size solve wall (1 h)
SAFE_HOST_MB = 2000        # min free host RAM before stopping
SAFE_GPU_MB = 1500         # min free GPU RAM before stopping
CUTOFF = 10.0


def _select_least_busy_gpu():
    """Return the CUDA device index with the most free memory."""
    if not torch.cuda.is_available():
        return None
    best_idx, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best_idx = i
    return best_idx


def _free_gpu_mb(device_idx):
    if device_idx is None:
        return None
    free, _ = torch.cuda.mem_get_info(device_idx)
    return free / 1024 / 1024


def _peak_rss_mb():
    # ru_maxrss is KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _free_host_mb():
    if psutil is None:
        return None
    return psutil.virtual_memory().available / 1024 / 1024


def make_cluster(nrep):
    sc = bulk("Si", "diamond", a=5.43) * (nrep, nrep, nrep)
    return Atoms(
        numbers=sc.get_atomic_numbers(), positions=sc.get_positions()
    )


def append_row(log_path, row):
    new = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow([
                "nrep", "n_atoms", "n_orb",
                "device", "asm_s", "solve_s",
                "nnz_H", "nnz_S",
                "peak_host_rss_MB", "free_host_MB_after",
                "free_gpu_MB_after",
            ])
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None,
                    help="explicit device; default auto-picks cuda if "
                    "available")
    ap.add_argument("--gpu", type=int, default=None,
                    help="CUDA device index (default: least-busy)")
    ap.add_argument("--nreps", type=int, nargs="+", default=None,
                    help="explicit nrep ladder (atoms = 2 * nrep^3)")
    ap.add_argument("--max-nrep", type=int, default=None,
                    help="cap on nrep when using default ladder")
    ap.add_argument("--no-solve", action="store_true",
                    help="skip the (CPU) eigensolve; only time assembly")
    ap.add_argument("--knear", type=int, default=KNEAR,
                    help="number of near-gap eigenpairs (default 8)")
    ap.add_argument("--cutoff", type=float, default=CUTOFF)
    ap.add_argument("--out-dir", default=os.path.dirname(__file__))
    args = ap.parse_args()

    log_path = os.path.join(args.out_dir, "max_atoms_gpu_log.csv")

    # ---- device selection ----------------------------------------
    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
        gpu_idx = None
        print("[*] running on CPU")
    else:
        gpu_idx = args.gpu if args.gpu is not None else _select_least_busy_gpu()
        device = torch.device(f"cuda:{gpu_idx}")
        name = torch.cuda.get_device_name(gpu_idx)
        free, total = torch.cuda.mem_get_info(gpu_idx)
        print(
            f"[*] GPU {gpu_idx}: {name}  "
            f"free {free/1e9:.1f} GB / total {total/1e9:.1f} GB"
        )

    # ---- size ladder --------------------------------------------
    if args.nreps:
        nreps = list(args.nreps)
    else:
        nreps = list(DEFAULT_NREPS)
        if args.max_nrep is not None:
            nreps = [n for n in nreps if n <= args.max_nrep]

    torch.set_default_dtype(torch.float64)

    # ---- model + feeds (load once on CPU) -----------------------
    t0 = time.perf_counter()
    model = default_model()
    shell_dict = generate_shell_dict_upto_Z65(model=model)
    skfs = model.get_updated_skfs()
    h_feed = create_feeds(skfs, shell_dict, "H")
    s_feed = create_feeds(skfs, shell_dict, "S")
    print(f"[*] model + feeds ready in {time.perf_counter() - t0:.1f}s")

    last_ok = None

    for nrep in nreps:
        ase_atoms = make_cluster(nrep)
        n_atoms = len(ase_atoms)
        try:
            geo = Geometry.from_ase_atoms([ase_atoms])
            # move atomic positions / cell to the chosen device so the
            # SK assembly happens there (slakonet handles cpu transparently).
            try:
                geo.positions = geo.positions.to(device)
                if geo.cell is not None:
                    geo.cell = geo.cell.to(device)
            except Exception:
                pass
            basis = Basis(geo.atomic_numbers, shell_dict)
            n_orb = int(basis.n_orbitals)

            free_gpu = _free_gpu_mb(gpu_idx)
            free_host = _free_host_mb()
            host_str = f"{free_host:.0f}" if free_host else "?"
            gpu_str = f"  gpu free {free_gpu:.0f}MB" if free_gpu else ""
            print(
                f"\n=== nrep={nrep}  n_atoms={n_atoms}  n_orb={n_orb}  "
                f"host free {host_str}MB{gpu_str} ===",
                flush=True,
            )

            if free_host is not None and free_host < SAFE_HOST_MB:
                print(f"[STOP] host RAM low ({free_host:.0f} MB)")
                break
            if free_gpu is not None and free_gpu < SAFE_GPU_MB:
                print(f"[STOP] GPU RAM low ({free_gpu:.0f} MB)")
                break

            # ---- assembly (chosen device) ----------------------
            t_asm = time.perf_counter()
            Hs = hs_matrix_sparse(geo, basis, h_feed, cutoff=args.cutoff)
            Ss = hs_matrix_sparse(geo, basis, s_feed, cutoff=args.cutoff)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            asm_s = time.perf_counter() - t_asm
            print(
                f"  assemble {asm_s:.2f}s   "
                f"nnz(H)={Hs._nnz()} nnz(S)={Ss._nnz()}", flush=True,
            )
            if asm_s > MAX_ASM_SEC:
                print(f"[STOP] assembly over budget ({asm_s:.0f}s)")
                append_row(log_path, [
                    nrep, n_atoms, n_orb, str(device),
                    asm_s, float("nan"),
                    Hs._nnz(), Ss._nnz(),
                    _peak_rss_mb(), _free_host_mb(),
                    _free_gpu_mb(gpu_idx),
                ])
                break

            # ---- solve (always CPU; scipy ARPACK) --------------
            solve_s = float("nan")
            if not args.no_solve:
                Hcpu = Hs.cpu()
                Scpu = Ss.cpu()
                sigma = float(torch.diagonal(Hcpu.to_dense()).median())
                t_slv = time.perf_counter()
                evs = solve_near_gap(
                    Hcpu, Scpu, k=args.knear, sigma=sigma,
                )
                solve_s = time.perf_counter() - t_slv
                print(
                    f"  solve(cpu) {solve_s:.2f}s    "
                    f"e[0]={evs[0]:+.4f}  e[-1]={evs[-1]:+.4f}",
                    flush=True,
                )
                if solve_s > MAX_SOLVE_SEC:
                    print(f"[STOP] solve over budget ({solve_s:.0f}s)")
                    append_row(log_path, [
                        nrep, n_atoms, n_orb, str(device),
                        asm_s, solve_s,
                        Hs._nnz(), Ss._nnz(),
                        _peak_rss_mb(), _free_host_mb(),
                        _free_gpu_mb(gpu_idx),
                    ])
                    break

            append_row(log_path, [
                nrep, n_atoms, n_orb, str(device),
                asm_s, solve_s,
                Hs._nnz(), Ss._nnz(),
                _peak_rss_mb(), _free_host_mb(),
                _free_gpu_mb(gpu_idx),
            ])
            last_ok = (nrep, n_atoms, n_orb, asm_s, solve_s)

            del Hs, Ss, geo, basis
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            print(f"[STOP] GPU OOM at nrep={nrep}: {e}", flush=True)
            break
        except MemoryError as e:
            print(f"[STOP] host OOM at nrep={nrep}: {e}", flush=True)
            break
        except Exception as e:
            print(f"[STOP] exception at nrep={nrep}: "
                  f"{type(e).__name__}: {str(e)[:300]}", flush=True)
            traceback.print_exc()
            break

    print("\n========== RESULT ==========")
    if last_ok is None:
        print("No successful run.")
    else:
        nrep, na, no, a, s = last_ok
        total = a + (s if s == s else 0)  # NaN-safe
        print(
            f"Largest successful size: nrep={nrep}  atoms={na}  "
            f"orbitals={no}  assemble={a:.1f}s  solve={s:.1f}s  "
            f"total={total:.1f}s"
        )
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
