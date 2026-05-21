"""Scaling comparison: dense vs sparse Slater-Koster.

For each finite Si cluster size, two implementations are timed in
*separate processes* (clean peak RSS, no slakonet feed-state bleed):

  dense  : slaterkoster.hs_matrix  + full generalized eigh (eighb)
  sparse : sparse_sk.hs_matrix_sparse + solve_near_gap (k near-gap
           states via shift-invert Lanczos)

Reports: assemble time, solve time, peak RSS, matrix footprint
(dense Norb^2 vs sparse nnz), and near-gap eigenvalue agreement.

Usage:
  python benchmark_scaling.py                 # run the ladder
  python benchmark_scaling.py <child> <n> <sigma>   # internal
"""

import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

# atoms = 2 * n^3 (diamond primitive has 2). Sparse runs all; dense is
# skipped once Norb gets large (O(Norb^2) mem / O(Norb^3) time).
NREPS = [2, 3, 4, 5]
DENSE_MAX_NORB = 7000          # ~780 atoms
KNEAR = 8


def _peak_rss_mb():
    # ru_maxrss is KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def make_cluster(nrep):
    """Finite (non-periodic) diamond-Si cluster, positions in Angstrom."""
    from ase.build import bulk

    sc = bulk("Si", "diamond", a=5.43) * (nrep, nrep, nrep)
    from ase import Atoms

    return Atoms(  # drop the cell -> finite system
        numbers=sc.get_atomic_numbers(), positions=sc.get_positions()
    )


def child(nrep, sigma_arg):
    import torch

    from slakonet.atoms import Geometry
    from slakonet.basis import Basis
    from slakonet.optim import default_model
    from slakonet.utils import (
        create_feeds,
        generate_shell_dict_upto_Z65,
        eighb,
    )
    from slakonet.slaterkoster import hs_matrix
    from slakonet.sparse_sk import hs_matrix_sparse, solve_near_gap

    torch.set_default_dtype(torch.float64)
    method = os.environ["BENCH_METHOD"]

    model = default_model()
    shell_dict = generate_shell_dict_upto_Z65(model=model)
    skfs = model.get_updated_skfs()
    feed = create_feeds(skfs, shell_dict, "H")
    sfeed = create_feeds(skfs, shell_dict, "S")

    atoms = make_cluster(nrep)
    geometry = Geometry.from_ase_atoms([atoms])
    basis = Basis(geometry.atomic_numbers, shell_dict)
    n_atoms = int(geometry.n_atoms)
    n_orb = int(basis.n_orbitals)

    out = {"n_atoms": n_atoms, "n_orb": n_orb, "method": method}

    if method == "dense":
        t0 = time.perf_counter()
        H = hs_matrix(geometry, basis, feed, cutoff=10.0)
        S = hs_matrix(geometry, basis, sfeed, cutoff=10.0)
        H = (H[0] if H.dim() == 3 else H).to(torch.float64)
        S = (S[0] if S.dim() == 3 else S).to(torch.float64)
        out["assemble_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ev, _ = eighb(H, S, scheme="chol")
        out["solve_s"] = time.perf_counter() - t0

        ev = np.sort(ev.real.detach().numpy().flatten())
        m = len(ev)
        lo, hi = m // 3, 2 * m // 3
        g = lo + int(np.argmax(np.diff(ev[lo : hi + 1])))
        sigma = float(0.5 * (ev[g] + ev[g + 1]))
        out["sigma"] = sigma
        # extra eigenvalues around sigma so the accuracy metric is robust
        # to nearest-k set-boundary ties near degeneracies.
        out["near"] = ev[
            np.argsort(np.abs(ev - sigma))[: 4 * KNEAR]
        ].tolist()
        out["matrix_mb"] = n_orb * n_orb * 8 / 1e6
    else:
        sigma = float(sigma_arg)
        t0 = time.perf_counter()
        Hs = hs_matrix_sparse(geometry, basis, feed, cutoff=10.0)
        Ss = hs_matrix_sparse(geometry, basis, sfeed, cutoff=10.0)
        out["assemble_s"] = time.perf_counter() - t0
        out["nnz"] = int(Hs._nnz())
        out["matrix_mb"] = Hs._nnz() * (8 + 16) / 1e6  # val+2 idx approx

        t0 = time.perf_counter()
        got = solve_near_gap(Hs, Ss, k=KNEAR, sigma=sigma)
        out["solve_s"] = time.perf_counter() - t0
        out["near"] = np.sort(got).tolist()

    out["peak_rss_mb"] = _peak_rss_mb()
    print("RESULT " + json.dumps(out))


def run_child(nrep, method, sigma):
    env = dict(os.environ, BENCH_METHOD=method)
    p = subprocess.run(
        [sys.executable, __file__, "child", str(nrep), str(sigma)],
        capture_output=True, text=True, env=env, timeout=1800,
    )
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    sys.stderr.write(p.stdout[-2000:] + "\n" + p.stderr[-2000:] + "\n")
    return None


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(int(sys.argv[2]), sys.argv[3])
        sys.exit(0)

    hdr = (
        f"{'atoms':>6} {'Norb':>6} | "
        f"{'D asm':>7} {'D eigh':>7} {'D mem':>8} {'D RSS':>8} | "
        f"{'S asm':>7} {'S solve':>8} {'S mem':>7} {'S RSS':>8} | "
        f"{'max|Δeig|':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    sigma = 0.0
    for nrep in NREPS:
        # dense first (also yields sigma); skip if too big
        d = None
        # cheap Norb estimate: 2*nrep^3 atoms * 9 orbs (Si spd)
        est_norb = 2 * nrep ** 3 * 9
        if est_norb <= DENSE_MAX_NORB:
            d = run_child(nrep, "dense", 0.0)
            if d:
                sigma = d["sigma"]
        s = run_child(nrep, "sparse", sigma)
        if s is None:
            print(f"{'?':>6} sparse run failed for nrep={nrep}")
            continue

        na, no = s["n_atoms"], s["n_orb"]
        if d:
            # accuracy = max over sparse eigenvalues of the distance to
            # the nearest dense eigenvalue (robust to which states fall
            # in the nearest-k window when levels are near-degenerate).
            dref = np.array(d["near"])
            sev = np.array(s["near"])
            err = float(
                np.max([np.min(np.abs(dref - e)) for e in sev])
            )
            print(
                f"{na:>6} {no:>6} | "
                f"{d['assemble_s']:>7.2f} {d['solve_s']:>7.2f} "
                f"{d['matrix_mb']:>7.0f}M {d['peak_rss_mb']:>7.0f}M | "
                f"{s['assemble_s']:>7.2f} {s['solve_s']:>8.2f} "
                f"{s['matrix_mb']:>6.0f}M {s['peak_rss_mb']:>7.0f}M | "
                f"{err:>10.2e}"
            )
        else:
            print(
                f"{na:>6} {no:>6} | "
                f"{'  --  ':>7} {'  --  ':>7} {'  -- ':>8} {'  -- ':>8} | "
                f"{s['assemble_s']:>7.2f} {s['solve_s']:>8.2f} "
                f"{s['matrix_mb']:>6.0f}M {s['peak_rss_mb']:>7.0f}M | "
                f"{'(no dense ref)':>10}"
            )
