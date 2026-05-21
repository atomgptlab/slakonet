"""Validate solve_near_gap (sparse shift-invert Lanczos) vs dense.

For each small system: build sparse H/S, pick an interior target energy
`sigma`, ask for the k eigenvalues nearest sigma via ARPACK shift-invert,
and compare against the k dense generalized eigenvalues nearest sigma.
Each case runs in its own process (slakonet dense-path feed state).
"""

import numpy as np
import torch
from ase import Atoms

from slakonet.atoms import Geometry
from slakonet.basis import Basis
from slakonet.optim import default_model
from slakonet.utils import create_feeds, generate_shell_dict_upto_Z65, eighb
from slakonet.slaterkoster import hs_matrix
from slakonet.sparse_sk import hs_matrix_sparse, solve_near_gap

torch.set_default_dtype(torch.float64)
CUTOFF = 10.0
K = 8

CASES = {
    "Si4": Atoms(
        "Si4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.35, 0.0, 0.0],
            [1.17, 2.05, 0.0],
            [1.17, 0.70, 1.95],
        ],
    ),
    "Si3C2": Atoms(
        "Si3C2",
        positions=[
            [0.0, 0.0, 0.0],
            [2.3, 0.0, 0.0],
            [1.1, 2.0, 0.0],
            [1.0, 0.6, 1.7],
            [3.0, 1.2, 0.4],
        ],
    ),
}


def run_case(cname):
    model = default_model()
    shell_dict = generate_shell_dict_upto_Z65(model=model)
    skfs = model.get_updated_skfs()
    geometry = Geometry.from_ase_atoms([CASES[cname]])
    basis = Basis(geometry.atomic_numbers, shell_dict)
    h_feed = create_feeds(skfs, shell_dict, "H")
    s_feed = create_feeds(skfs, shell_dict, "S")

    Hs = hs_matrix_sparse(geometry, basis, h_feed, cutoff=CUTOFF)
    Ss = hs_matrix_sparse(geometry, basis, s_feed, cutoff=CUTOFF)

    Hd = hs_matrix(geometry, basis, h_feed, cutoff=CUTOFF)
    Sd = hs_matrix(geometry, basis, s_feed, cutoff=CUTOFF)
    Hd = (Hd[0] if Hd.dim() == 3 else Hd).to(torch.float64)
    Sd = (Sd[0] if Sd.dim() == 3 else Sd).to(torch.float64)

    # dense generalized reference (all eigenvalues)
    ev_all, _ = eighb(Hd, Sd, scheme="chol")
    ev_all = torch.sort(ev_all.real.flatten())[0].numpy()

    n = ev_all.shape[0]
    # interior target ~ a real "gap": midpoint of the widest gap among the
    # central third of the spectrum (guarantees sigma is not an eigenvalue,
    # mirroring the physical band-gap use case).
    lo, hi = n // 3, 2 * n // 3
    gaps = np.diff(ev_all[lo : hi + 1])
    g = lo + int(np.argmax(gaps))
    sigma = float(0.5 * (ev_all[g] + ev_all[g + 1]))

    # k dense eigenvalues nearest sigma (the reference set)
    ref = ev_all[np.argsort(np.abs(ev_all - sigma))[:K]]
    ref.sort()

    got = solve_near_gap(Hs, Ss, k=K, sigma=sigma)

    err = float(np.abs(got - ref).max())
    ok = err < 1e-6  # ARPACK shift-invert is iterative (sub-uHa here)
    print(
        f"=== {cname}: n_orb={n}  sigma={sigma:+.4f} Ha ===\n"
        f"  dense  nearest-{K}: {np.array2string(ref, precision=5)}\n"
        f"  sparse shift-inv  : {np.array2string(got, precision=5)}\n"
        f"  max|Δ eig| = {err:.2e}  -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


if __name__ == "__main__":
    import subprocess
    import sys

    if len(sys.argv) > 1:
        sys.exit(0 if run_case(sys.argv[1]) else 1)

    all_pass = True
    for c in CASES:
        all_pass &= (
            subprocess.run([sys.executable, __file__, c]).returncode == 0
        )
    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)
