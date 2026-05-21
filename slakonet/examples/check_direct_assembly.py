"""Validate the direct vectorized SK assembly vs the pairwise reference.

The pairwise path was earlier shown to be bit-exact vs slakonet's dense
hs_matrix; this script uses it as a golden oracle for the new direct
path.
"""

import os
import sys
import time

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk

from slakonet.atoms import Geometry
from slakonet.basis import Basis
from slakonet.optim import default_model
from slakonet.utils import create_feeds, generate_shell_dict_upto_Z65
from slakonet.sparse_sk import hs_matrix_sparse

torch.set_default_dtype(torch.float64)

CASES = {
    "Si4": Atoms(
        "Si4",
        positions=[
            [0.0, 0.0, 0.0], [2.35, 0.0, 0.0],
            [1.17, 2.05, 0.0], [1.17, 0.70, 1.95],
        ],
    ),
    "Si3C2": Atoms(
        "Si3C2",
        positions=[
            [0.0, 0.0, 0.0], [2.3, 0.0, 0.0],
            [1.1, 2.0, 0.0], [1.0, 0.6, 1.7],
            [3.0, 1.2, 0.4],
        ],
    ),
    "bulk_Si_222": bulk("Si", "diamond", a=5.43) * (2, 2, 2),  # 16 atoms PBC
}


def run_case(name):
    atoms = CASES[name]
    # use finite (non-periodic) by stripping cell for clusters; keep
    # cell for the bulk case
    if name.startswith("bulk_"):
        ase_atoms = atoms
    else:
        ase_atoms = Atoms(numbers=atoms.get_atomic_numbers(),
                          positions=atoms.get_positions())

    model = default_model()
    shell_dict = generate_shell_dict_upto_Z65(model=model)
    skfs = model.get_updated_skfs()

    geo = Geometry.from_ase_atoms([ase_atoms])
    basis = Basis(geo.atomic_numbers, shell_dict)
    print(f"=== {name}: n_atoms={int(geo.n_atoms)}  "
          f"n_orb={int(basis.n_orbitals)}  "
          f"periodic={bool(geo.is_periodic)} ===")

    case_ok = True
    for kind in ("H", "S"):
        feed = create_feeds(skfs, shell_dict, kind)
        kpoint = [0.13, 0.27, 0.41] if geo.is_periodic else None

        t1 = time.perf_counter()
        Hp = hs_matrix_sparse(
            geo, basis, feed, cutoff=10.0,
            kpoint=kpoint, assembly="pairwise",
        )
        t_pair = time.perf_counter() - t1

        t2 = time.perf_counter()
        Hd = hs_matrix_sparse(
            geo, basis, feed, cutoff=10.0,
            kpoint=kpoint, assembly="direct",
        )
        t_dir = time.perf_counter() - t2

        Mp = Hp.to_dense()
        Md = Hd.to_dense()
        diff = (Mp - Md).abs().max().item()
        norm = max(Mp.abs().max().item(), 1e-30)
        rel = diff / norm
        speedup = t_pair / max(t_dir, 1e-9)
        ok = rel < 1e-9
        case_ok &= ok
        print(
            f"  [{kind}] max|Δ|={diff:.2e}  rel={rel:.2e}  "
            f"pair={t_pair:.3f}s  direct={t_dir:.3f}s  "
            f"speedup={speedup:.1f}x  "
            f"{'PASS' if ok else 'FAIL'}"
        )
    return case_ok


def main():
    all_pass = True
    for name in CASES:
        rc = run_case(name)
        all_pass &= rc
    print("\n" + ("OVERALL PASS" if all_pass else "OVERALL FAIL"))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
