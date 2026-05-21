"""Spin-polarized bandstructure of elemental Ni (fcc).

Note: the slakonet universal SKF set gives Ni an s,p-only basis (no d-shell),
so this is a demo of the spin-polarized band API on elemental Ni, not a
quantitative ferromagnetic-Ni calculation. The 3d-driven Stoner moment and
~0.3 eV exchange splitting at Ef that real Ni has cannot be reproduced from
this parameter set - you would need refit Ni SKFs with the 3d shell.
"""
from __future__ import annotations

import numpy as np
import torch
from ase.build import bulk

from slakonet.atoms import Geometry
from slakonet.main import SimpleDftb
from slakonet.optim import default_model
from slakonet.magnetism import compute_spin_polarized_bands, plot_spin_bands

# Impose a large Stoner I on the present Ni-p shell so the demo shows a
# visible up/down splitting. A physical Ni-3d Stoner value (~0.037 Ha) would
# produce zero effect here because l=2 is absent from the basis.
STONER_I = {28: {0: 0.0, 1: 0.15, 2: 0.037}}


def ni_fcc():
    # fcc Ni primitive cell, experimental a = 3.52 A
    return bulk("Ni", crystalstructure="fcc", a=3.52)


def fcc_klines(n_seg: int = 20) -> torch.Tensor:
    pts = {
        "G": [0.0, 0.0, 0.0],
        "X": [0.5, 0.0, 0.5],
        "W": [0.5, 0.25, 0.75],
        "K": [0.375, 0.375, 0.75],
        "L": [0.5, 0.5, 0.5],
    }
    segs = ["L", "G", "X", "W", "K", "G"]
    rows = [[*pts[a], *pts[b], n_seg] for a, b in zip(segs[:-1], segs[1:])]
    return torch.tensor([rows], dtype=torch.float64)


def main():
    atoms = ni_fcc()
    print(f"fcc Ni: {len(atoms)} atom(s), {atoms.get_chemical_symbols()}, "
          f"a = {atoms.cell.lengths()[0]:.3f} A")

    model = default_model()
    geometry = Geometry.from_ase_atoms([atoms])
    calc = SimpleDftb(
        geometry, model,
        klines=fcc_klines(n_seg=20),
        device="cpu",
        with_eigenvectors=True,
        compute_forces=False,
        include_dos_data=False,
        repulsive=False,
    )
    calc.calculate()

    print(f"orbs_per_atom: {calc.basis.orbs_per_atom.tolist()}")

    m0 = np.array([2.0])  # one Ni atom
    print(f"Imposed moment on Ni: {m0.tolist()} mu_B")

    result = compute_spin_polarized_bands(
        calc,
        stoner_I=STONER_I,
        initial_moments=torch.tensor(m0),
        scf=False,
    )

    eu = result["eigenvalues_up"]
    ed = result["eigenvalues_dn"]
    max_split = (eu - ed).abs().max().item()
    print(f"Fermi level         : {result['fermi_eV']:.3f} eV")
    print(f"Max up/down splitting: {max_split:.3f} eV")

    plot_spin_bands(
        result,
        fermi_shift_eV=result["fermi_eV"],
        filename="ni_spin_bands.png",
    )
    print("Wrote ni_spin_bands.png")


if __name__ == "__main__":
    main()
