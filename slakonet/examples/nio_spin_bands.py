"""Spin-polarized bandstructure of NiO - simple demo.

IMPORTANT: the current slakonet universal parameter set (slakonet_v0) only
includes s and p shells for Ni (no d-shell - Ni orbs_per_atom = 4). This
means the Ni d-manifold that drives NiO magnetism is absent from the
Hamiltonian, and a physical Stoner model on l=2 produces zero splitting.

This script therefore does a *demonstration* of the spin-polarized band
API: it imposes fixed atomic moments and applies the Stoner-like exchange
shift on whichever shells are present (s + p for Ni in the universal set).
The resulting plot shows how the up and down channels of the Ni-p bands
separate under an applied exchange field - it is not a quantitatively
correct NiO calculation. For production NiO you would need refit Ni/O
SKFs including the Ni-3d shell.

Change AFM = True for a 1x1x2 supercell with two Ni atoms (+/- moments).
"""
from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk

from slakonet.atoms import Geometry
from slakonet.main import SimpleDftb
from slakonet.optim import default_model
from slakonet.magnetism import compute_spin_polarized_bands, plot_spin_bands

AFM = False
# Exchange strength per shell (Hartree). d-entry is harmless if no d-shell
# is present in the basis; p-entry is what actually drives the splitting
# in the current universal SKF.
STONER_I = {
    28: {0: 0.0, 1: 0.15, 2: 0.05},
    8:  {0: 0.0, 1: 0.00},
}


def nio_structure() -> Atoms:
    unit = bulk("NiO", crystalstructure="rocksalt", a=4.17)
    return unit.repeat((1, 1, 2)) if AFM else unit


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
    atoms = nio_structure()
    print(f"NiO ({'AFM 2-Ni supercell' if AFM else 'FM primitive'}): "
          f"{len(atoms)} atoms = {atoms.get_chemical_symbols()}")

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

    print(f"Shells per Z available: {calc.shell_dict}")
    print(f"orbs_per_atom: {calc.basis.orbs_per_atom.tolist()}")

    Z = atoms.get_atomic_numbers()
    m0 = np.zeros(len(Z))
    ni_idx = [i for i, z in enumerate(Z) if z == 28]
    for k, i in enumerate(ni_idx):
        m0[i] = (-1) ** k * 2.0 if AFM else 2.0
    print(f"Imposed Ni moments: {m0.tolist()}")

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
        filename="nio_spin_bands.png",
    )
    print("Wrote nio_spin_bands.png")


if __name__ == "__main__":
    main()
