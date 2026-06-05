"""Spin-polarized bandstructure of NiO - simple demo.

The current slakonet universal parameter set (slakonet_v1) includes the
Ni 3d shell (Ni shells = [s, p, d], orbs_per_atom = 9). NiO magnetism is
driven by that Ni-3d manifold, so we apply the Stoner-like exchange shift
on l=2 with a physical Stoner parameter (~0.037 Ha for Ni).

This is a Stoner-rigid demonstration of the spin-polarized band API: it
imposes a fixed Ni moment and splits the two spin channels by a diagonal
on-site exchange field. It is not a full spin-polarized SCC-DFTB result,
but with the exchange on the d-shell it now produces a physically sensible
d-band splitting rather than the artificially large p-shell split used in
the older s/p-only parameter set.

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
SCF = True  # self-consistently relax the moments (predict them) vs single-shot

# Exchange strength per shell (Hartree). NiO magnetism lives on the Ni-3d
# manifold, so the exchange is placed on l=2; s and p stay at 0.0 (a large
# field on the p-shell, as the old s/p-only set required, gives an
# unphysically large split).
#
# NOTE on the value: the free-atom Ni Stoner parameter is ~0.037 Ha
# (DEFAULT_STONER_I), but for this tight-binding model the Ni-3d DOS at E_F
# puts the Stoner criterion threshold I*N(E_F) > 1 between ~0.04 and 0.08 Ha,
# so 0.037 yields a non-magnetic (m -> 0) self-consistent solution. We use an
# effective I_d = 0.12 Ha, which is above threshold and self-consistently
# predicts Ni ~1.8 mu_B / net 2.0 mu_B per cell - close to experimental NiO
# (~1.7-1.9 mu_B on Ni). Lower it toward 0.037 to see the moment collapse.
STONER_I = {
    28: {0: 0.0, 1: 0.0, 2: 0.12},
    8:  {0: 0.0, 1: 0.0},
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


def matched_exchange_splitting(result, calc):
    """Physical exchange splitting (eV).

    The naive ``(eu - ed).abs().max()`` subtracts the two spin spectra by
    *band index*, but the channels are diagonalized and sorted
    independently, so equal indices need not label the same physical state
    once the exchange field reorders the bands. Here we instead match each
    spin-up state to the spin-down state of maximum eigenvector overlap
    (via the overlap matrix S) and take the energy difference of matched
    pairs. Returns (max_split, mean_split) in eV.
    """
    eu = result["eigenvalues_up"]            # [Nband, Nk] eV
    ed = result["eigenvalues_dn"]
    cu = result["eigenvectors_up"]           # [Norb, Nband, Nk]
    cd = result["eigenvectors_dn"]
    S = calc._results["overlap"]
    if S.ndim == 4:
        S = S[0]
    S = S.to(torch.complex128)               # [Norb, Norb, Nk]

    Nband, Nk = eu.shape
    diffs = []
    for ik in range(Nk):
        Cu = cu[..., ik].to(torch.complex128)
        Cd = cd[..., ik].to(torch.complex128)
        if Cu.ndim == 3:
            Cu, Cd = Cu.squeeze(0), Cd.squeeze(0)
        # overlap <up_i | S | dn_j>, shape [Nband_up, Nband_dn]
        O = (Cu.conj().transpose(0, 1) @ (S[..., ik] @ Cd)).abs()
        match = O.argmax(dim=1)              # best down-state per up-state
        diffs.append((eu[:, ik] - ed[match, ik]).abs())
    diffs = torch.stack(diffs)
    return diffs.max().item(), diffs.mean().item()


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
        scf=SCF,
        verbose=SCF,
    )

    # Predicted magnetic moments (Mulliken spin density n_up - n_dn).
    # With SCF=True these are self-consistently relaxed; with SCF=False they
    # just echo the imposed initial moments above.
    symbols = atoms.get_chemical_symbols()
    moments = result["moments"].tolist()
    print(f"SCF converged           : {result['converged']}")
    print("Predicted atom-wise moments (mu_B):")
    for i, (s, mu) in enumerate(zip(symbols, moments)):
        print(f"    atom {i:2d}  {s:>2}  {mu:+.4f}")
    print(f"Net magnetic moment     : {result['total_moment']:+.4f} mu_B/cell")

    max_split, mean_split = matched_exchange_splitting(result, calc)
    print(f"Fermi level             : {result['fermi_eV']:.3f} eV")
    print(f"Max exchange splitting  : {max_split:.3f} eV  (state-matched)")
    print(f"Mean exchange splitting : {mean_split:.3f} eV  (state-matched)")

    plot_spin_bands(
        result,
        fermi_shift_eV=result["fermi_eV"],
        filename="nio_spin_bands.png",
    )
    print("Wrote nio_spin_bands.png")


if __name__ == "__main__":
    main()
