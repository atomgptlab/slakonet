"""Validate SlaKoNet stress vs finite-difference of the energy.

For a periodic Si cell apply small affine strains in all 6 Voigt
components and central-difference the energy. ASE's convention is
tensile-positive,

    sigma_a = -(1/V) * (E(eps) - E(-eps)) / (2 * eps)

(positive stress => cell wants to expand). Compare to the calculator's
reported stress (eV/Ang^3, Voigt). Requires the Bohr->Ang unit-fix in
SimpleDftb (otherwise the magnitude is off by 1/BOHR^3 ~ 6.7x).
"""

import numpy as np
import torch
from ase.build import bulk

from slakonet.optim import default_model
from slakonet.ase_calc import SlaKoNetCalculator

EPS = 1.0e-3   # engineering strain magnitude


def strained(atoms, voigt_idx, eps):
    """Apply a small Voigt-engineering strain to a copy of `atoms`."""
    e = np.zeros((3, 3))
    if voigt_idx == 0:    # xx
        e[0, 0] = eps
    elif voigt_idx == 1:  # yy
        e[1, 1] = eps
    elif voigt_idx == 2:  # zz
        e[2, 2] = eps
    elif voigt_idx == 3:  # yz
        e[1, 2] = e[2, 1] = eps * 0.5    # engineering -> symmetric
    elif voigt_idx == 4:  # xz
        e[0, 2] = e[2, 0] = eps * 0.5
    elif voigt_idx == 5:  # xy
        e[0, 1] = e[1, 0] = eps * 0.5
    F = np.eye(3) + e
    a = atoms.copy()
    a.set_cell(atoms.cell @ F.T, scale_atoms=True)
    return a


model = default_model().float()
calc = SlaKoNetCalculator(model, kpoints=(3, 3, 3), device="cpu", beta=1.0)

si = bulk("Si", "diamond", a=5.43)
si.calc = calc
E0 = si.get_potential_energy()
S_calc = si.get_stress()
V = si.get_volume()
print(f"E0 = {E0:.6f} eV   V = {V:.4f} A^3")
print("stress (eV/A^3, Voigt) reported by calc:", np.round(S_calc, 6))

S_fd = np.zeros(6)
for i in range(6):
    ap = strained(si, i, +EPS); ap.calc = calc
    am = strained(si, i, -EPS); am.calc = calc
    Ep = ap.get_potential_energy()
    Em = am.get_potential_energy()
    # ASE convention: sigma_a = -(1/V) dE/d(eps), engineering strain
    S_fd[i] = -(Ep - Em) / (2 * EPS) / V

print("stress (eV/A^3, Voigt) finite-difference :", np.round(S_fd, 6))
diff = S_calc - S_fd
print("difference                                :", np.round(diff, 6))
maxabs = float(np.abs(diff).max())
print(f"\nmax|sigma_auto - sigma_fd| = {maxabs:.3e} eV/A^3"
      f"  ({maxabs * 160.21766208:.3e} GPa)")
ratio_norm = np.linalg.norm(S_fd) / max(np.linalg.norm(S_calc), 1e-12)
print(f"|S_fd| / |S_calc| ratio  = {ratio_norm:.4f}  (1.0 = exact)")

ok = maxabs < 5e-5 or ratio_norm > 0.9 and ratio_norm < 1.1
print("STRESS OK" if ok else "MISMATCH - investigate")
