"""Verify SlaKoNetCalculator forces come from torch autograd.

Strategy: displace an atom off equilibrium so forces are sizable, take
the calculator's forces (beta=1.0 so F = -dE/dx exactly), and compare to
a central finite-difference of the potential energy. Agreement => the
reported forces really are the autograd gradient of the energy.
"""

import numpy as np
import torch
from ase.build import bulk

from slakonet.optim import default_model
from slakonet.ase_calc import SlaKoNetCalculator

model = default_model().float()

# beta=1.0  => forces == -dE/dx (default beta=0.1 scales them down 10x)
calc = SlaKoNetCalculator(model, kpoints=(2, 2, 2), beta=1.0, device="cpu")


def energy(pos):
    a = bulk("Si", "diamond", a=5.43)
    a.positions = pos
    a.calc = calc
    return a.get_potential_energy()


# equilibrium structure with one atom pushed off-site -> real forces
base = bulk("Si", "diamond", a=5.43)
p0 = base.positions.copy()
p0[1] += np.array([0.12, -0.07, 0.05])  # Angstrom

a0 = bulk("Si", "diamond", a=5.43)
a0.positions = p0
a0.calc = calc
E0 = a0.get_potential_energy()  # triggers calculate; forces filled too
F_auto = a0.get_forces()
print("forces present in calc.results:", "forces" in calc.results)
print("F_autograd (eV/Ang):\n", np.round(F_auto, 5))

# central finite difference of the energy
h = 2e-3
F_fd = np.zeros_like(p0)
for i in range(len(p0)):
    for d in range(3):
        pp = p0.copy(); pp[i, d] += h
        pm = p0.copy(); pm[i, d] -= h
        F_fd[i, d] = -(energy(pp) - energy(pm)) / (2 * h)

print("F_finite_diff (eV/Ang):\n", np.round(F_fd, 5))

den = np.maximum(np.abs(F_fd).max(), 1e-8)
max_abs = np.abs(F_auto - F_fd).max()
print(f"\nmax|F_auto - F_fd|      = {max_abs:.3e} eV/Ang")
print(f"relative (vs max|F_fd|) = {max_abs / den:.2%}")
ok = max_abs < 5e-3 or (max_abs / den) < 0.02
print("AUTOGRAD FORCES VERIFIED" if ok else "MISMATCH - investigate")

# also show the beta scaling explicitly
calc_b = SlaKoNetCalculator(model, kpoints=(2, 2, 2), beta=0.1, device="cpu")
ab = bulk("Si", "diamond", a=5.43); ab.positions = p0; ab.calc = calc_b
ab.get_potential_energy()
F_b = ab.get_forces()
ratio = np.linalg.norm(F_b) / max(np.linalg.norm(F_auto), 1e-12)
print(f"\nbeta=0.1 vs beta=1.0 force-norm ratio = {ratio:.3f} "
      f"(expected ~0.1: forces are -beta*dE/dx)")
