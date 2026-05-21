"""Example: SlaKoNet as an ASE calculator.

Demonstrates:
  * loading the model ONCE and reusing it for every structure/call
  * energy / forces / stress via the standard ASE API
  * the compute_forces / compute_stress toggles (energy-only fast path)
  * band structure (-> PNG) and total DOS (-> PNG)
  * a second structure with NO model reload
"""

import time

import numpy as np
from ase.build import bulk

from slakonet.optim import default_model
from slakonet.ase_calc import SlaKoNetCalculator

# ---- one-time model load -------------------------------------------------
t0 = time.perf_counter()
MODEL = default_model().float()
print(f"[*] model loaded once in {time.perf_counter() - t0:.1f}s")

# ---- full calculator (energy + forces + stress) --------------------------
calc = SlaKoNetCalculator(MODEL, kpoints=(3, 3, 3), device="cpu")

si = bulk("Si", "diamond", a=5.43)
si.calc = calc
t0 = time.perf_counter()
E = si.get_potential_energy()
F = si.get_forces()
S = si.get_stress()
print(f"\n[Si] E = {E:.4f} eV   ({time.perf_counter() - t0:.1f}s)")
print(f"[Si] |F|max = {np.abs(F).max():.3e} eV/Ang   F.shape={F.shape}")
print(f"[Si] stress(Voigt, eV/Ang^3) = {np.round(S, 6)}")
print(f"[Si] gap = {calc.get_bandgap():.3f} eV   "
      f"E_fermi = {calc.get_fermi_level():.3f} eV")

# ---- energy-only fast path (forces OFF) ---------------------------------
calc_fast = SlaKoNetCalculator(
    MODEL, kpoints=(3, 3, 3), device="cpu",
    compute_forces=False, compute_stress=False,
)
si2 = bulk("Si", "diamond", a=5.43)
si2.calc = calc_fast
t0 = time.perf_counter()
E2 = si2.get_potential_energy()
print(f"\n[Si fast] E = {E2:.4f} eV (forces off, "
      f"{time.perf_counter() - t0:.1f}s)")

# ---- band structure + DOS (same loaded model, no reload) -----------------
bs = calc.band_structure(si, path="GXWKGL", npoints=120,
                          savefig="si_v1_bands.png")
print(f"\n[Si] band path {bs['path']}  gap={bs['gap']:.3f} eV  "
      f"VBM={bs['vbm']:.3f}  CBM={bs['cbm']:.3f}  -> si_v1_bands.png")

e_grid, dos = calc.dos(si)
print(f"[Si] DOS: {len(e_grid)} pts, "
      f"E in [{e_grid.min():.1f}, {e_grid.max():.1f}] eV")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(e_grid, dos, lw=1.0)
ax.axvline(0.0, color="k", ls="--", lw=0.6)
ax.set_xlabel(r"E - E$_F$ (eV)")
ax.set_ylabel("DOS")
ax.set_title("Si total DOS (slakonet v1)")
ax.set_xlim(-10, 10)
fig.tight_layout()
plt.savefig("si_v1_dos.png", dpi=200)
plt.close(fig)
print("[Si] DOS -> si_v1_dos.png")

# ---- SECOND structure, SAME calculator, NO reload ------------------------
ge = bulk("Ge", "diamond", a=5.66)
ge.calc = calc  # reuse: model already in memory
t0 = time.perf_counter()
print(f"\n[Ge] E = {ge.get_potential_energy():.4f} eV   "
      f"gap = {calc.get_bandgap():.3f} eV   "
      f"(no model reload, {time.perf_counter() - t0:.1f}s)")

print("\n[*] done")
