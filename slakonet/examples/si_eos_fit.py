"""Fit a short-range repulsive correction for Si so the E-V curve has a
physical equilibrium at a = 5.43 A.

Approach:
  1. Sweep a, record E_elec(V) and E_rep_orig(V) from the loaded model.
  2. Build a Birch-Murnaghan reference with V0 = 40.47 A^3 (a = 5.43),
     B0 = 98 GPa, B' = 4.
  3. Fit a 2-parameter correction  V_corr(r_nn) = A * exp(-alpha (r_nn - r_ref))
     via least-squares against (E_BM - E_elec - E_rep_orig).
  4. Plot the corrected E-V curve on top of the original.

This does not modify the SKF - it demonstrates that a simple additive
correction reshapes the curve correctly. For a production fix you would
then fit the actual spline_coef of the Si-Si r_spline with the same
residuals target.
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from ase.build import bulk

from slakonet.atoms import Geometry
from slakonet.main import SimpleDftb
from slakonet.optim import MultiElementSkfParameterOptimizer

import os
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tests", "Si_only.pt"
)
A_SCAN = np.linspace(4.6, 7.0, 25)

# Birch-Murnaghan reference
V0_TARGET = 40.47        # A^3, Si diamond a = 5.43 A
B0_GPA = 98.0            # GPa (experimental)
GPA_TO_EV_A3 = 1.0 / 160.2176634
B0_EV_A3 = B0_GPA * GPA_TO_EV_A3
BP = 4.0                 # dimensionless


def birch_murnaghan(V, V0, B0, Bp, E0):
    x = (V0 / V) ** (2.0 / 3.0)
    return E0 + 9.0 * V0 * B0 / 16.0 * (
        (x - 1.0) ** 3 * Bp + (x - 1.0) ** 2 * (6.0 - 4.0 * x)
    )


def scan(model, a_vals):
    rows = []
    for a in a_vals:
        at = bulk("Si", "diamond", a=float(a))
        geom = Geometry.from_ase_atoms([at])
        calc = SimpleDftb(
            geom, model, kpoints=torch.tensor([3, 3, 3]), device="cpu",
            with_eigenvectors=False, compute_forces=False,
            include_dos_data=False, repulsive=True, alpha=1.0,
        )
        res = calc.calculate()
        V = float(np.abs(np.linalg.det(at.cell)))
        dv = calc.periodic.distance_vectors.detach().cpu().numpy()
        dn = np.linalg.norm(dv, axis=-1)
        nn_bohr = float(dn[dn > 1e-3].min())
        rows.append(dict(
            a=float(a), V=V, nn_bohr=nn_bohr,
            E_elec=float(res["electronic_energy"]),
            E_rep=float(res["potential_energy"]),
        ))
    return rows


def main():
    model = MultiElementSkfParameterOptimizer.load_model(
        MODEL_PATH, method="compact")
    model.eval()

    rows = scan(model, A_SCAN)
    a = np.array([r["a"] for r in rows])
    V = np.array([r["V"] for r in rows])
    r_nn = np.array([r["nn_bohr"] for r in rows])
    E_elec = np.array([r["E_elec"] for r in rows])
    E_rep = np.array([r["E_rep"] for r in rows])

    # Anchor Birch-Murnaghan E0 so that E_BM(V0) = E_elec(V0) + E_rep_orig(V0),
    # keeping absolute offset comparable.
    iV0 = int(np.argmin(np.abs(V - V0_TARGET)))
    E_anchor = E_elec[iV0] + E_rep[iV0]
    E_BM = birch_murnaghan(V, V0_TARGET, B0_EV_A3, BP, E0=E_anchor)

    # residual we want the correction to match
    target_corr = E_BM - E_elec - E_rep

    r_ref = r_nn[iV0]
    def model_corr(params):
        A, alpha = params
        return A * np.exp(-alpha * (r_nn - r_ref))
    def residuals(params):
        return model_corr(params) - target_corr

    # initial guess
    p0 = np.array([5.0, 2.0])
    lsq = least_squares(residuals, p0, bounds=([0.0, 0.1], [50.0, 10.0]))
    A_fit, alpha_fit = lsq.x
    print(f"Fit: A = {A_fit:.4f} eV, alpha = {alpha_fit:.4f} 1/Bohr "
          f"(r_ref = {r_ref:.3f} Bohr)")
    print(f"Target V0 = {V0_TARGET:.2f} A^3, B0 = {B0_GPA:.1f} GPa")

    E_corr = model_corr(lsq.x)
    E_tot_fixed = E_elec + E_rep + E_corr
    imin = int(np.argmin(E_tot_fixed))
    print(f"After correction: min at a = {a[imin]:.3f} A "
          f"(V = {V[imin]:.2f} A^3), E_tot = {E_tot_fixed[imin]:.4f} eV")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(V, E_elec + E_rep, "kx-", lw=1, label="original E_tot")
    axes[0].plot(V, E_tot_fixed, "b.-", lw=1.5, label="corrected E_tot")
    axes[0].plot(V, E_BM, "r--", lw=1, label="Birch-Murnaghan target")
    axes[0].axvline(V0_TARGET, color="gray", ls=":", lw=0.8,
                    label=f"V0 = {V0_TARGET:.1f} A^3")
    axes[0].set_xlabel(r"Volume (${\mathrm{\AA}}^3$)")
    axes[0].set_ylabel("E (eV)")
    axes[0].set_title("Si E-V (with correction fit)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(r_nn, E_rep, "kx-", lw=1, label="E_rep original")
    axes[1].plot(r_nn, E_corr, "b.-", lw=1.5, label="E_corr = A exp(-a(r-r_ref))")
    axes[1].plot(r_nn, E_rep + E_corr, "g-", lw=1, label="total repulsive")
    axes[1].set_xlabel(r"$r_{NN}$ (Bohr)")
    axes[1].set_ylabel("E (eV)")
    axes[1].set_title(f"Repulsive correction  "
                      f"A={A_fit:.2f} eV, $\\alpha$={alpha_fit:.2f} /Bohr")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_yscale("symlog", linthresh=0.1)

    plt.tight_layout()
    plt.savefig("si_eos_fit.png", dpi=130)
    plt.close()
    print("Wrote si_eos_fit.png")


if __name__ == "__main__":
    main()
