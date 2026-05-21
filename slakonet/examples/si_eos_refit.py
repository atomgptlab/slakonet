"""Refit the Si-Si repulsive against a Birch-Murnaghan E-V target.

Uses a flexible 6-parameter functional form (double-exponential + quadratic
cutoff term), fits (A1, α1, A2, α2, B, r_c) against the BM reference via
scipy.optimize.least_squares. This diagnostic answers: can *any* additive
pairwise repulsive reproduce the target E-V on top of slakonet's electronic
energy?

If the fit produces a monotonically-decreasing repulsive that matches BM,
we're in business. If the optimizer is forced into non-physical (non-
monotonic or attractive) regions, we've confirmed the underlying electronic
model can't be rescued by a pairwise repulsive alone.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from ase.build import bulk

from slakonet.atoms import Geometry
from slakonet.main import SimpleDftb
from slakonet.optim import MultiElementSkfParameterOptimizer

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tests", "Si_only.pt"
)
A_SCAN = np.linspace(4.6, 6.8, 20)

V0 = 40.47
B0_GPA = 98.0
B0 = B0_GPA / 160.2176634
BP = 4.0
E0_SHIFT = -42.0  # eV; adjusted so BM(V) lies near E_elec+E_rep_orig scale


def birch_murnaghan(V, V0, B0, Bp, E0):
    x = (V0 / V) ** (2.0 / 3.0)
    return E0 + 9.0 * V0 * B0 / 16.0 * (
        (x - 1.0) ** 3 * Bp + (x - 1.0) ** 2 * (6.0 - 4.0 * x)
    )


def gather_bonds(calc, cutoff_bohr: float):
    """Flat list of all pair distances < cutoff in Bohr, with multiplicities."""
    dv = calc.periodic.distance_vectors.detach().cpu().numpy()
    d = np.linalg.norm(dv, axis=-1)
    mask = (d > 1e-3) & (d < cutoff_bohr)
    return d[mask]


def rep_func(r, params):
    A1, a1, A2, a2, B, rc = params
    # double exponential + cutoff-quadratic
    e = A1 * np.exp(-a1 * r) + A2 * np.exp(-a2 * r)
    e = np.where(r < rc, e + B * (rc - r) ** 2, e)
    return e


def main():
    model = MultiElementSkfParameterOptimizer.load_model(
        MODEL_PATH, method="compact"
    )
    model.eval()

    cutoff_bohr = float(model.get_updated_skfs()["Si-Si"].r_spline.cutoff)
    print(f"SKF cutoff = {cutoff_bohr:.3f} Bohr")

    # Precompute per-geometry bond list + electronic energy (both fixed)
    rows = []
    for a in A_SCAN:
        at = bulk("Si", "diamond", a=float(a))
        geom = Geometry.from_ase_atoms([at])
        calc = SimpleDftb(
            geom, model, kpoints=torch.tensor([3, 3, 3]), device="cpu",
            with_eigenvectors=False, compute_forces=False,
            include_dos_data=False, repulsive=False, alpha=1.0,
        )
        res = calc.calculate()
        V = float(np.abs(np.linalg.det(at.cell)))
        bonds = gather_bonds(calc, cutoff_bohr)
        rows.append(dict(
            a=float(a), V=V, E_elec=float(res["electronic_energy"]),
            bonds=bonds,
        ))

    V_arr = np.array([r["V"] for r in rows])
    E_elec = np.array([r["E_elec"] for r in rows])
    E_BM = birch_murnaghan(V_arr, V0, B0, BP, E0_SHIFT)
    target_E_rep = E_BM - E_elec   # what E_rep(V) must be

    # Residual: for each geometry, sum rep_func over its bonds and 0.5 for pairs
    def residuals(params):
        err = np.empty(len(rows))
        for i, r in enumerate(rows):
            E_rep_i = 0.5 * rep_func(r["bonds"], params).sum()
            err[i] = (E_elec[i] + E_rep_i) - E_BM[i]
        return err

    # start: small double-exp + zero quadratic
    p0 = np.array([0.5, 0.6, 0.5, 1.5, 0.0, cutoff_bohr])
    lb = np.array([-50, 0.01, -50, 0.01, -5.0, cutoff_bohr - 1e-6])
    ub = np.array([ 50, 10.0,  50, 10.0,  5.0, cutoff_bohr + 1e-6])
    lsq = least_squares(residuals, p0, bounds=(lb, ub), max_nfev=500)
    A1, a1, A2, a2, B, rc = lsq.x
    print(f"Fit: A1={A1:.3f}, α1={a1:.3f},  A2={A2:.3f}, α2={a2:.3f},  "
          f"B={B:.3f}, rc={rc:.3f}")
    print(f"    residual-norm = {np.linalg.norm(lsq.fun):.3f} eV")

    # Check monotonicity on a fine grid
    r_fine = np.linspace(2.0, cutoff_bohr, 400)
    E_fine = rep_func(r_fine, lsq.x)
    dE = np.diff(E_fine)
    monotone_decreasing = np.all(dE <= 1e-6)
    nonnegative = np.all(E_fine >= -1e-3)
    print(f"Fitted E_rep monotonically decreasing? {monotone_decreasing}")
    print(f"Fitted E_rep non-negative?            {nonnegative}")
    if not (monotone_decreasing and nonnegative):
        print(">>> Fit requires unphysical (non-monotone or attractive) repulsive.")
        print(">>> Interpretation: slakonet's electronic energy alone cannot be")
        print(">>> balanced by any physical pairwise repulsive to give min at V0.")
    else:
        print(">>> Fit is physically reasonable; refitting the SKF spline will work.")

    # Plot: E-V curve with fitted repulsive + E_rep(r) curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    E_fit = np.array(
        [E_elec[i] + 0.5 * rep_func(rows[i]["bonds"], lsq.x).sum()
         for i in range(len(rows))]
    )
    axes[0].plot(V_arr, E_elec, "g.-", lw=0.8, alpha=0.6, label="E_elec only")
    axes[0].plot(V_arr, E_fit, "b-", lw=1.5, label="E_elec + E_rep(fit)")
    axes[0].plot(V_arr, E_BM, "r--", lw=1, label="BM target")
    axes[0].axvline(V0, color="gray", ls=":", lw=0.8, label=f"V0={V0:.1f}")
    axes[0].set_xlabel(r"V (${\mathrm{\AA}}^3$)")
    axes[0].set_ylabel("E (eV)")
    axes[0].set_title("Refit of repulsive vs Birch-Murnaghan")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(r_fine, E_fine, "b-", lw=1.5, label="fitted E_rep(r)")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("r (Bohr)")
    axes[1].set_ylabel("E_rep (eV per pair)")
    axes[1].set_title("Fitted repulsive shape")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("si_eos_refit.png", dpi=130)
    plt.close()
    print("Wrote si_eos_refit.png")


if __name__ == "__main__":
    main()
