"""E-V curve for diamond Si using the Si_only.pt model.

Scans the cubic lattice parameter, computes the total DFTB energy
(electronic + repulsive) at each point, and plots E(V).
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from ase.build import bulk

from slakonet.atoms import Geometry
from slakonet.main import SimpleDftb
from slakonet.optim import MultiElementSkfParameterOptimizer

MODEL_PATH = "../tests/Si_only.pt"
A_VALUES = np.linspace(4.2, 7.2, 25)


def compute_energy(a, model, kpoints=(3, 3, 3)):
    at = bulk("Si", "diamond", a=a)
    geom = Geometry.from_ase_atoms([at])
    calc = SimpleDftb(
        geom, model,
        kpoints=torch.tensor(list(kpoints)),
        device="cpu",
        with_eigenvectors=False,
        compute_forces=False,
        include_dos_data=False,
        repulsive=True,
        alpha=1.0,
    )
    res = calc.calculate()
    vol = float(np.abs(np.linalg.det(at.cell)))
    return {
        "a": float(a),
        "volume": vol,
        "E_rep": float(res["potential_energy"]),
        "E_elec": float(res["electronic_energy"]),
        "E_tot": float(res["energy"]),
    }


def main():
    model = MultiElementSkfParameterOptimizer.load_model(
        MODEL_PATH, method="compact"
    )
    model.eval()

    rows = [compute_energy(a, model) for a in A_VALUES]
    a = np.array([r["a"] for r in rows])
    V = np.array([r["volume"] for r in rows])
    E_tot = np.array([r["E_tot"] for r in rows])
    E_elec = np.array([r["E_elec"] for r in rows])
    E_rep = np.array([r["E_rep"] for r in rows])

    print(f"{'a (A)':>7} {'V (A^3)':>9} {'E_elec (eV)':>12} {'E_rep (eV)':>12} {'E_tot (eV)':>12}")
    for r in rows:
        print(f"{r['a']:>7.3f} {r['volume']:>9.3f} {r['E_elec']:>12.4f} {r['E_rep']:>12.4f} {r['E_tot']:>12.4f}")

    imin = int(np.argmin(E_tot))
    print(f"\nMin of E_tot at a = {a[imin]:.3f} A, V = {V[imin]:.3f} A^3, E = {E_tot[imin]:.4f} eV")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # E vs V (all three components)
    axes[0].plot(V, E_tot, "ko-", lw=1.5, label="E_tot")
    axes[0].plot(V, E_elec, "b.-", lw=0.8, alpha=0.7, label="E_elec")
    axes[0].plot(V, E_rep, "r.-", lw=0.8, alpha=0.7, label="E_rep")
    axes[0].set_xlabel(r"Volume (${\mathrm{\AA}}^3$)")
    axes[0].set_ylabel("Energy (eV)")
    axes[0].set_title("Si E-V (diamond)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].axvline(V[imin], color="gray", ls=":", lw=0.8)

    # Zoom on E_tot near minimum
    axes[1].plot(V, E_tot - E_tot.min(), "ko-", lw=1.5)
    axes[1].axvline(V[imin], color="gray", ls=":", lw=0.8)
    axes[1].set_xlabel(r"Volume (${\mathrm{\AA}}^3$)")
    axes[1].set_ylabel("E_tot - min (eV)")
    axes[1].set_title(f"E_tot (relative)  min@ a={a[imin]:.2f} A")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("si_eos.png", dpi=130)
    plt.close()
    print("Wrote si_eos.png")


if __name__ == "__main__":
    main()
