"""SCC-DFTB demo on rocksalt NaCl.

Runs slakonet with and without the self-consistent-charge correction on a
charge-transfer system and reports the resulting charges, SCC energy, gap,
and total energy. NaCl is a good showcase because a non-SCC TB model gets
the ionicity qualitatively wrong - only the SCC on-site penalty stabilises
a proper Na+/Cl- solution.

The universal slakonet SKF set gives a weaker ionic character than real
DFT/experiment (~0.2 e transfer here versus ~0.85 e Bader), which reflects
parameter-set limits and not a bug in the SCC loop.
"""
from __future__ import annotations

import torch
from ase.build import bulk

from slakonet.atoms import Geometry
from slakonet.main import SimpleDftb
from slakonet.optim import default_model


def run(atoms, use_scc):
    model = default_model()
    geom = Geometry.from_ase_atoms([atoms])
    calc = SimpleDftb(
        geom,
        model,
        kpoints=torch.tensor([3, 3, 3]),
        device="cpu",
        with_eigenvectors=True,
        compute_forces=False,
        include_dos_data=False,
        repulsive=False,
        alpha=1.0,
        use_scc=use_scc,
    )
    res = calc.calculate()
    out = {
        "E_elec": float(res["electronic_energy"]),
        "bandgap": float(res["bandgap"]),
        "E_tot": float(res["energy"]),
    }
    if use_scc:
        info = calc._scc_info
        dq = info["delta_q"].detach().cpu().numpy()
        out.update({
            "delta_q": dq.tolist(),
            "E_scc": float(info["E_scc_eV"]),
            "iter": info["n_iter"],
            "converged": bool(info["converged"]),
            "mu_eV": float(info["mu_Ha"]) * calc.H2E,
        })
    return out


def main():
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    print(f"Rocksalt NaCl, {len(atoms)} atoms: {atoms.get_chemical_symbols()}")

    no_scc = run(atoms, use_scc=False)
    sc = run(atoms, use_scc=True)

    col = lambda k, fmt=".4f": f"{no_scc.get(k, float('nan')):{fmt}}  |  {sc.get(k, float('nan')):{fmt}}"
    print(f"\n{'quantity':<22}  {'non-SCC':>12}  |  {'SCC':>12}")
    print("-" * 54)
    print(f"{'E_elec (eV)':<22}  {col('E_elec')}")
    print(f"{'E_scc  (eV)':<22}  {'—':>12}  |  {sc['E_scc']:>12.4f}")
    print(f"{'E_tot  (eV)':<22}  {col('E_tot')}")
    print(f"{'bandgap (eV)':<22}  {col('bandgap')}")
    print(f"{'delta_q [Na, Cl] (e)':<22}  {'—':>12}  |  "
          f"[{sc['delta_q'][0]:+.3f}, {sc['delta_q'][1]:+.3f}]")
    print(f"{'SCC iterations':<22}  {'—':>12}  |  {sc['iter']:>12d}")
    print(f"{'mu (eV)':<22}  {'—':>12}  |  {sc['mu_eV']:>12.4f}")


if __name__ == "__main__":
    main()
