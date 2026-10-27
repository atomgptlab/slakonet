"""Round-4 DFT+U J(U) against experiment and against the slakonet +U model,
by anion class (O, F, halide, chalcogenide, metallic).

Input: ``round4_JU.json`` (collect_round4.py) and the +U precompute cache of
fit_stoner_table.py (J_TB at U_TB = 1-15 eV for the dataset compounds).

A DFT shell at a given U is "reliable" when err < max(1 meV, 0.3 |JS2|).
Outputs per anion class:

(a) DFT vs experiment at each DFT U_eff (couplings matched by distance within
    0.15 A): fraction within x2, same sign, median log10(|J_dft/J_exp|);
(b) U equivalence: the slakonet U_TB whose J_TB equals the DFT J (same sign,
    within x1.5) at each DFT U, median per class; compared with the default
    ANION_U_EV of hubbard_spin.py.

Usage:
    python analyze_round4.py --work stoner_fit --ju dft_reference/round4_JU.json \
        --out dft_reference/round4_analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import fit_stoner_table as f  # noqa: E402

HARTREE_EV = 27.211386
D_TOL = 0.15


def reliable(j, e):
    return (
        j is not None
        and e is not None
        and np.isfinite(e)
        and e < max(1.0, 0.3 * abs(j))
    )


def anion_class(comp):
    from ase import Atoms

    from slakonet.magnetic_sites import site_anion_class

    atoms = Atoms(
        comp["symbols"],
        positions=comp["positions"],
        cell=comp["cell"],
        pbc=True,
    )
    i = next(k for k, m in enumerate(comp["initial_moments"]) if m != 0)
    return site_anion_class(atoms, i)


def tb_curve(comp, recs, d):
    vals = [
        (
            f._match(comp, r, {"pair": "", "distance": d})
            if f._valid(r, comp["m_ref"])
            else None
        )
        for r in recs
    ]
    return np.array([np.nan if v is None else v for v in vals])


def u_equivalent(curve, grid_ev, j_dft, max_ratio=1.5):
    """U_TB (eV) where the TB curve (log-interpolated) equals j_dft, or None."""
    ok = np.isfinite(curve) & (np.sign(curve) == np.sign(j_dft)) & (curve != 0)
    if ok.sum() == 0:
        return None
    u, c = np.log(grid_ev[ok]), np.log(np.abs(curve[ok]))
    fine = np.linspace(u[0], u[-1], 400) if len(u) > 1 else u
    cf = np.interp(fine, u, c) if len(u) > 1 else c
    k = int(np.argmin(np.abs(cf - np.log(abs(j_dft)))))
    if abs(cf[k] - np.log(abs(j_dft))) > np.log(max_ratio):
        return None
    return float(np.exp(fine[k]))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work", default="stoner_fit")
    p.add_argument("--ju", required=True)
    p.add_argument("--out", default="round4_analysis.json")
    args = p.parse_args()
    from slakonet.hubbard_spin import ANION_U_EV

    data, grid, table = f._load(args.work, "hubbard")
    grid_ev = np.array(grid) * HARTREE_EV
    comps = {c["id"]: c for c in data["compounds"]}
    ju = json.loads(Path(args.ju).read_text())

    vs_exp = defaultdict(list)  # (class, U) -> [(J_exp, J_dft)]
    equiv = defaultdict(list)  # (class, U) -> [U_TB]
    per_compound = []
    for rec in ju:
        name = rec["name"]
        if name not in comps:
            print(f"{name}: not in the dataset, skipped")
            continue
        comp = comps[name]
        cls = anion_class(comp)
        row = {"name": name, "class": cls, "shells": []}
        for s, d in enumerate(rec["shells_A"]):
            exp = min(comp["couplings"], key=lambda c: abs(c["distance"] - d))
            exp = exp if abs(exp["distance"] - d) < D_TOL else None
            curve = tb_curve(comp, table[name], d)
            sh = {"d": d, "J_exp": exp["JS2"] if exp else None, "dft": {}}
            for u, v in rec["U"].items():
                if not v["JS2"] or s >= len(v["JS2"]):
                    continue
                j, e = v["JS2"][s], v["err"][s]
                if not reliable(j, e):
                    continue
                ueq = u_equivalent(curve, grid_ev, j)
                sh["dft"][u] = {"J": j, "U_tb_equiv": ueq}
                ub = round(float(u))  # 3.32 -> 3, 5.3 -> 5 ...
                if exp and abs(exp["JS2"]) >= 1.0:
                    vs_exp[(cls, ub)].append((exp["JS2"], j))
                if ueq is not None and abs(j) >= 1.0:
                    equiv[(cls, ub)].append(ueq)
            row["shells"].append(sh)
        per_compound.append(row)

    summary = {
        "dft_vs_exp": {},
        "u_equivalence": {},
        "default_anion_U_eV": ANION_U_EV,
    }
    print("\n(a) DFT vs experiment (|J_exp| >= 1 meV)")
    print(
        f"{'class':13s} {'U':>3s} {'N':>3s} {'x2':>5s} {'sign':>5s} {'med log10 ratio':>16s}"
    )
    for cls, u in sorted(vs_exp):
        pairs = np.array(vs_exp[(cls, u)])
        ratio = np.abs(pairs[:, 1] / pairs[:, 0])
        sign = np.sign(pairs[:, 1]) == np.sign(pairs[:, 0])
        s = {
            "N": len(pairs),
            "within_x2": float(np.mean(sign & (ratio > 0.5) & (ratio < 2))),
            "sign": float(np.mean(sign)),
            "median_log10_ratio": float(np.median(np.log10(ratio))),
        }
        summary["dft_vs_exp"][f"{cls}|{u}"] = s
        print(
            f"{cls:13s} {u:3d} {s['N']:3d} {s['within_x2']:5.2f} {s['sign']:5.2f} "
            f"{s['median_log10_ratio']:+16.2f}"
        )
    print("\n(b) slakonet U_TB reproducing the DFT J (|J_dft| >= 1 meV)")
    print(
        f"{'class':13s} {'U_dft':>5s} {'N':>3s} {'median U_TB':>12s} {'IQR':>12s}  default U_TB"
    )
    for cls, u in sorted(equiv):
        x = np.array(equiv[(cls, u)])
        q1, q2, q3 = np.percentile(x, [25, 50, 75])
        summary["u_equivalence"][f"{cls}|{u}"] = {
            "N": len(x),
            "median": float(q2),
            "q25": float(q1),
            "q75": float(q3),
        }
        print(
            f"{cls:13s} {u:5d} {len(x):3d} {q2:12.2f} {q1:5.1f}-{q3:<6.1f}  "
            f"{ANION_U_EV.get(cls, float('nan')):.2f}"
        )
    Path(args.out).write_text(
        json.dumps({"summary": summary, "compounds": per_compound}, indent=1)
    )


if __name__ == "__main__":
    main()
