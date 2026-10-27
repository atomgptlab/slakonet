"""Add in-house DFT exchange constants (VASP PBE+U energy mapping, Magnetic_MLIP
round 1) to the fit of the universal +U parameter.

1. validation: the current model (universal U) against the DFT J and, for
   reference, DFT against experiment on the shared couplings;
2. joint fit: loss = experimental + w * DFT, leave-one-compound-out, always
   scored against the experimental J (DFT only helps the fit).

DFT J come from ``results_ins/dft_shells.json``: one entry per compound and
shell, JS2 per bond in meV (J > 0 AFM), the same convention as ExpMagDB, and
a "reliable" flag (well-conditioned shell fit). Only reliable shells are used.

Usage:
    python fit_with_dft.py --work stoner_fit --dft .../results_ins/dft_shells.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import fit_stoner_table as f  # noqa: E402

HARTREE_EV = 27.211386


def dft_curves(data, grid, table, dft_rows, tol=0.15):
    """{id: [(JS2_dft, curve over grid)]} for the reliable DFT shells of the
    compounds in the precompute cache (same cells: the round-1 VASP cells are
    the INS-set structures of the dataset)."""
    out = {}
    comps = {c["id"]: c for c in data["compounds"]}
    for row in dft_rows:
        if not row.get("reliable") or row["name"] not in comps:
            continue
        comp = comps[row["name"]]
        coupling = {"pair": "", "distance": float(row["d"])}
        recs = table[comp["id"]]
        ok = [f._valid(r, comp["m_ref"]) for r in recs]
        vals = [
            (f._match(comp, r, coupling, tol) if good else None)
            for r, good in zip(recs, ok)
        ]
        curve = np.array([np.nan if v is None else v for v in vals])
        if np.isfinite(curve).any():
            out.setdefault(comp["id"], []).append((float(row["dft"]), curve))
    return out


def fit_U(ids_exp, ids_dft, cexp, cdft, grid, w):
    fine = np.exp(np.linspace(np.log(grid[0]), np.log(grid[-1]), 200))
    loss = []
    for U in fine:
        L = sum(f._compound_loss(cexp[i], U, grid) for i in ids_exp)
        L += w * sum(f._compound_loss(cdft[i], U, grid) for i in ids_dft)
        loss.append(L)
    return float(fine[int(np.argmin(loss))])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work", default="stoner_fit")
    p.add_argument("--dft", required=True)
    p.add_argument(
        "--weights", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0]
    )
    args = p.parse_args()

    data, grid, table = f._load(args.work, "hubbard")
    cexp = f._curves(data, grid, table)
    with open(args.dft) as fh:
        dft_rows = json.load(fh)
    cdft = dft_curves(data, grid, table, dft_rows)
    n_dft = sum(len(v) for v in cdft.values())
    print(
        f"experimental: {len(cexp)} compounds, {sum(len(v) for v in cexp.values())} couplings; "
        f"DFT (reliable, matched): {len(cdft)} compounds, {n_dft} couplings"
    )

    report = {}
    # 1. validation of the current universal U against DFT
    U0 = 2.97 / HARTREE_EV
    pred = []
    for i, rows in cdft.items():
        pred += f._predict_all(rows, U0, grid)
    report["model_vs_dft_universal_U"] = f._stats(pred)
    print("model (U = 2.97 eV) vs DFT:", report["model_vs_dft_universal_U"])

    # DFT vs experiment on shared couplings (same compound, distance within 0.15 A)
    exp_by_id = {c["id"]: c["couplings"] for c in data["compounds"]}
    pairs = []
    for row in dft_rows:
        if not row.get("reliable") or row["name"] not in exp_by_id:
            continue
        best = min(
            exp_by_id[row["name"]], key=lambda c: abs(c["distance"] - row["d"])
        )
        if abs(best["distance"] - row["d"]) < 0.15:
            pairs.append((best["JS2"], float(row["dft"])))
    report["dft_vs_experiment"] = f._stats(pairs)
    print("DFT (PBE+U) vs experiment:", report["dft_vs_experiment"])

    # 2. joint fit, leave-one-compound-out, scored on experiment
    ids = sorted(cexp)
    for w in args.weights:
        loo = []
        for i in ids:
            U = fit_U(
                [j for j in ids if j != i],
                [j for j in cdft if j != i],
                cexp,
                cdft,
                grid,
                w,
            )
            loo += f._predict_all(cexp[i], U, grid)
        U_all = fit_U(ids, list(cdft), cexp, cdft, grid, w)
        report[f"joint_w{w}"] = {
            "U_eV": round(U_all * HARTREE_EV, 2),
            "loo_exp": f._stats(loo),
        }
        print(
            f"w = {w}: U = {U_all * HARTREE_EV:.2f} eV  LOO vs experiment {f._stats(loo)}"
        )

    with open(os.path.join(args.work, "fit_with_dft_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
