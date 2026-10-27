"""J(U) from DFT+U (Magnetic_MLIP round-2 "uscan": U_eff = 0, 2.5, 5, 7.5 eV
on the 3d species, plus round 1 at the Materials Project U) against J(U) of
the slakonet +U model.

For each compound the shells are those of the combined round-1 + round-2 fit
(``dft_exchange.full.json``); each U batch is fitted with those shells (one
energy offset). Outputs, per compound and shell:

* DFT JS2 at every U_eff and the log-slope d ln|J| / d ln(U_eff) over U >= 2.5
  (superexchange J ~ t^2/U gives about -1);
* the slakonet +U JS2(U) from the precompute cache and its log-slope;
* the slakonet U that reproduces the DFT J at each DFT U (U equivalence).

Usage:
    python analyze_dft_uscan.py --mlip .../Magnetic_MLIP --work stoner_fit \
        --dft-full dft_reference/dft_exchange.full.json --out dft_reference/uscan.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import collect_dft_exchange as cde  # noqa: E402
import fit_stoner_table as f  # noqa: E402

HARTREE_EV = 27.211386
MP_U = {"V": 3.25, "Cr": 3.7, "Mn": 3.9, "Fe": 5.3, "Co": 3.32, "Ni": 6.2}


def fit_batch(batch_dir, bonds, shells, tol=0.03):
    """JS2 on fixed shells from one batch, or None."""
    _, rows = cde.batch_runs(batch_dir)
    if len(rows) < len(shells) + 2:
        return None
    sig = np.vstack([r[0] for r in rows])
    e = np.array([r[1] for r in rows]) * 1e3
    X_all, d_all = cde.shell_design(bonds, sig, 12)
    cols = []
    for d in shells:
        k = int(np.argmin([abs(x - d) for x in d_all]))
        if abs(d_all[k] - d) > tol:
            return None
        cols.append(k)
    A = np.hstack([np.ones((len(e), 1)), X_all[:, cols]])
    if np.linalg.matrix_rank(A) < A.shape[1] or np.linalg.cond(A) > 1e3:
        return None
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    return coef[1:].tolist()


def log_slope(U, J):
    U, J = np.asarray(U, float), np.asarray(J, float)
    ok = (U > 0) & (np.abs(J) > 1e-6)
    if ok.sum() < 2 or len(set(np.sign(J[ok]))) > 1:
        return None
    return float(np.polyfit(np.log(U[ok]), np.log(np.abs(J[ok])), 1)[0])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mlip", required=True)
    p.add_argument("--work", default="stoner_fit")
    p.add_argument("--dft-full", required=True)
    p.add_argument("--out", default="uscan.json")
    args = p.parse_args()
    mlip = Path(args.mlip)
    full = {r["name"]: r for r in json.loads(Path(args.dft_full).read_text())}
    data, grid, table = f._load(args.work, "hubbard")
    comps = {c["id"]: c for c in data["compounds"]}
    from ase.io import read

    out = []
    for udir in sorted((mlip / "dft_round2" / "uscan").glob("*_U*")):
        name, utag = udir.name.rsplit("_U", 1)
        if name not in full:
            continue
        rec = next((r for r in out if r["name"] == name), None)
        if rec is None:
            shells = [s["d"] for s in full[name]["shells"] if s["reliable"]]
            if not shells:
                continue
            reps = tuple(
                json.loads((udir / "manifest.json").read_text())["reps"]
            )
            n_prim = len(read(mlip / "results_ins" / name / "prim_mp.extxyz"))
            bonds = cde.tile(
                cde.load_bonds(
                    mlip / "results_ins" / name / "model_bonds_prim.npy"
                ),
                n_prim,
                reps,
            )
            el = json.loads((udir / "manifest.json").read_text())["U_species"][
                0
            ]
            rec = {
                "name": name,
                "element": el,
                "shells": shells,
                "dft": {},
                "_bonds": bonds,
            }
            # round 1 at the Materials Project U
            j_mp = fit_batch(mlip / "dft_inputs" / name, bonds, shells)
            if j_mp is not None:
                rec["dft"][MP_U.get(el, np.nan)] = j_mp
            out.append(rec)
        j = fit_batch(udir, rec["_bonds"], rec["shells"])
        if j is not None:
            rec["dft"][float(utag.replace("p", "."))] = j

    report = []
    for rec in out:
        name = rec["name"]
        Us = sorted(rec["dft"])
        entry = {"name": name, "element": rec["element"], "shells": []}
        for s_idx, d in enumerate(rec["shells"]):
            J_dft = [rec["dft"][u][s_idx] for u in Us]
            shell = {
                "d": d,
                "U_dft_eV": Us,
                "JS2_dft": J_dft,
                "slope_dft": log_slope(
                    [u for u in Us if u >= 2.5],
                    [j for u, j in zip(Us, J_dft) if u >= 2.5],
                ),
            }
            if name in comps:
                curve = [
                    (
                        f._match(comps[name], r, {"pair": "", "distance": d})
                        if r is not None
                        and "error" not in r
                        and r.get("converged")
                        else None
                    )
                    for r in table[name]
                ]
                U_tb = [
                    u * HARTREE_EV
                    for u, v in zip(grid, curve)
                    if v is not None
                ]
                J_tb = [v for v in curve if v is not None]
                shell.update(
                    U_tb_eV=U_tb, JS2_tb=J_tb, slope_tb=log_slope(U_tb, J_tb)
                )
                # slakonet U reproducing the DFT J at each DFT U (monotonic part)
                equiv = []
                for u_d, j_d in zip(Us, J_dft):
                    cand = [
                        (abs(jt - j_d), ut)
                        for ut, jt in zip(U_tb, J_tb)
                        if np.sign(jt) == np.sign(j_d)
                    ]
                    equiv.append(
                        min(cand)[1]
                        if cand and min(cand)[0] < 0.3 * abs(j_d)
                        else None
                    )
                shell["U_tb_equivalent_eV"] = equiv
            entry["shells"].append(shell)
        report.append(entry)
        for sh in entry["shells"]:
            print(
                f"{name:8s} d={sh['d']:.2f}  DFT J(U) "
                + " ".join(
                    f"{u:.1f}:{j:+.1f}"
                    for u, j in zip(sh["U_dft_eV"], sh["JS2_dft"])
                )
                + f"  slope DFT {sh['slope_dft'] if sh['slope_dft'] is None else round(sh['slope_dft'], 2)}"
                + (
                    f" TB {round(sh['slope_tb'], 2) if sh.get('slope_tb') is not None else None}"
                    f"  U_TB equiv {[None if x is None else round(x, 1) for x in sh.get('U_tb_equivalent_eV', [])]}"
                    if "slope_tb" in sh
                    else ""
                )
            )
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
