"""Analysis of a spin-polarized SK training run (cluster training results).

1. Bands: RMSE per structure from evaluate_r*.json. The consistent baseline
   is the base model with its own potentials (evaluate_r0, "base"); round N
   "trained" uses the potentials converged with the corrections of round N-1.
   Compared on the structures usable in every round, by ligand class.
2. Corrections: the largest changes of the integrals and on-site levels.
3. Exchange: J(U) caches of the base model (local stoner_fit) and of the
   trained one (jval_rN), same 70-compound experimental set. On the same
   couplings: leave-one-compound-out with one U per ligand class, the
   in-sample fit, and the oracle (best U per compound). A coupling without a
   physical solution counts as calc = 0.

Usage:
    python analyze_spin_training.py --results training_results \
        --base-work stoner_fit --round r2 --out spin_training_r2.json
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


def band_summary(res):
    ev = {
        r: json.loads((res / "data" / f"evaluate_{r}.json").read_text())
        for r in ("r0", "r1", "r2")
        if (res / "data" / f"evaluate_{r}.json").exists()
    }
    table = defaultdict(dict)  # id -> {label: rmse}
    meta = {}
    for r, rows in ev.items():
        for row in rows:
            key = (
                "base"
                if (row["model"] == "base" and r == "r0")
                else (f"trained_{r}" if row["model"] == "trained" else None)
            )
            if key:
                table[row["id"]][key] = row["rmse_eV"]
                meta[row["id"]] = (
                    row["compound"],
                    row["class"],
                    row["holdout"],
                )
    labels = ["base"] + [f"trained_{r}" for r in ev]
    common = [i for i, v in table.items() if all(k in v for k in labels)]
    out = {
        "labels": labels,
        "n_common": len(common),
        "by_class": {},
        "by_compound": {},
    }
    groups = defaultdict(list)
    for i in common:
        comp, cls, hold = meta[i]
        groups[("holdout" if hold else "train", cls)].append(i)
        groups[("holdout" if hold else "train", "all")].append(i)
        out["by_compound"].setdefault(
            comp, {"class": cls, "holdout": hold, **{k: [] for k in labels}}
        )
        for k in labels:
            out["by_compound"][comp][k].append(table[i][k])
    for (split, cls), ids in sorted(groups.items()):
        out["by_class"][f"{split}|{cls}"] = {
            "N": len(ids),
            **{k: float(np.mean([table[i][k] for i in ids])) for k in labels},
        }
    for c in out["by_compound"].values():
        for k in labels:
            c[k] = float(np.mean(c[k]))
    return out


def corrections_summary(res, rnd, top=15):
    s = json.loads((res / "data" / f"corr_{rnd}.json").read_text())
    ints = sorted(
        s["integrals"].items(), key=lambda kv: -abs(kv[1]["log_scale"])
    )
    ons = sorted(s["onsite_eV"].items(), key=lambda kv: -abs(kv[1]))
    nz = [v for v in s["integrals"].values() if abs(v["log_scale"]) > 0.01]
    return {
        "meta": s.get("meta", {}),
        "n_integrals": len(s["integrals"]),
        "n_changed_gt_1pct": len(nz),
        "largest_scale": [
            (
                k,
                round(np.exp(v["log_scale"]), 3),
                round(v["decay_per_bohr"], 3),
            )
            for k, v in ints[:top]
        ],
        "largest_onsite_eV": [(k, round(v, 3)) for k, v in ons[:top]],
    }


def anion_classes(data):
    from ase import Atoms

    from slakonet.magnetic_sites import site_anion_class

    out = {}
    for c in data["compounds"]:
        atoms = Atoms(
            c["symbols"], positions=c["positions"], cell=c["cell"], pbc=True
        )
        i = next(k for k, m in enumerate(c["initial_moments"]) if m != 0)
        out[c["id"]] = site_anion_class(atoms, i)
    return out


def j_stats(work, keys, classes):
    """In-sample, leave-one-out (one U per class) and oracle on ``keys``
    = [(compound, coupling index)]; missing curves count as no solution."""
    data, grid, table = f._load(work, "hubbard")
    comps = {c["id"]: c for c in data["compounds"]}
    curves = {}
    for cid, k in keys:
        comp, c = comps[cid], comps[cid]["couplings"][k]
        recs = table[cid]
        vals = [
            (f._match(comp, r, c) if f._valid(r, comp["m_ref"]) else None)
            for r in recs
        ]
        curves.setdefault(cid, []).append(
            (c["JS2"], np.array([np.nan if v is None else v for v in vals]))
        )
    ids = sorted(curves)
    by_cls = defaultdict(list)
    for i in ids:
        by_cls[classes[i]].append(i)
    fitted = {
        cls: f._fit_element(g, curves, grid) for cls, g in by_cls.items()
    }
    insample, loo, oracle, per_class = [], [], [], defaultdict(list)
    for i in ids:
        others = [j for j in by_cls[classes[i]] if j != i]
        U = (
            f._fit_element(others, curves, grid)
            if others
            else fitted[classes[i]]
        )
        p = f._predict_all(curves[i], U, grid)
        loo += p
        per_class[classes[i]] += p
        insample += f._predict_all(curves[i], fitted[classes[i]], grid)
        oracle += f._predict_all(
            curves[i], f._fit_element([i], curves, grid), grid
        )
    return {
        "U_by_class_eV": {
            k: round(v * HARTREE_EV, 2) for k, v in fitted.items()
        },
        "in_sample": f._stats(insample),
        "leave_one_out": f._stats(loo),
        "oracle": f._stats(oracle),
        "loo_by_class": {k: f._stats(v) for k, v in per_class.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", required=True)
    p.add_argument("--base-work", default="stoner_fit")
    p.add_argument("--round", default="r2")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    res = Path(args.results)
    report = {
        "bands": band_summary(res),
        "corrections": corrections_summary(res, args.round),
    }

    b = report["bands"]
    print(
        f"BANDS (mean RMSE eV, {b['n_common']} structures usable in every round)"
    )
    print(
        f"{'split|class':22s} {'N':>4s} "
        + " ".join(f"{k:>11s}" for k in b["labels"])
    )
    for key, v in b["by_class"].items():
        print(
            f"{key:22s} {v['N']:4d} "
            + " ".join(f"{v[k]:11.3f}" for k in b["labels"])
        )

    c = report["corrections"]
    print(
        f"\nCORRECTIONS {args.round}: {c['n_changed_gt_1pct']}/{c['n_integrals']} integrals "
        "changed by > 1 %"
    )
    print("largest scale factors (integral, factor, decay /bohr):")
    for row in c["largest_scale"]:
        print("   ", row)
    print("largest on-site shifts (eV):", c["largest_onsite_eV"][:10])

    jv = res / f"jval_{args.round}"
    data, _, _ = f._load(args.base_work, "hubbard")
    classes = anion_classes(data)
    keys = [
        (cmp["id"], k)
        for cmp in data["compounds"]
        for k, cp in enumerate(cmp["couplings"])
        if abs(cp["JS2"]) >= 1.0
    ]
    report["exchange"] = {
        "base": j_stats(args.base_work, keys, classes),
        "trained": j_stats(str(jv), keys, classes),
    }
    print(
        "\nEXCHANGE (same couplings, |J_exp| >= 1 meV; no solution counts as 0)"
    )
    for label in ("base", "trained"):
        e = report["exchange"][label]
        for kind in ("in_sample", "leave_one_out", "oracle"):
            s = e[kind]
            print(
                f"{label:8s} {kind:14s} N={s['N']} within x2 {s['within_factor_2']:3d} "
                f"sign {s['sign_ok']:3d} median ratio {s['median_abs_ratio']:.2f} "
                f"MAE {s['MAE_meV']:.1f} no-solution {s['no_physical_solution']}"
            )
        print(f"{label:8s} U by class (eV): {e['U_by_class_eV']}")
    print("\nleave-one-out by class: within x2 / sign / N   (base -> trained)")
    for cls in sorted(report["exchange"]["base"]["loo_by_class"]):
        sb = report["exchange"]["base"]["loo_by_class"][cls]
        st = report["exchange"]["trained"]["loo_by_class"].get(cls, {})
        print(
            f"   {cls:13s} {sb['within_factor_2']:3d}/{sb['sign_ok']:3d}/{sb['N']:3d}  ->  "
            f"{st.get('within_factor_2', 0):3d}/{st.get('sign_ok', 0):3d}/{st.get('N', 0):3d}"
        )
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
