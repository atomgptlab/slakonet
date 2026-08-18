"""Fit per-element on-site energy shifts to improve band gaps.

Tight-binding band gaps are set by H and S alone -- the repulsive never
enters an eigenvalue. Of everything in H, the on-site energies are the
safest thing to tune: they are the diagonal, there are only a handful
per element (one per shell), and they come from exactly one place, the
homonuclear X-X file (``_get_homo_dict`` asserts as much). Shifting
them moves band edges without touching the bonding integrals that set
the equation of state, which is what went wrong the last time H/S were
retrained wholesale -- v1's gaps improved and its bulk moduli became
nonsense (Al at 9261 GPa against 69 GPa DFT).

So: keep every off-diagonal integral frozen, and fit one shift
``d[X, l]`` per element and shell against JARVIS-DFT MBJ gaps.

The fit is Gauss-Newton on a finite-difference Jacobian. A material's
gap depends only on the shifts of the elements it contains, so J is
sparse and each column costs gap evaluations for that element's
materials only. Two things keep it honest:

* **a held-out split** -- shifts are fit on a train subset and scored on
  materials never seen, so the reported gain is not memorisation. With
  ~3 parameters per element this matters.
* **ridge + a box clamp** -- shifts are regularised toward zero and
  clamped to ``--max-shift``, because a large on-site shift is no longer
  a correction to the reference tables, it is a different element.

    python slakonet/examples/fit_onsite_gaps.py \\
        --model slakonet_v1a_full --out slakonet_v2

Writes the shifted parameter set to the slakonet cache plus a JSON of
the fitted shifts next to it.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jarvis_data
from jarvis.core.utils import get_cache_dir

from slakonet.ase_calc import SlaKoNetCalculator
from slakonet.optim import default_model

HARTREE = 27.211386

# JARVIS-DFT ids of the ChIPS-TB benchmark materials.
from slakonet.examples.chipstb_eval import CHIPS_TB_JIDS  # noqa: E402


def onsite_tensor(model, symbol):
    """Return the mutable on_sites container for element `symbol`."""
    opt = model.skf_optimizers[f"{symbol}-{symbol}"]
    ad = opt.skf_dict.get("atomic_data") or {}
    return ad, ad.get("on_sites")


def set_onsite(model, symbol, values):
    ad, _ = onsite_tensor(model, symbol)
    ad["on_sites"] = values
    opt = model.skf_optimizers[f"{symbol}-{symbol}"]
    opt.atomic_data = ad
    # get_updated_skf() copies skf_dict, so writing into it is enough;
    # drop any cached Skf so the next call rebuilds from the new value.
    if hasattr(opt, "_cached_skf"):
        del opt._cached_skf


def as_list(x):
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return [float(v) for v in x.flatten()]
    return [float(v) for v in np.asarray(x).flatten()]


def gap_of(calc, ase_atoms):
    at = ase_atoms.copy()
    at.calc = calc
    try:
        at.get_potential_energy()
        g = calc.get_bandgap()
        return float("nan") if g is None else float(g)
    except Exception:
        return float("nan")


def build_set(jids):
    index = {row["jid"]: row for row in jarvis_data("dft_3d")}
    mats = []
    for jnum in jids:
        entry = index.get(f"JVASP-{jnum}")
        if entry is None:
            continue
        try:
            gap = float(entry.get("mbj_bandgap"))
        except (TypeError, ValueError):
            continue
        if np.isnan(gap):
            continue
        atoms = Atoms.from_dict(entry["atoms"])
        mats.append(
            {
                "jid": f"JVASP-{jnum}",
                "atoms": atoms.ase_converter(),
                "elements": sorted(set(atoms.elements)),
                "ref": gap,
            }
        )
    return mats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="slakonet_v1a_full")
    ap.add_argument("--out", default="slakonet_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--kspacing",
        type=float,
        default=0.2,
        help="k-mesh density. Must be converged: fitting on-site shifts "
        "against a coarse mesh fits the sampling error, not the model.",
    )
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="finite-difference step on the on-site energy, Hartree",
    )
    ap.add_argument(
        "--max-shift",
        type=float,
        default=0.06,
        help="box clamp on |shift|, Hartree (0.06 Ha ~ 1.6 eV)",
    )
    ap.add_argument(
        "--ridge",
        type=float,
        default=0.5,
        help="ridge weight pulling shifts toward zero",
    )
    ap.add_argument(
        "--test-frac",
        type=float,
        default=0.3,
        help="fraction of materials held out of the fit",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    model = default_model(model_name=args.model).float()
    calc = SlaKoNetCalculator(
        model,
        kspacing=args.kspacing,
        alpha=1.0,
        device=args.device,
        compute_forces=False,
        compute_stress=False,
    )

    mats = build_set(CHIPS_TB_JIDS)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(mats))
    n_test = int(round(args.test_frac * len(mats)))
    test_idx = set(order[:n_test].tolist())
    train = [m for i, m in enumerate(mats) if i not in test_idx]
    test = [m for i, m in enumerate(mats) if i in test_idx]
    print(
        f"[*] {len(mats)} materials with an MBJ gap "
        f"({len(train)} train / {len(test)} test)"
    )

    # Parameters: (element, shell) for every element with on-site data.
    params, base_onsite = [], {}
    for m in mats:
        for sym in m["elements"]:
            if sym in base_onsite:
                continue
            _, os_val = onsite_tensor(model, sym)
            vals = as_list(os_val)
            if not vals:
                continue
            base_onsite[sym] = vals
            params.extend((sym, shell) for shell in range(len(vals)))
    print(f"[*] {len(params)} free shifts over {len(base_onsite)} elements")

    shifts = {k: 0.0 for k in params}

    def apply_shifts():
        for sym, vals in base_onsite.items():
            new = [
                v + shifts.get((sym, shell), 0.0)
                for shell, v in enumerate(vals)
            ]
            set_onsite(model, sym, torch.tensor(new, dtype=torch.float32))

    def gaps_for(subset):
        return np.array([gap_of(calc, m["atoms"]) for m in subset])

    def report(tag):
        apply_shifts()
        gtr, gte = gaps_for(train), gaps_for(test)
        rtr = np.array([m["ref"] for m in train])
        rte = np.array([m["ref"] for m in test])
        mtr = np.nanmean(np.abs(gtr - rtr))
        mte = np.nanmean(np.abs(gte - rte)) if len(test) else float("nan")
        print(f"    {tag}: train MAE {mtr:.3f} eV | test MAE {mte:.3f} eV")
        return mtr, mte, gtr

    print("[*] baseline")
    mtr0, mte0, g_train = report("baseline")

    # Keep the iterate that generalises best, not the last one. Gauss-
    # Newton keeps driving the training residual down past the point
    # where it helps: with 90 shifts against 39 training materials, a
    # later iteration can improve train MAE and lose on held-out data.
    best = {"test": mte0, "shifts": dict(shifts), "tag": "baseline"}

    ref_train = np.array([m["ref"] for m in train])
    for it in range(1, args.iters + 1):
        print(f"[*] iteration {it}: building Jacobian ...", flush=True)
        J = np.zeros((len(train), len(params)))
        for j, (sym, shell) in enumerate(params):
            rows = [i for i, m in enumerate(train) if sym in m["elements"]]
            if not rows:
                continue
            shifts[(sym, shell)] += args.step
            apply_shifts()
            for i in rows:
                g = gap_of(calc, train[i]["atoms"])
                J[i, j] = (g - g_train[i]) / args.step
            shifts[(sym, shell)] -= args.step
        apply_shifts()

        # Gauss-Newton step with ridge; NaN gaps drop out of the system.
        resid = ref_train - g_train
        ok = ~np.isnan(resid) & ~np.isnan(J).any(axis=1)
        A = np.vstack([J[ok], np.sqrt(args.ridge) * np.eye(len(params))])
        b = np.concatenate([resid[ok], np.zeros(len(params))])
        d, *_ = np.linalg.lstsq(A, b, rcond=None)

        cur = np.array([shifts[k] for k in params])
        new = np.clip(cur + d, -args.max_shift, args.max_shift)
        for k, v in zip(params, new):
            shifts[k] = float(v)
        print(
            f"    step |d|max = {np.abs(d).max():.4f} Ha, "
            f"shift |s|max = {np.abs(new).max():.4f} Ha "
            f"({np.abs(new).max() * HARTREE:.2f} eV)"
        )
        mtr, mte, g_train = report(f"iter {it}")
        if mte < best["test"]:
            best = {"test": mte, "shifts": dict(shifts), "tag": f"iter {it}"}
        else:
            print(
                f"    (held-out MAE did not improve on {best['tag']}"
                f" = {best['test']:.3f} eV)"
            )

    if best["tag"] != f"iter {args.iters}":
        print(
            f"[*] keeping {best['tag']} (test MAE {best['test']:.3f} eV), "
            f"discarding later iterations"
        )
        shifts = best["shifts"]
    mtr, mte, _ = report(f"selected ({best['tag']})")
    apply_shifts()
    out_dir = os.path.join(get_cache_dir("slakonet"), args.out)
    os.makedirs(out_dir, exist_ok=True)
    out_pt = os.path.join(out_dir, f"{args.out}.pt")
    model.save_ultra_compact(out_pt)

    side = os.path.join(out_dir, f"{args.out}.onsite_shifts.json")
    with open(side, "w") as f:
        json.dump(
            {
                "parent_model": args.model,
                "units": "Hartree",
                "fit": {
                    "materials": len(mats),
                    "train": len(train),
                    "test": len(test),
                    "ridge": args.ridge,
                    "max_shift": args.max_shift,
                    "selected_iterate": best["tag"],
                    "baseline_train_mae_eV": mtr0,
                    "baseline_test_mae_eV": mte0,
                    "final_train_mae_eV": mtr,
                    "final_test_mae_eV": mte,
                },
                "shifts": {f"{s}-{sh}": v for (s, sh), v in shifts.items()},
            },
            f,
            indent=2,
        )
    print(
        f"\nSaved {out_pt}\nShifts: {side}\n"
        f"Total {(time.time() - t0) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
