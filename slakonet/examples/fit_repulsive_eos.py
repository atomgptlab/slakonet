"""Refit repulsive splines against equations of state, gaps untouched.

The repulsive never enters an eigenvalue, so scaling it cannot move a
band gap by even a millivolt. That makes it the right and only safe
instrument for repairing the equation of state after a fit that *did*
touch H -- ``fit_onsite_gaps.py`` improves gaps but leaves the
bulk-modulus bias worse, and this is the stage that pays that back.

The trick that makes it cheap: the repulsive is a sum of independent
pair terms, so for one material

    E(V; s) = E_band(V) + sum_p  s_p * R_p(V)

with E_band and every R_p(V) collected in a single E(V) scan
(``_compute_repulsive_energy(by_pair=True)``). After that one pass the
total energy for *any* per-pair scaling is arithmetic -- no eigenvalue
problem is solved again -- so optimising the scale factors costs
seconds rather than GPU-hours.

Fitted against JARVIS-DFT ``bulk_modulus_kv`` and the DFT equilibrium
volume, with a held-out split and a ridge pulling every scale toward 1.

    python slakonet/examples/fit_repulsive_eos.py \\
        --model slakonet_v2 --out slakonet_v2r

Caveat kept in view: this E(V) scan holds fractional coordinates fixed,
which cannot soften layered or framework structures (MoS2 comes out at
178 GPa frozen against 74 relaxed, reference 70). Materials whose
predicted B0 sits far *above* reference are therefore dropped by
``--max-overshoot`` -- fitting them would fit the scan protocol rather
than the parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.utils import get_cache_dir
from jarvis.db.figshare import data as jarvis_data
from scipy.optimize import minimize

from slakonet.ase_calc import SlaKoNetCalculator
from slakonet.atoms import Geometry
from slakonet.examples.chipstb_eval import (
    CHIPS_TB_JIDS,
    birch_murnaghan_b0,
)
from slakonet.main import SimpleDftb
from slakonet.optim import default_model


def collect(model, calc, entry, scales, kspacing, device):
    """One E(V) scan -> (volumes, E_band, {pair: R_p(V)}).

    E_band is the piece the repulsive scaling cannot touch, so it is
    stored once and reused for every trial scaling afterwards.
    """
    base = Atoms.from_dict(entry["atoms"]).ase_converter()
    cell0 = base.get_cell()
    vols, e_band, rep = [], [], {}
    for i, s in enumerate(scales):
        at = base.copy()
        at.set_cell(cell0 * s, scale_atoms=True)
        at.calc = calc
        e_tot = at.get_potential_energy()

        geo = Geometry.from_ase_atoms([at])
        sim = SimpleDftb(
            geo,
            model,
            kpoints=torch.tensor(calc.kpoints_for(at)),
            device=device,
            with_eigenvectors=False,
            compute_forces=False,
        )
        by_pair = sim._compute_repulsive_energy(by_pair=True)
        by_pair = {k: float(v) for k, v in by_pair.items()}

        for k in by_pair:
            rep.setdefault(k, [0.0] * len(scales))
        for k, v in by_pair.items():
            rep[k][i] = v
        vols.append(at.get_volume())
        e_band.append(e_tot - sum(by_pair.values()))
    return (
        np.array(vols),
        np.array(e_band),
        {k: np.array(v) for k, v in rep.items()},
    )


def predict(rec, s_of):
    """(V0, B0) for this material under the scaling `s_of(pair)`."""
    e = rec["e_band"].copy()
    for pair, r in rec["rep"].items():
        e = e + s_of(pair) * r
    return birch_murnaghan_b0(rec["vols"], e)


def objective(x, recs, pairs, ridge, w_vol):
    idx = {p: i for i, p in enumerate(pairs)}
    total = 0.0
    for rec in recs:
        v0, b0 = predict(rec, lambda p: x[idx[p]] if p in idx else 1.0)
        if np.isnan(b0) or b0 <= 0:
            # A scaling that destroys the minimum is worse than any
            # finite error; penalise rather than silently skipping it.
            total += 4.0
            continue
        total += ((b0 - rec["b0_ref"]) / rec["b0_ref"]) ** 2
        total += w_vol * ((v0 - rec["v0_ref"]) / rec["v0_ref"]) ** 2
    total /= max(len(recs), 1)
    return total + ridge * float(((x - 1.0) ** 2).sum())


def apply_scales(model, scales):
    """Multiply each pair's repulsive by its scale, in place.

    The exponential head is exp(-a r + b) + c, so scaling the whole term
    by s means b -> b + ln s and c -> s c; spline and tail coefficients
    scale directly.
    """
    for pair, s in scales.items():
        if abs(s - 1.0) < 1e-12 or pair not in model.skf_optimizers:
            continue
        opt = model.skf_optimizers[pair]
        rs = getattr(opt, "r_spline", None)
        if rs is None:
            continue
        exp_coef = torch.as_tensor(rs.exp_coef).clone().float()
        exp_coef[1] = exp_coef[1] + float(np.log(s))
        exp_coef[2] = exp_coef[2] * s
        opt.r_spline = type(rs)(
            grid=torch.as_tensor(rs.grid).clone().float(),
            cutoff=torch.as_tensor(rs.cutoff).clone().float(),
            spline_coef=torch.as_tensor(rs.spline_coef).clone().float() * s,
            exp_coef=exp_coef,
            tail_coef=torch.as_tensor(rs.tail_coef).clone().float() * s,
        )
        # Keep skf_dict in step: get_updated_skf prefers r_spline, but the
        # dict copy is what save_ultra_compact carries into skf_metadata.
        if isinstance(opt.skf_dict.get("r_spline"), dict):
            opt.skf_dict["r_spline"] = {
                "grid": opt.r_spline.grid,
                "cutoff": opt.r_spline.cutoff,
                "spline_coef": opt.r_spline.spline_coef,
                "exp_coef": opt.r_spline.exp_coef,
                "tail_coef": opt.r_spline.tail_coef,
            }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="slakonet_v2")
    ap.add_argument("--out", default="slakonet_v2r")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kspacing", type=float, default=0.3)
    ap.add_argument("--ridge", type=float, default=0.02)
    ap.add_argument(
        "--w-vol",
        type=float,
        default=0.5,
        help="weight on the equilibrium-volume term relative to B0",
    )
    ap.add_argument(
        "--max-overshoot",
        type=float,
        default=1.8,
        help="drop a material only if its frozen scan is suspect (it has "
        "free internal coordinates) AND its B0 exceeds this multiple of "
        "the reference. Rigid structures are always kept: with no free "
        "coordinate to relax, the frozen scan is exact and an overshoot "
        "there is a real parameter error, not an artifact.",
    )
    ap.add_argument("--bounds", type=float, nargs=2, default=(0.5, 1.6))
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
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
    # Separate probe calculator: forces are needed once per material to
    # decide whether the frozen scan is trustworthy, but computing them
    # throughout the scan would slow every energy call.
    probe_calc = SlaKoNetCalculator(
        model,
        kspacing=args.kspacing,
        alpha=1.0,
        device=args.device,
        compute_forces=True,
        compute_stress=False,
    )
    index = {row["jid"]: row for row in jarvis_data("dft_3d")}
    scan = np.linspace(0.90, 1.14, 9)

    jids = CHIPS_TB_JIDS[: args.limit] if args.limit else CHIPS_TB_JIDS
    recs, dropped = [], []
    for n, jnum in enumerate(jids, 1):
        entry = index.get(f"JVASP-{jnum}")
        if entry is None:
            continue
        try:
            b0_ref = float(entry.get("bulk_modulus_kv"))
        except (TypeError, ValueError):
            continue
        if np.isnan(b0_ref) or b0_ref <= 0:
            continue
        t1 = time.time()
        vols, e_band, rep = collect(
            model, calc, entry, scan, args.kspacing, args.device
        )
        base = Atoms.from_dict(entry["atoms"]).ase_converter()
        rec = {
            "jid": f"JVASP-{jnum}",
            "formula": base.get_chemical_formula(),
            "vols": vols,
            "e_band": e_band,
            "rep": rep,
            "b0_ref": b0_ref,
            "v0_ref": base.get_volume(),
        }
        _, b0_now = predict(rec, lambda p: 1.0)
        rec["b0_now"] = b0_now

        # Uniform scaling preserves the space group, so vanishing forces
        # at the reference mean the frozen scan is exact at every volume.
        probe = base.copy()
        probe.calc = probe_calc
        fmax0 = float(np.abs(probe.get_forces()).max())
        rigid = fmax0 < 0.05

        if np.isnan(b0_now):
            keep, why = False, "no minimum"
        elif rigid:
            keep, why = True, "rigid"
        elif b0_now > args.max_overshoot * b0_ref:
            keep, why = False, "soft-DOF overshoot"
        else:
            keep, why = True, "relaxable"
        rec["kept"] = bool(keep)
        print(
            f"  [{n:3d}] {rec['formula']:10s} B0 now "
            f"{b0_now if not np.isnan(b0_now) else float('nan'):7.1f} "
            f"ref {b0_ref:7.1f}  fmax {fmax0:6.3f}  pairs {len(rep):2d}  "
            f"{'keep' if keep else 'DROP'} ({why})  "
            f"{time.time() - t1:.0f}s",
            flush=True,
        )
        if keep:
            recs.append(rec)
        else:
            dropped.append(rec["formula"])

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(recs))
    n_test = int(round(args.test_frac * len(recs)))
    test = [recs[i] for i in order[:n_test]]
    train = [recs[i] for i in order[n_test:]]
    pairs = sorted({p for r in train for p in r["rep"]})
    print(
        f"\n[*] {len(recs)} materials kept "
        f"({len(train)} train / {len(test)} test), "
        f"{len(pairs)} repulsive scales to fit"
    )

    def report(tag, x):
        idx = {p: i for i, p in enumerate(pairs)}

        def sof(p):
            return x[idx[p]] if p in idx else 1.0

        out = []
        for subset, name in ((train, "train"), (test, "test")):
            errs = []
            for rec in subset:
                _, b0 = predict(rec, sof)
                if not np.isnan(b0):
                    errs.append(abs(b0 - rec["b0_ref"]))
            out.append(np.mean(errs) if errs else float("nan"))
        print(
            f"    {tag}: train B0 MAE {out[0]:6.1f} | test {out[1]:6.1f} GPa"
        )
        return out

    x0 = np.ones(len(pairs))
    report("baseline", x0)
    res = minimize(
        objective,
        x0,
        args=(train, pairs, args.ridge, args.w_vol),
        method="L-BFGS-B",
        bounds=[tuple(args.bounds)] * len(pairs),
        options={"maxiter": 400},
    )
    print(f"[*] optimiser: {res.message} ({res.nit} iterations)")
    mae_tr, mae_te = report("fitted", res.x)

    scales = {p: float(v) for p, v in zip(pairs, res.x)}
    apply_scales(model, scales)
    out_dir = os.path.join(get_cache_dir("slakonet"), args.out)
    os.makedirs(out_dir, exist_ok=True)
    out_pt = os.path.join(out_dir, f"{args.out}.pt")
    model.save_ultra_compact(out_pt)
    with open(
        os.path.join(out_dir, f"{args.out}.repulsive_scales.json"), "w"
    ) as f:
        json.dump(
            {
                "parent_model": args.model,
                "fit": {
                    "train": len(train),
                    "test": len(test),
                    "ridge": args.ridge,
                    "w_vol": args.w_vol,
                    "max_overshoot": args.max_overshoot,
                    "train_b0_mae_GPa": mae_tr,
                    "test_b0_mae_GPa": mae_te,
                    "dropped": dropped,
                },
                "scales": scales,
            },
            f,
            indent=2,
        )
    print(f"\nSaved {out_pt}\nTotal {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
