"""Score a SlaKoNet parameter set on band gaps and bulk moduli.

Evaluates a parameter set over the ChIPS-TB material list against
JARVIS-DFT references, reporting the two numbers that matter for
benchmarking a tight-binding set:

* **band gap** -- from a Monkhorst-Pack solve, against ``mbj_bandgap``
* **bulk modulus** -- from an E(V) scan fit to a 3rd-order
  Birch-Murnaghan equation of state, against ``bulk_modulus_kv``

The two probe different halves of the model and are worth reading
separately: eigenvalues (hence gaps) depend only on H/S, while the EOS
also feels the repulsive. A change that moves one and not the other is
usually a sign the fit is doing what it claims.

    python slakonet/examples/chipstb_eval.py --model slakonet_v1a_full
    python slakonet/examples/chipstb_eval.py --model slakonet_v1a --no-eos

Writes ``chipstb_eval_<model>.csv`` and prints a summary table.
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import torch
from ase.optimize import FIRE
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jarvis_data

from slakonet.ase_calc import SlaKoNetCalculator
from slakonet.optim import default_model

GPA = 160.2176634

# ChIPS-TB benchmark materials (JARVIS-DFT ids).
CHIPS_TB_JIDS = [
    1174,
    1002,
    1195,
    8118,
    8158,
    107,
    1327,
    91,
    41,
    104,
    113,
    1145,
    116,
    1180,
    1183,
    1189,
    1198,
    1201,
    1267,
    1294,
    1300,
    1312,
    1315,
    1393,
    1408,
    1453,
    17,
    1702,
    1954,
    23,
    299,
    30,
    32,
    39,
    5,
    54,
    57,
    7630,
    7678,
    7762,
    7844,
    7860,
    8003,
    8169,
    8566,
    8583,
    890,
    95,
    96,
    97,
    816,
    867,
    1029,
    825,
    34674,
]


def birch_murnaghan_b0(volumes, energies):
    """Fit E(V) to BM3 and return (V0 in A^3, B0 in GPa).

    Returns (nan, nan) if the scan does not bracket a minimum -- an
    extrapolated B0 from a monotonic scan is meaningless and reporting
    it as a number would quietly corrupt the MAE.
    """
    v = np.asarray(volumes, dtype=float)
    e = np.asarray(energies, dtype=float)
    if len(v) < 5 or np.argmin(e) in (0, len(e) - 1):
        return float("nan"), float("nan")

    # BM3 is a cubic polynomial in x = V^(-2/3); fit that, then take the
    # derivatives analytically at the minimum.
    x = v ** (-2.0 / 3.0)
    c = np.polyfit(x, e, 3)
    dp = np.polyder(c)
    roots = [r.real for r in np.roots(dp) if abs(r.imag) < 1e-8 and r.real > 0]
    if not roots:
        return float("nan"), float("nan")
    # Pick the root that is a minimum and lies inside the scanned range.
    ddp = np.polyder(dp)
    cand = [
        r for r in roots if np.polyval(ddp, r) > 0 and x.min() <= r <= x.max()
    ]
    if not cand:
        return float("nan"), float("nan")
    x0 = cand[0]
    v0 = x0 ** (-3.0 / 2.0)

    # B0 = V d2E/dV2 at V0.  dx/dV = -2/3 V^(-5/3)
    dxdv = -2.0 / 3.0 * v0 ** (-5.0 / 3.0)
    d2xdv2 = 10.0 / 9.0 * v0 ** (-8.0 / 3.0)
    d2edv2 = np.polyval(ddp, x0) * dxdv**2 + np.polyval(dp, x0) * d2xdv2
    return v0, v0 * d2edv2 * GPA


def evaluate(
    model_name,
    jids,
    do_eos=True,
    kpoints=(3, 3, 3),
    kspacing=0.2,
    device=None,
    scales=None,
    limit=None,
    relax=False,
    fmax=0.05,
    relax_steps=40,
):
    scales = scales or np.linspace(0.90, 1.14, 9)
    model = default_model(model_name=model_name).float()
    calc = SlaKoNetCalculator(
        model,
        kpoints=list(kpoints),
        kspacing=kspacing,
        alpha=1.0,
        device=device,
        compute_forces=relax,
        compute_stress=False,
    )
    print(f"[*] k-mesh: {'kspacing=%s' % kspacing if kspacing else kpoints}")
    if relax:
        print(
            "[*] EOS relaxes internal coordinates at each volume "
            f"(fmax={fmax}, <={relax_steps} steps)"
        )

    index = {row["jid"]: row for row in jarvis_data("dft_3d")}
    rows = []
    for n, jnum in enumerate(jids[:limit] if limit else jids, 1):
        jid = f"JVASP-{jnum}"
        entry = index.get(jid)
        if entry is None:
            continue
        ref_gap = entry.get("mbj_bandgap")
        ref_b0 = entry.get("bulk_modulus_kv")
        ref_gap = float(ref_gap) if _num(ref_gap) else float("nan")
        ref_b0 = float(ref_b0) if _num(ref_b0) else float("nan")

        base = Atoms.from_dict(entry["atoms"]).ase_converter()
        rec = {
            "jid": jid,
            "formula": base.get_chemical_formula(),
            "natoms": len(base),
            "ref_gap": ref_gap,
            "ref_b0": ref_b0,
        }
        t0 = time.perf_counter()
        try:
            at = base.copy()
            at.calc = calc
            at.get_potential_energy()
            gap = calc.get_bandgap()
            rec["gap"] = float("nan") if gap is None else float(gap)
        except Exception as exc:
            rec["gap"] = float("nan")
            rec["error"] = f"gap: {type(exc).__name__}: {exc}"

        if do_eos:
            vols, enes = [], []
            try:
                cell0 = base.get_cell()
                # Uniform scaling preserves the space group, so if the
                # forces vanish by symmetry at the reference geometry they
                # vanish at every scaled volume too -- rocksalt, zincblende
                # and fcc metals have no free internal coordinate to relax.
                # Checking once turns relaxation from unaffordable into a
                # cost paid only where it can change anything.
                needs_relax = False
                if relax:
                    probe = base.copy()
                    probe.calc = calc
                    fmax0 = np.abs(probe.get_forces()).max()
                    needs_relax = fmax0 > fmax
                    rec["fmax_ref"] = round(float(fmax0), 4)
                    rec["relaxed"] = bool(needs_relax)
                for s in scales:
                    at = base.copy()
                    at.set_cell(cell0 * s, scale_atoms=True)
                    at.calc = calc
                    if needs_relax:
                        FIRE(at, logfile=None).run(
                            fmax=fmax, steps=relax_steps
                        )
                    enes.append(at.get_potential_energy())
                    vols.append(at.get_volume())
                v0, b0 = birch_murnaghan_b0(vols, enes)
                rec["v0"], rec["b0"] = v0, b0
                rec["v0_ref"] = base.get_volume()
            except Exception as exc:
                rec["v0"] = rec["b0"] = float("nan")
                rec["error"] = f"eos: {type(exc).__name__}: {exc}"

        rec["seconds"] = round(time.perf_counter() - t0, 1)
        rows.append(rec)
        print(
            f"  [{n:3d}] {jid:14s} {rec['formula']:12s} "
            f"gap {_f(rec.get('gap'))} (ref {_f(ref_gap)})  "
            f"B0 {_f(rec.get('b0'), 1)} (ref {_f(ref_b0, 1)})  "
            f"{rec['seconds']}s",
            flush=True,
        )
    return rows


def _num(x):
    try:
        return not np.isnan(float(x))
    except (TypeError, ValueError):
        return False


def _f(x, nd=3):
    return "   n/a" if x is None or np.isnan(x) else f"{x:6.{nd}f}"


def summarize(rows, label):
    def mae(pred_key, ref_key):
        pairs = [
            (r[pred_key], r[ref_key])
            for r in rows
            if _num(r.get(pred_key)) and _num(r.get(ref_key))
        ]
        if not pairs:
            return float("nan"), 0
        p, q = np.array(pairs).T
        return float(np.abs(p - q).mean()), len(pairs)

    gap_mae, n_gap = mae("gap", "ref_gap")
    b0_mae, n_b0 = mae("b0", "ref_b0")
    vol_mae, n_vol = mae("v0", "v0_ref")
    print(
        f"\n=== {label} ===\n"
        f"  band gap MAE vs MBJ      : {gap_mae:8.3f} eV   (n={n_gap})\n"
        f"  bulk modulus MAE vs DFT  : {b0_mae:8.1f} GPa  (n={n_b0})\n"
        f"  equilibrium volume MAE   : {vol_mae:8.2f} A^3  (n={n_vol})"
    )
    return {"gap_mae": gap_mae, "b0_mae": b0_mae, "vol_mae": vol_mae}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="parameter set name")
    ap.add_argument("--no-eos", action="store_true", help="gaps only")
    ap.add_argument("--device", default=None, help="cuda / cpu")
    ap.add_argument(
        "--kspacing",
        type=float,
        default=0.2,
        help="reciprocal-space mesh density (A^-1). A fixed 3x3x3 grid "
        "gives fcc Al a spurious 1.5 eV gap; 0.2 is converged.",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--relax",
        action="store_true",
        help="relax internal coordinates at each EOS volume. Frozen "
        "coordinates cannot soften layered/framework structures: "
        "MoS2 comes out at 178 GPa frozen vs 74 relaxed (ref 70).",
    )
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--relax-steps", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    name = args.model or "default"
    rows = evaluate(
        args.model,
        CHIPS_TB_JIDS,
        do_eos=not args.no_eos,
        kspacing=args.kspacing,
        device=args.device,
        limit=args.limit,
        relax=args.relax,
        fmax=args.fmax,
        relax_steps=args.relax_steps,
    )
    summarize(rows, name)

    out = args.out or f"chipstb_eval_{name}.csv"
    fields = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
