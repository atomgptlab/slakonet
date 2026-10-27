"""Fit a per-element Stoner parameter table I_d(Z) to experimental exchange
constants, so that slakonet -> TB2J needs no user-chosen parameter.

Stages (each resumable; results cached under --work):

    dataset     ExpMagDB compounds with experimental J, a 3d magnetic
                species, all elements in slakonet and <= --max-atoms atoms in
                the reference magnetic cell. Reference collinear order:
                MAGNDATA moments (MAGMOM_exp) or, for the INS set, the ground
                state of the INS Hamiltonian (results_ins/<id>/result.json).
    precompute  For every compound and every I_d of the grid: converge the
                slakonet spin model in the reference order, run TB2J, store
                the shell-averaged J. Runs with --nproc workers (1 thread
                each); start it under `nice`.
    fit         Per-element I_d minimising a robust relative error over all
                couplings (each compound weighted equally), plus
                leave-one-compound-out cross-validation, compared with the
                atomic Stoner values. A grid point counts only if the SCF
                converged to a physical moment, 0.3 m_ref <= |m| <= 1.3 m_ref
                (m_ref: 2S of the INS set or the MAGNDATA moment); otherwise
                the compound has no physical solution at that I and gets the
                maximum loss.

Only experimental J enter the fit.

Conventions: ExpMagDB JS2 = J S^2 per bond (J > 0 AFM) = -2 J_TB2J.

Usage
-----
    python fit_stoner_table.py dataset    --db .../Magnetic_MLIP --work stoner_fit
    nice -n 19 python fit_stoner_table.py precompute --work stoner_fit --nproc 4
    python fit_stoner_table.py fit        --work stoner_fit
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

I_GRID = [0.020, 0.030, 0.045, 0.065, 0.090, 0.120, 0.160, 0.210]  # Hartree
# +U model (slakonet.hubbard_spin: atomic Stoner + U + SCC): U_eff grid, Ha
U_GRID = [
    u / 27.211386
    for u in (1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 12.5, 15.0)
]
MODELS = {"stoner": ("cache", I_GRID), "hubbard": ("cache_hubbard", U_GRID)}
MAGNETIC_3D = ["V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
K_DENSITY = 24.0  # k-points per direction ~ K_DENSITY / |a_i| (Angstrom)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
def _reference_order(db_root, cid):
    """(ase.Atoms, signed initial moments, source) or None."""
    from ase.io import read

    entry = os.path.join(db_root, "expdb", "entries", cid)
    magmom_file = os.path.join(entry, "MAGMOM_exp")
    if os.path.exists(magmom_file):
        atoms = read(os.path.join(entry, "POSCAR"))
        with open(magmom_file) as fh:
            line = next(l for l in fh if l.strip().startswith("MAGMOM"))
        m = np.array([float(x) for x in line.split("=")[1].split()])
        if len(m) != len(atoms):
            return None
        return atoms, (2.0 * np.sign(m)).tolist(), "MAGNDATA MAGMOM_exp"
    result = os.path.join(db_root, "results_ins", cid, "result.json")
    prim_file = os.path.join(db_root, "results_ins", cid, "prim_mp.extxyz")
    if os.path.exists(result) and os.path.exists(prim_file):
        with open(result) as fh:
            gs = json.load(fh)["order"]["ref_gs"]
        atoms = read(prim_file).repeat(gs["reps"])
        return (
            atoms,
            (2.0 * np.asarray(gs["sigma"])).tolist(),
            "INS ground state",
        )
    return None


def _magndata_moments(db_root, cid):
    with open(
        os.path.join(db_root, "expdb", "entries", cid, "MAGMOM_exp")
    ) as fh:
        line = next(l for l in fh if l.strip().startswith("MAGMOM"))
    return [float(x) for x in line.split("=")[1].split()]


def build_dataset(db_root, work, max_atoms):
    import pandas as pd
    from ase.data import atomic_numbers
    from slakonet.optim import default_model

    # elements with a homonuclear SKF pair in the model (shell_dict alone
    # lists Z = 1-99, including elements without parameters, e.g. Lu)
    skfs = default_model().get_updated_skfs()
    covered = {
        atomic_numbers[k.split("-")[0]]
        for k in skfs
        if k.split("-")[0] == k.split("-")[1]
    }
    exp = pd.read_csv(os.path.join(db_root, "expdb", "expdb.csv")).set_index(
        "id"
    )
    J = pd.read_csv(os.path.join(db_root, "expdb", "expdb_J.csv"))
    J = J[
        (J.kind == "exp")
        & J.JS2_meV.notna()
        & J.flag.isna()
        & J.distance_A.notna()
    ]

    dataset, skipped = [], {}
    for cid in sorted(set(J.id)):
        species = str(exp.loc[cid, "magnetic_species"]).split()
        if not species or any(s not in MAGNETIC_3D for s in species):
            skipped[cid] = f"magnetic species {species} not 3d"
            continue
        ref = _reference_order(db_root, cid)
        if ref is None:
            skipped[cid] = "no reference collinear order"
            continue
        atoms, m0, source = ref
        if len(atoms) > max_atoms:
            skipped[cid] = f"{len(atoms)} atoms > {max_atoms}"
            continue
        if any(int(z) not in covered for z in set(atoms.numbers)):
            skipped[cid] = "element outside slakonet"
            continue
        if not any(
            abs(m) > 0
            for m, s in zip(m0, atoms.get_chemical_symbols())
            if s in species
        ):
            skipped[cid] = (
                "reference order has no moment on the magnetic species"
            )
            continue
        rows = J[J.id == cid]
        # reference moment: spin-only 2S of the INS set, else the
        # experimental MAGNDATA moments of the magnetic species
        S = exp.loc[cid, "S"]
        if pd.notna(S):
            m_ref = 2.0 * float(S)
        else:
            m_ref = float(
                np.mean(
                    [
                        abs(m)
                        for m, s in zip(
                            _magndata_moments(db_root, cid),
                            atoms.get_chemical_symbols(),
                        )
                        if s in species and m
                    ]
                )
            )
        dataset.append(
            {
                "id": cid,
                "species": species,
                "order_source": source,
                "symbols": atoms.get_chemical_symbols(),
                "cell": atoms.cell.array.tolist(),
                "positions": atoms.positions.tolist(),
                "initial_moments": m0,
                "m_ref": m_ref,
                "couplings": [
                    {
                        "shell": r.shell,
                        "pair": r.pair,
                        "distance": float(r.distance_A),
                        "JS2": float(r.JS2_meV),
                        "reference": r.reference,
                    }
                    for r in rows.itertuples()
                ],
            }
        )
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "dataset.json"), "w") as fh:
        json.dump(
            {"compounds": dataset, "skipped": skipped, "I_grid": I_GRID},
            fh,
            indent=1,
        )
    single = sum(len(c["species"]) == 1 for c in dataset)
    print(
        f"{len(dataset)} compounds ({single} single magnetic species), "
        f"{sum(len(c['couplings']) for c in dataset)} couplings; skipped {len(skipped)}"
    )


# ---------------------------------------------------------------------------
# precompute
# ---------------------------------------------------------------------------
_MODEL = None


def _init_worker():
    import torch

    global _MODEL
    torch.set_num_threads(
        1
    )  # launch the parent under `nice`; workers inherit it
    from slakonet.optim import default_model

    from slakonet.spin_training import maybe_corrected

    _MODEL = maybe_corrected(default_model())


def _kmesh(atoms):
    return tuple(
        max(2, int(np.ceil(K_DENSITY / L))) for L in atoms.cell.lengths()
    )


def _run_task(task):
    """One (compound, I) point: spin model + TB2J, shell-averaged J."""
    import contextlib
    import io
    import pickle
    import shutil
    import tempfile

    from ase import Atoms
    from ase.data import atomic_numbers
    from slakonet.tb2j_interface import build_spin_model, gen_exchange_slakonet

    comp, I, out_file, spin_model_kind = task  # I: Stoner I or Hubbard U (Ha)
    atoms = Atoms(
        comp["symbols"],
        positions=comp["positions"],
        cell=comp["cell"],
        pbc=True,
    )
    stoner = {
        atomic_numbers[s]: {0: 0.0, 1: 0.0, 2: I} for s in comp["species"]
    }
    rcut = max(c["distance"] for c in comp["couplings"]) + 0.3
    km = _kmesh(atoms)
    t0 = time.time()
    record = {"id": comp["id"], "I": I, "kmesh": km}
    tmp = tempfile.mkdtemp(prefix=f"tb2j_{comp['id']}_")
    try:
        with contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(io.StringIO()):
            if spin_model_kind == "hubbard":
                from slakonet.hubbard_spin import build_hubbard_model

                sm = build_hubbard_model(
                    atoms,
                    _MODEL,
                    hubbard_U={s: I for s in comp["species"]},
                    initial_moments=comp["initial_moments"],
                    kmesh=km,
                    max_iter=300,
                    verbose=False,
                )
            else:
                sm = build_spin_model(
                    atoms,
                    _MODEL,
                    stoner_I=stoner,
                    initial_moments=comp["initial_moments"],
                    kmesh=km,
                    max_iter=300,
                    verbose=False,
                )
            if (
                not sm.converged and spin_model_kind == "stoner"
            ):  # retry, linear mixing
                sm = build_spin_model(
                    atoms,
                    _MODEL,
                    stoner_I=stoner,
                    initial_moments=comp["initial_moments"],
                    kmesh=km,
                    max_iter=600,
                    mixer="linear",
                    mixing=0.2,
                    verbose=False,
                )
            gen_exchange_slakonet(
                atoms,
                comp["species"],
                spin_model=sm,
                kmesh=km,
                nz=50,
                Rcut=rcut,
                output_path=tmp,
                verbose=False,
            )
        with open(os.path.join(tmp, "TB2J.pickle"), "rb") as fh:
            d = pickle.load(fh)
        symbols = atoms.get_chemical_symbols()
        spin_atom = {v: k for k, v in enumerate(d["index_spin"]) if v >= 0}
        bonds = []
        for (R, i, j), Jij in d["exchange_Jdict"].items():
            pair = "-".join(
                sorted((symbols[spin_atom[i]], symbols[spin_atom[j]]))
            )
            bonds.append(
                (
                    float(d["distance_dict"][(R, i, j)][1]),
                    pair,
                    float(Jij) * 1e3,
                )
            )
        mags = [
            abs(m) for m, s in zip(sm.moments, symbols) if s in comp["species"]
        ]
        record.update(
            {
                "converged": bool(sm.converged),
                "n_iter": sm.n_iter,
                "mean_abs_moment": float(np.mean(mags)),
                "bonds": bonds,
                "time_s": round(time.time() - t0, 1),
            }
        )
    except (
        Exception
    ) as err:  # keep the failure in the cache, visible in the fit
        record.update(
            {
                "error": f"{type(err).__name__}: {err}",
                "time_s": round(time.time() - t0, 1),
            }
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(out_file, "w") as fh:
        json.dump(record, fh)
    return record


def precompute(work, nproc, spin_model_kind="stoner"):
    import multiprocessing as mp

    with open(os.path.join(work, "dataset.json")) as fh:
        data = json.load(fh)
    cache_dir, grid = MODELS[spin_model_kind]
    cache = os.path.join(work, cache_dir)
    os.makedirs(cache, exist_ok=True)
    tasks = []
    # largest cells first so that the long tasks do not end up last
    for comp in sorted(data["compounds"], key=lambda c: -len(c["symbols"])):
        for I in grid:
            out = os.path.join(cache, f"{comp['id']}__I{I:.3f}.json")
            if not os.path.exists(out):
                tasks.append((comp, I, out, spin_model_kind))
    print(f"{len(tasks)} tasks to run with {nproc} workers", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(nproc, initializer=_init_worker) as pool:
        for k, rec in enumerate(pool.imap_unordered(_run_task, tasks), 1):
            status = rec.get("error") or (
                f"m={rec['mean_abs_moment']:.2f} conv={rec['converged']}"
            )
            print(
                f"[{k}/{len(tasks)}] {rec['id']:<14s} I={rec['I']:.3f} "
                f"{rec['time_s']:7.1f}s  {status}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------
def _annotate(data):
    """Add the automatic site class (environment, formal oxidation) of the
    magnetic species and, when the oxidation is known, the high-spin
    spin-only moment as m_ref (more robust than the S stored in the DB)."""
    from ase import Atoms
    from slakonet.magnetic_sites import site_details

    for comp in data["compounds"]:
        atoms = Atoms(
            comp["symbols"],
            positions=comp["positions"],
            cell=comp["cell"],
            pbc=True,
        )
        el = comp["species"][0]
        info = site_details(atoms, comp["symbols"].index(el))
        comp["env"], comp["ox"] = info["env"], info["ox"]
        from slakonet.magnetic_sites import site_anion_class

        comp["anion"] = site_anion_class(atoms, comp["symbols"].index(el))
        m_formal = info["m_expected"]
        comp["m_ref_db"] = comp["m_ref"]
        if m_formal:
            comp["m_ref"] = m_formal


GROUPINGS = {
    "global": lambda c: ("3d",),
    "env": lambda c: ("3d", c["env"]),
    "anion": lambda c: ("3d", c["anion"]),
    "element": lambda c: (c["species"][0],),
    "element_env": lambda c: (c["species"][0], c["env"]),
    "element_ox": lambda c: (c["species"][0], c["env"], c["ox"]),
}
MIN_GROUP = 3  # smaller groups fall back to the parent group


def _load(work, spin_model_kind="stoner"):
    with open(os.path.join(work, "dataset.json")) as fh:
        data = json.load(fh)
    _annotate(data)
    cache_dir, grid = MODELS[spin_model_kind]
    grid = np.array(grid)
    table = {}  # id -> list over grid of record
    for comp in data["compounds"]:
        recs = []
        for I in grid:
            f = os.path.join(work, cache_dir, f"{comp['id']}__I{I:.3f}.json")
            recs.append(json.load(open(f)) if os.path.exists(f) else None)
        table[comp["id"]] = recs
    return data, grid, table


def _match(comp, rec, coupling, tol=0.15):
    """Mean calculated JS2 (= -2 J_TB2J) of the bonds closest to a reference
    coupling (same species pair when the reference gives one)."""
    want = (
        coupling["pair"] if isinstance(coupling["pair"], str) else ""
    )  # NaN: unknown
    pair = (
        "-".join(sorted(want.split("-")))
        if want != "-" and "-" in want
        else None
    )
    cands = [(d, J) for d, p, J in rec["bonds"] if pair is None or p == pair]
    if not cands:
        return None
    dmin = min(abs(d - coupling["distance"]) for d, _ in cands)
    if dmin > tol:
        return None
    sel = [
        J for d, J in cands if abs(abs(d - coupling["distance"]) - dmin) < 0.02
    ]
    return -2.0 * float(np.mean(sel))


def _valid(rec, m_ref, lo=0.3, hi=1.3):
    """A grid point is usable when the spin-SCF converged to a magnetic
    solution with a physical moment: lo*m_ref <= |m| <= hi*m_ref. Large I
    can drive the Stoner model (no SCC) into collapsed (m = 0) or
    charge-transfer high-spin states (e.g. Cu d9 at 4.8 mu_B)."""
    return (
        rec is not None
        and "error" not in rec
        and rec.get("converged", False)
        and lo * m_ref <= rec["mean_abs_moment"] <= hi * m_ref
    )


def _curves(data, grid, table):
    """{id: [(ref, calc array over grid, NaN where invalid)]}."""
    out = {}
    for comp in data["compounds"]:
        recs = table[comp["id"]]
        ok = [_valid(r, comp["m_ref"]) for r in recs]
        rows = []
        for c in comp["couplings"]:
            vals = [
                (_match(comp, r, c) if good else None)
                for r, good in zip(recs, ok)
            ]
            vals = np.array([np.nan if v is None else v for v in vals])
            if np.isfinite(vals).any():
                rows.append((c["JS2"], vals))
        if rows:
            out[comp["id"]] = rows
    return out


def _predict(curve, I, grid):
    """JS2 at I by log-I interpolation between the two neighbouring grid
    points; None if either is invalid."""
    x, xs = np.log(I), np.log(grid)
    j = int(np.clip(np.searchsorted(xs, x), 1, len(xs) - 1))
    a, b = curve[j - 1], curve[j]
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    t = np.clip((x - xs[j - 1]) / (xs[j] - xs[j - 1]), 0.0, 1.0)
    return float(a + t * (b - a))


def _residual(ref, calc, floor=1.0):
    """Relative error with a floor (meV); robust to sign changes."""
    return (calc - ref) / (abs(ref) + floor)


INVALID_LOSS = 1.0  # the model has no physical magnetic solution at this I
LOSS = "logratio"  # "relative" (v1, biased towards J -> 0) or "logratio"
LOG_EPS = 0.5  # meV, regularises the log ratio for tiny couplings
WRONG_SIGN_LOSS = 2.0


def _coupling_loss(ref, calc):
    """Loss of one coupling.

    "logratio": |log((|calc| + eps) / (|ref| + eps))| under a Huber, so a
    factor-2 over- and under-estimate cost the same; a wrong sign or no
    physical solution costs WRONG_SIGN_LOSS. "relative": Huber of
    (calc - ref) / (|ref| + 1 meV); predicting J = 0 then costs at most
    |ref| / (|ref| + 1) < 1, which biases the fit towards small J.
    """
    if LOSS == "logratio":
        if calc is None or np.sign(calc) != np.sign(ref):
            return WRONG_SIGN_LOSS
        r = abs(np.log((abs(calc) + LOG_EPS) / (abs(ref) + LOG_EPS)))
    else:
        if calc is None:
            return INVALID_LOSS
        r = abs(_residual(ref, calc))
    return r**2 / 2 if r < 1.0 else r - 0.5


def _compound_loss(rows, I, grid):
    """Mean loss of one compound at parameter I."""
    return float(
        np.mean(
            [
                _coupling_loss(ref, _predict(curve, I, grid))
                for ref, curve in rows
            ]
        )
    )


def _fit_element(ids, curves, grid):
    """Best I on a fine log grid for the compounds ``ids`` (equal weights)."""
    fine = np.exp(np.linspace(np.log(grid[0]), np.log(grid[-1]), 200))
    tot = [sum(_compound_loss(curves[i], I, grid) for i in ids) for I in fine]
    return float(fine[int(np.argmin(tot))])


def _predict_all(rows, I, grid):
    """[(ref, calc or None)]"""
    return [(ref, _predict(curve, I, grid)) for ref, curve in rows]


def _stats(pairs, min_ref=1.0):
    """Statistics over couplings with |ref| >= min_ref. A coupling without a
    physical solution at that I counts as calc = 0 and is reported."""
    pairs = [(r, c) for r, c in pairs if abs(r) >= min_ref]
    if not pairs:
        return {}
    n_invalid = sum(c is None for _, c in pairs)
    r, c = np.array([(r, 0.0 if c is None else c) for r, c in pairs]).T
    ratio = np.abs(c / r)
    same_sign = np.sign(c) == np.sign(r)
    return {
        "N": len(r),
        "no_physical_solution": int(n_invalid),
        "MAE_meV": round(float(np.mean(np.abs(c - r))), 3),
        "median_abs_error_meV": round(float(np.median(np.abs(c - r))), 3),
        "sign_ok": int(same_sign.sum()),
        "median_abs_ratio": round(float(np.median(ratio)), 3),
        "within_factor_2": int(
            np.sum(same_sign & (ratio > 0.5) & (ratio < 2))
        ),
    }


def _group_keys(comp, grouping):
    """Hierarchy of keys, most specific first (e.g. (Mn, ionic, 2), (Mn, ionic), (Mn,))."""
    key = GROUPINGS[grouping](comp)
    return [key[:n] for n in range(len(key), 0, -1)]


def _fit_groups(ids, comps, curves, grid, grouping):
    """{key: I} for every level of the hierarchy with >= MIN_GROUP compounds
    (the element level is always fitted)."""
    members = {}
    for i in ids:
        for key in _group_keys(comps[i], grouping):
            members.setdefault(key, []).append(i)
    return {
        k: _fit_element(v, curves, grid)
        for k, v in members.items()
        if len(v) >= MIN_GROUP or len(k) == 1
    }


def _lookup(table, comp, grouping):
    for key in _group_keys(comp, grouping):
        if key in table:
            return table[key]
    return None


def fit(work, grouping="element", spin_model_kind="stoner"):
    from ase.data import atomic_numbers
    from slakonet.magnetism import DEFAULT_STONER_I

    data, grid, table = _load(work, spin_model_kind)
    curves = _curves(data, grid, table)
    comps = {c["id"]: c for c in data["compounds"] if c["id"] in curves}
    fitted = _fit_groups(list(comps), comps, curves, grid, grouping)

    def atomic(e):
        return DEFAULT_STONER_I.get(atomic_numbers[e], {}).get(2, 0.035)

    # leave-one-compound-out: refit every group without the compound
    loo, atom_pred, fit_pred, per_compound = [], [], [], {}
    for i in sorted(comps):
        e = comps[i]["species"][0]
        others = [j for j in comps if j != i and comps[j]["species"][0] == e]
        if not others:
            continue
        table_loo = _fit_groups(
            [j for j in comps if j != i], comps, curves, grid, grouping
        )
        I_loo = _lookup(table_loo, comps[i], grouping)
        p_loo = _predict_all(curves[i], I_loo, grid)
        p_atom = _predict_all(curves[i], atomic(e), grid)
        p_fit = _predict_all(
            curves[i], _lookup(fitted, comps[i], grouping), grid
        )
        loo += p_loo
        atom_pred += p_atom
        fit_pred += p_fit
        per_compound[i] = {
            "class": list(GROUPINGS[grouping](comps[i])),
            "I_loo": I_loo,
            "couplings": [
                {"ref": r, "atomic_I": a, "fitted_I": f, "loo": l}
                for (r, a), (_, f), (_, l) in zip(p_atom, p_fit, p_loo)
            ],
        }

    report = {
        "grouping": grouping,
        "spin_model": spin_model_kind,
        "loss": LOSS,
        "parameter": (
            "Stoner I_d (Ha)"
            if spin_model_kind == "stoner"
            else "Hubbard U_eff (Ha)"
        ),
        "I_grid_Ha": grid.tolist(),
        "fitted_I_d_Ha": {
            "/".join(map(str, k)): round(v, 4)
            for k, v in sorted(fitted.items(), key=str)
        },
        "atomic_I_d_Ha": {
            e: atomic(e)
            for e in sorted({c["species"][0] for c in comps.values()})
        },
        "stats_atomic_I": _stats(atom_pred),
        "stats_fitted_in_sample": _stats(fit_pred),
        "stats_leave_one_out": _stats(loo),
        "not_in_fit": sorted(
            {c["id"] for c in data["compounds"]} - set(comps)
        ),
        "per_compound": per_compound,
    }
    suffix = "" if spin_model_kind == "stoner" else f"_{spin_model_kind}"
    with open(
        os.path.join(work, f"fit_report_{grouping}{suffix}.json"), "w"
    ) as fh:
        json.dump(report, fh, indent=1)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("per_compound", "not_in_fit")
            },
            indent=1,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=["dataset", "precompute", "fit"])
    p.add_argument(
        "--db", help="Magnetic_MLIP directory (with expdb/, results_ins/)"
    )
    p.add_argument("--work", default="stoner_fit")
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--max-atoms", type=int, default=40)
    p.add_argument("--grouping", choices=list(GROUPINGS), default="element")
    p.add_argument("--spin-model", choices=list(MODELS), default="stoner")
    p.add_argument(
        "--loss", choices=["logratio", "relative"], default="logratio"
    )
    args = p.parse_args()
    global LOSS
    LOSS = args.loss
    if args.stage == "dataset":
        build_dataset(args.db, args.work, args.max_atoms)
    elif args.stage == "precompute":
        precompute(args.work, args.nproc, args.spin_model)
    else:
        fit(args.work, args.grouping, args.spin_model)


if __name__ == "__main__":
    main()
