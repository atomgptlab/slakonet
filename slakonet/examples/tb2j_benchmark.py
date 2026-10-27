"""Benchmark slakonet -> TB2J exchange constants against ExpMagDB.

For each compound: read the reference structure, set the experimental
collinear order (supercell if needed), converge the slakonet spin model,
run TB2J (ExchangeCL2) and compare shell-averaged J with the references.

Conventions
-----------
TB2J:     E = -sum_{i != j} J_ij e_i.e_j   (each pair twice, |e| = 1, J>0 FM)
ExpMagDB: E =  sum_{pairs} JS2 s_i.s_j     (each pair once, JS2 = J S^2, J>0 AFM)
=> JS2 = -2 J_TB2J.

Stoner parameter
----------------
--mode hubbard    : default model of the interface: atomic Stoner + universal
                    +U (fitted to experiment) + SCC (slakonet.hubbard_spin)
--mode fitted     : Stoner with the per-site table fitted to experiment
--mode default    : atomic Stoner I (slakonet.magnetism) + per-case overrides
--mode calibrated : d-shell I calibrated to the experimental moment of the
                    database (see ``experimental_moment``)
--scc             : also converge SCC charges (periodic DFTB gamma)

Usage
-----
    python tb2j_benchmark.py --db /path/to/Magnetic_MLIP/expdb --out bench_out
    python tb2j_benchmark.py --db ... --cases Fe NiO --mode calibrated --scc
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from ase.build import make_supercell
from ase.io import read

from slakonet.optim import default_model
from slakonet.tb2j_interface import (
    build_spin_model,
    calibrate_stoner,
    gen_exchange_slakonet,
)


def _rocksalt_afm2(atoms, magnetic):
    """AFM-II (k = 1/2 1/2 1/2) cell from the fcc primitive rocksalt cell."""
    P = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    sc = make_supercell(atoms, P)
    frac = np.linalg.solve(atoms.cell.T, sc.positions.T).T
    parity = np.rint(frac).astype(int).sum(axis=1) % 2
    m = [
        (1 - 2 * p) * 2.0 if s in magnetic else 0.0
        for p, s in zip(parity, sc.get_chemical_symbols())
    ]
    return sc, m


def _rutile_afm(atoms, magnetic):
    """Corner (+) / body-centre (-) order of rutile MF2."""
    m = []
    for s, f in zip(
        atoms.get_chemical_symbols(), atoms.get_scaled_positions()
    ):
        if s in magnetic:
            m.append(2.0 if np.allclose(f % 1.0, 0.0, atol=1e-3) else -2.0)
        else:
            m.append(0.0)
    return atoms, m


def _corundum_afm(atoms, magnetic):
    """+-+- along the trigonal axis (Cr2O3)."""
    f = atoms.get_scaled_positions()
    idx = [
        i for i, s in enumerate(atoms.get_chemical_symbols()) if s in magnetic
    ]
    order = sorted(idx, key=lambda i: f[i].sum() % 3.0)
    m = np.zeros(len(atoms))
    for k, i in enumerate(order):
        m[i] = 2.0 * (-1) ** k
    return atoms, m.tolist()


def _fm(atoms, magnetic):
    return atoms, [
        2.0 if s in magnetic else 0.0 for s in atoms.get_chemical_symbols()
    ]


# id: (magnetic element, order builder, stoner_I override, SCF mesh, TB2J mesh)
CASES = {
    "Fe": ("Fe", _fm, None, (12, 12, 12), (12, 12, 12)),
    "Co": ("Co", _fm, None, (12, 12, 8), (12, 12, 8)),
    "Ni": ("Ni", _fm, None, (8, 8, 8), (8, 8, 8)),
    "NiO": ("Ni", _rocksalt_afm2, {28: {2: 0.12}}, (6, 6, 6), (6, 6, 6)),
    "MnO": ("Mn", _rocksalt_afm2, None, (6, 6, 6), (6, 6, 6)),
    "MnF2": ("Mn", _rutile_afm, None, (5, 5, 7), (5, 5, 7)),
    "FeF2": ("Fe", _rutile_afm, None, (5, 5, 7), (5, 5, 7)),
    "Cr2O3": ("Cr", _corundum_afm, None, (5, 5, 5), (5, 5, 5)),
    "CrI3": ("Cr", _fm, None, (5, 5, 3), (5, 5, 3)),
}


def experimental_moment(db, name):
    """Target moment (mu_B) of the magnetic species of ExpMagDB entry ``name``.

    Priority:
    1. MAGNDATA ordered moment (ExpMagDB-full, collinear commensurate entries
       with the same formula; largest value), capped at the spin-only 2S of
       the dataset when S is known (MAGNDATA values can include orbital
       moments or be outliers, e.g. MnO 5.66 mu_B).
    2. Spin-only 2S of the dataset (INS set).
    3. S quoted with the LKAG references of the couplings (elemental metals).
    Returns (moment, source).
    """
    row = pd.read_csv(os.path.join(db, "expdb.csv")).set_index("id").loc[name]
    S = float(row["S"]) if pd.notna(row["S"]) else None
    full_csv = os.path.join(
        os.path.dirname(os.path.abspath(db)), "expdb_full", "expdb_full.csv"
    )
    if os.path.exists(full_csv):
        full = pd.read_csv(full_csv)
        sel = full[
            (full.formula == row["formula"])
            & (full.collinear == True)  # noqa: E712
            & full.k_type.isin([0, 1])
            & full.mean_moment_muB.notna()
        ]
        if len(sel):
            m = float(sel.mean_moment_muB.max())
            if S is not None and m > 2 * S:
                return 2 * S, f"MAGNDATA {m:.2f} capped at 2S"
            return m, "MAGNDATA"
    if S is not None:
        return 2 * S, "2S (INS dataset)"
    with open(os.path.join(db, "entries", name, "info.json")) as fh:
        couplings = json.load(fh)["couplings"]
    S_ref = [c["S"] for c in couplings if c.get("S")]
    if S_ref:
        return 2 * float(np.median(S_ref)), "2S (LKAG reference)"
    raise ValueError(f"No experimental moment for {name}")


def shell_average(path, tol=0.03):
    """[(distance, n_bonds, mean J_TB2J meV, spread)] from TB2J.pickle."""
    with open(os.path.join(path, "TB2J.pickle"), "rb") as fh:
        d = pickle.load(fh)
    rows = sorted(
        (d["distance_dict"][key][1], J * 1e3)
        for key, J in d["exchange_Jdict"].items()
    )
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if r[0] - cur[-1][0] < tol:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r]
    groups.append(cur)
    return [
        (
            float(np.mean([g[0] for g in grp])),
            len(grp),
            float(np.mean([g[1] for g in grp])),
            float(np.ptp([g[1] for g in grp])),
        )
        for grp in groups
    ]


def run_case(name, db, out, model, nz, mode="default", use_scc=False):
    elem, builder, stoner, scf_mesh, tb2j_mesh = CASES[name]
    entry = os.path.join(db, "entries", name)
    with open(os.path.join(entry, "info.json")) as fh:
        info = json.load(fh)
    refs = [c for c in info["couplings"] if c.get("JS2_meV") is not None]
    atoms, m0 = builder(read(os.path.join(entry, "POSCAR")), {elem})
    rcut = max(c["distance_A"] for c in refs) + 0.3

    t0 = time.time()
    target = source = None
    if mode == "hubbard":
        from slakonet.hubbard_spin import build_hubbard_model

        sm = build_hubbard_model(
            atoms,
            model,
            initial_moments=m0,
            kmesh=scf_mesh,
            max_iter=400,
            verbose=False,
        )
    elif mode == "fitted":
        sm = build_spin_model(
            atoms,
            model,
            initial_moments=m0,
            kmesh=scf_mesh,
            use_scc=use_scc,
            verbose=False,
            max_iter=400,
        )
    elif mode == "calibrated":
        target, source = experimental_moment(db, name)
        stoner, sm = calibrate_stoner(
            atoms,
            {elem: target},
            model,
            initial_moments=m0,
            kmesh=scf_mesh,
            use_scc=use_scc,
            max_iter=400,
            verbose=False,
        )
    else:
        sm = build_spin_model(
            atoms,
            model,
            stoner_I=stoner,
            initial_moments=m0,
            kmesh=scf_mesh,
            use_scc=use_scc,
            verbose=False,
            max_iter=400,
            stoner_table="atomic",
        )
    t_scf = time.time() - t0
    path = os.path.join(out, name)
    gen_exchange_slakonet(
        atoms,
        [elem],
        spin_model=sm,
        kmesh=tb2j_mesh,
        nz=nz,
        Rcut=rcut,
        output_path=path,
        verbose=False,
    )
    t_tot = time.time() - t0

    shells = shell_average(path)
    mags = [
        float(m)
        for m, s in zip(sm.moments, atoms.get_chemical_symbols())
        if s == elem
    ]
    comparison = []
    for c in refs:
        match = min(shells, key=lambda s: abs(s[0] - c["distance_A"]))
        comparison.append(
            {
                "shell": c["shell"],
                "distance_ref": c["distance_A"],
                "kind": c["kind"],
                "JS2_ref": c["JS2_meV"],
                "distance_calc": round(match[0], 3),
                "JS2_calc": round(-2.0 * match[2], 4),
                "n_bonds": match[1],
                "reference": c.get("reference", ""),
            }
        )
    return {
        "id": name,
        "natoms": len(atoms),
        "moments_mag": mags,
        "scf_converged": sm.converged,
        "scf_iter": sm.n_iter,
        "efermi": sm.efermi,
        "stoner_I": str(sm.stoner_I),
        "site_stoner_I": str(sm.site_stoner_I),
        "hubbard_U": str(sm.hubbard_U),
        "mode": mode,
        "scc": use_scc,
        "target_moment": target,
        "target_source": source,
        "time_scf_s": round(t_scf, 1),
        "time_total_s": round(t_tot, 1),
        "shells": shells,
        "comparison": comparison,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", required=True, help="ExpMagDB directory (expdb/)")
    p.add_argument("--out", default="tb2j_benchmark")
    p.add_argument("--cases", nargs="*", default=list(CASES))
    p.add_argument("--nz", type=int, default=60)
    p.add_argument(
        "--mode",
        choices=["hubbard", "fitted", "default", "calibrated"],
        default="hubbard",
    )
    p.add_argument("--scc", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from slakonet.spin_training import maybe_corrected

    model = maybe_corrected(default_model())
    results = []
    for name in args.cases:
        print(f"=== {name}", flush=True)
        try:
            r = run_case(
                name, args.db, args.out, model, args.nz, args.mode, args.scc
            )
        except Exception as err:  # record the failure, keep benchmarking
            print(f"  FAILED: {type(err).__name__}: {err}")
            results.append(
                {"id": name, "error": f"{type(err).__name__}: {err}"}
            )
            continue
        results.append(r)
        if r["target_moment"] is not None:
            print(
                f"  target {r['target_moment']:.2f} mu_B ({r['target_source']})  "
                f"Stoner I {r['stoner_I']}"
            )
        print(
            f"  moments {np.round(r['moments_mag'], 3).tolist()}  "
            f"SCF conv={r['scf_converged']}  time={r['time_total_s']} s"
        )
        for c in r["comparison"]:
            print(
                f"  {c['shell']:<22s} d={c['distance_ref']:.2f}/{c['distance_calc']:.2f} A  "
                f"JS2 ref={c['JS2_ref']:+8.3f}  calc={c['JS2_calc']:+8.3f} meV  ({c['kind']})"
            )
        with open(os.path.join(args.out, "benchmark.json"), "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
