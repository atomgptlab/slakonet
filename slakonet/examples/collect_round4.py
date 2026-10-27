"""J(U) from the round-4 DFT+U U-scan (dft_round4_uscan/), by energy mapping.

For every compound and every U batch (<compound>_U<u>/, plus the round-1/2
runs of Magnetic_MLIP at their U for group A):

    E_c = E0_batch + sum_shell JS2_shell * sum_{bonds in shell} s_i s_j

Bonds (each pair once): group A from the round-1 model bonds
(``results_ins/<name>/model_bonds_prim.npy`` tiled to the supercell), group B
from ``cell.extxyz`` (all magnetic pairs up to RMAX).

Shells are chosen once per compound from the design of all its
configurations (nearest first, cond < 1e3, parameters < configurations), so
that every U uses the same shells; a batch whose accepted runs cannot
resolve them all keeps the longest well-conditioned prefix.

Accepted runs: converged, collinear pattern of MAGMOM kept (|m| > 0.15 and
relative signs), and same spin state as the batch: |m - batch median| <=
0.5 muB (a spin-state change moves |m| by ~2 muB; itinerant reductions such
as Cu d9 in FM are smaller). Unlike collect_dft_exchange.py, no FM run is
required.

Conventions: JS2 per bond (meV), E = sum_bonds JS2 s_i.s_j, J > 0 AFM.

Usage:
    python collect_round4.py --round4 dft_round4_uscan --mlip .../Magnetic_MLIP \
        --out dft_reference/round4_JU.json
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

RMAX = 9.2  # Angstrom, group-B bonds (LaMnAsO interlayer 9.03)
SHELL_TOL = 0.03  # Angstrom, distances closer than this form one shell
SPIN_STATE_MUB = 0.5
MAX_SHELLS = 6


def accepted_runs(batch_dir):
    """[(sigma, E eV, label, mean |m|)] of the usable runs of one batch."""
    man = json.loads((batch_dir / "manifest.json").read_text())
    rows = []
    for c in man["configs"]:
        cdir = batch_dir / c["dir"]
        e = cde.vasp_energy(cdir)
        if e is None:
            continue
        mom = cde.outcar_moments(cdir)
        sig = cde.incar_sigma(cdir)
        if not cde.pattern_kept(sig, mom):
            continue
        rows.append(
            (
                np.array(c["sigma"], dtype=float),
                e,
                c["label"],
                float(np.abs(mom[sig != 0]).mean()),
            )
        )
    if not rows:
        return []
    median = float(np.median([r[3] for r in rows]))
    return [r for r in rows if abs(r[3] - median) <= SPIN_STATE_MUB]


def group_b_bonds(cell_file, sigma):
    """(i, j, None, 0, d) for every magnetic pair once, up to RMAX."""
    from ase.io import read
    from ase.neighborlist import neighbor_list

    atoms = read(cell_file)
    mag = np.asarray(sigma) != 0
    ii, jj, dd = neighbor_list("ijd", atoms, RMAX)
    # i < j keeps each pair once (neighbor_list lists i->j and j->i). Bonds of
    # an atom with its own periodic image give s_i s_i = 1 in every
    # configuration, a constant absorbed by the batch offset: left out.
    return [
        (int(i), int(j), None, 0.0, float(d))
        for i, j, d in zip(ii, jj, dd)
        if mag[i] and mag[j] and i < j
    ]


def shell_columns(bonds, sigmas):
    """X[c, s] = sum_{bonds in shell s} s_i s_j, and the shell distances."""
    ds = sorted({round(b[4], 3) for b in bonds})
    groups = []
    for d in ds:
        if groups and d - groups[-1][-1] < SHELL_TOL:
            groups[-1].append(d)
        else:
            groups.append([d])
    centers = [float(np.mean(g)) for g in groups]
    X = np.zeros((len(sigmas), len(groups)))
    index = {d: k for k, g in enumerate(groups) for d in g}
    for i, j, _, _, d in bonds:
        X[:, index[round(d, 3)]] += sigmas[:, i] * sigmas[:, j]
    return X, centers


def well_posed(A, n_rows):
    return (
        A.shape[1] < n_rows
        and np.linalg.matrix_rank(A) == A.shape[1]
        and np.linalg.cond(A) < 1e3
    )


def choose_shells(X, offsets):
    keep = []
    for k in range(X.shape[1]):
        if len(keep) == MAX_SHELLS:
            break
        A = np.hstack([offsets, X[:, keep + [k]]])
        if well_posed(A, len(X)):
            keep.append(k)
    return keep


def fit(X, e, offsets):
    A = np.hstack([offsets, X])
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    dof = len(e) - A.shape[1]
    if dof > 0:
        s2 = float(np.sum((A @ coef - e) ** 2) / dof)
        err = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    else:
        err = np.full(A.shape[1], np.nan)
    nb = offsets.shape[1]
    return coef[nb:], err[nb:]


def compound_batches(name, round4, mlip):
    """{U: [batch dirs]}; group A adds the round-1/2 runs at the round-1 U."""
    out = {}
    for b in sorted(round4.glob(f"{name}_U*")):
        man = json.loads((b / "manifest.json").read_text())
        out.setdefault(float(man["U_eff_eV"]), []).append(b)
    man = json.loads(
        next(iter(out.values()))[0].joinpath("manifest.json").read_text()
    )
    if man["group"] == "A":
        u1 = float(man["round1_U_eV"])
        for d in (
            mlip / "dft_inputs" / name,
            mlip / "dft_round2" / "configs" / name,
        ):
            if (d / "manifest.json").exists():
                out.setdefault(u1, []).append(d)
    return man, out


def collect(name, round4, mlip):
    man, by_u = compound_batches(name, round4, mlip)
    runs = {
        u: [(b, accepted_runs(b)) for b in dirs] for u, dirs in by_u.items()
    }
    if man["group"] == "A":
        from ase.io import read

        n_prim = len(read(mlip / "results_ins" / name / "prim_mp.extxyz"))
        bonds = cde.tile(
            cde.load_bonds(
                mlip / "results_ins" / name / "model_bonds_prim.npy"
            ),
            n_prim,
            tuple(man["reps"]),
        )
    else:
        first = next(b for dirs in by_u.values() for b in dirs)
        sigma0 = json.loads((first / "manifest.json").read_text())["configs"][
            0
        ]["sigma"]
        bonds = group_b_bonds(first / "cell.extxyz", sigma0)

    # shells from all configurations of all U (one offset per batch)
    all_rows = [
        (k, r)
        for k, (u, lst) in enumerate(
            (u, rows) for u, bl in runs.items() for _, rows in bl
        )
        for r in lst
    ]
    n_batches = 1 + max(k for k, _ in all_rows)
    sig = np.vstack([r[0] for _, r in all_rows])
    X_all, centers = shell_columns(bonds, sig)
    offsets = np.eye(n_batches)[[k for k, _ in all_rows]]
    offsets = offsets[:, offsets.any(axis=0)]
    shells = choose_shells(X_all, offsets)

    result = {
        "name": name,
        "group": man["group"],
        "shells_A": [centers[k] for k in shells],
        "U": {},
    }
    for u in sorted(runs):
        rows = [(k, r) for k, (_, lst) in enumerate(runs[u]) for r in lst]
        if len(rows) < 3:
            result["U"][f"{u:g}"] = {"n_runs": len(rows), "JS2": None}
            continue
        sig_u = np.vstack([r[0] for _, r in rows])
        e_u = np.array([r[1] for _, r in rows]) * 1e3
        X_u, _ = shell_columns(bonds, sig_u)
        off_u = np.eye(len(runs[u]))[[k for k, _ in rows]]
        off_u = off_u[:, off_u.any(axis=0)]
        n_ok = len(shells)
        while n_ok and not well_posed(
            np.hstack([off_u, X_u[:, shells[:n_ok]]]), len(e_u)
        ):
            n_ok -= 1
        if n_ok == 0:
            result["U"][f"{u:g}"] = {"n_runs": len(rows), "JS2": None}
            continue
        J, err = fit(X_u[:, shells[:n_ok]], e_u, off_u)
        result["U"][f"{u:g}"] = {
            "n_runs": len(rows),
            "moment": float(np.median([r[3] for _, r in rows])),
            "JS2": [float(x) for x in J] + [None] * (len(shells) - n_ok),
            "err": [float(x) for x in err] + [None] * (len(shells) - n_ok),
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--round4", required=True)
    p.add_argument("--mlip", required=True)
    p.add_argument("--out", default="round4_JU.json")
    args = p.parse_args()
    round4, mlip = Path(args.round4), Path(args.mlip)
    names = sorted(
        {
            b.name.rsplit("_U", 1)[0]
            for b in round4.glob("*_U*")
            if (b / "manifest.json").exists()
        }
    )
    out = []
    for name in names:
        try:
            r = collect(name, round4, mlip)
        except Exception as err:  # report and continue with the next compound
            print(f"{name}: FAILED ({type(err).__name__}: {err})")
            continue
        out.append(r)
        print(
            f"{name:12s} {r['group']} shells "
            + " ".join(f"{d:.2f}" for d in r["shells_A"])
        )
        for u, v in r["U"].items():
            js = "  ".join(
                "   --   " if j is None else f"{j:+7.2f}±{e:.2f}"
                for j, e in zip(v["JS2"] or [], v.get("err") or [])
            )
            print(
                f"    U={u:>4s} n={v['n_runs']:2d}  {js if v['JS2'] else 'not enough runs'}"
            )
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{len(out)} compounds -> {args.out}")


if __name__ == "__main__":
    main()
