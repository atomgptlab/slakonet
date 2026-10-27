"""Exchange constants from the in-house VASP runs of Magnetic_MLIP (round 1 +
round-2 "configs"), by energy mapping, for use as reference data.

For every compound, the converged collinear configurations of both rounds
(same cell, same settings) are fitted together:

    E_c = E0_batch + sum_shell JS2_shell * sum_{bonds in shell} s_i s_j

with one energy offset per batch (each batch has its own FM run). Shells are
added nearest first while the design stays well conditioned (cond < 1e3)
and the number of parameters is smaller than the number of configurations.
A shell is "reliable" when its standard error is below max(1 meV, 0.3 |JS2|).
Runs whose magnetic pattern or spin state changed in DFT are excluded.

Conventions: JS2 per bond (meV), E = sum_bonds JS2 s_i.s_j, J > 0 AFM, as in
ExpMagDB. Level of theory: PBE+U (Materials Project U), collinear.

The helpers below are adapted from Magnetic_MLIP/benchmark/dft_collect.py,
ins.py and ins_mc.py (copied to avoid their torch/numba imports).

Usage:
    python collect_dft_exchange.py --mlip /path/to/Magnetic_MLIP --out dft_exchange.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


# --- adapted from Magnetic_MLIP/benchmark -----------------------------------
def tile(bonds, n_prim, reps):
    """Map primitive-cell bonds onto a diagonal supercell (ase .repeat order)."""
    L = np.array(reps)
    cells = list(itertools.product(*[range(n) for n in L]))
    cid = {c: k for k, c in enumerate(cells)}
    out = []
    for i, j, S, J, d in bonds:
        for c in cells:
            t = np.array(c) + S
            cj = t % L
            out.append(
                (cid[c] * n_prim + i, cid[tuple(cj)] * n_prim + j, None, J, d)
            )
    return out


def load_bonds(path):
    return [
        (int(r[0]), int(r[1]), r[2:5].astype(int), float(r[5]), float(r[6]))
        for r in np.load(path)
    ]


def vasp_energy(cdir):
    """Free energy (eV) of a converged run, else None: CONVERGED marker, or
    (older round-1 runs) an OUTCAR that reached EDIFF plus the last F= of
    OSZICAR. FAILED_FINAL runs are rejected."""
    if (cdir / "FAILED_FINAL").exists():
        return None
    marker = cdir / "CONVERGED"
    if marker.exists():
        fields = dict(
            kv.split("=", 1) for kv in marker.read_text().split() if "=" in kv
        )
        return float(fields["energy_eV"])
    out, osz = cdir / "OUTCAR", cdir / "OSZICAR"
    if not (out.exists() and osz.exists()):
        return None
    if "aborting loop because EDIFF is reached" not in out.read_text(
        errors="replace"
    ):
        return None
    lines = [l for l in osz.read_text().splitlines() if "F=" in l]
    return float(lines[-1].split("F=")[1].split()[0]) if lines else None


def outcar_moments(cdir):
    out = cdir / "OUTCAR"
    if not out.exists():
        return None
    lines = out.read_text(errors="replace").splitlines()
    starts = [
        k
        for k, l in enumerate(lines)
        if l.strip().startswith("magnetization (x)")
    ]
    if not starts:
        return None
    mom = []
    for l in lines[starts[-1] + 4 :]:
        parts = l.split()
        if not parts or not parts[0].isdigit():
            break
        mom.append(float(parts[-1]))
    return np.array(mom)


def incar_sigma(cdir):
    incar = (
        cdir / "INCAR.orig"
        if (cdir / "INCAR.orig").exists()
        else cdir / "INCAR"
    )
    line = next(
        l
        for l in incar.read_text().splitlines()
        if l.strip().upper().startswith("MAGMOM")
    )
    vals = []
    for tok in line.split("=", 1)[1].split():
        n, v = tok.split("*") if "*" in tok else ("1", tok)
        vals += [float(v)] * int(n)
    return np.sign(vals)


def pattern_kept(sigma, mom, threshold=0.15):
    sigma = np.asarray(sigma, dtype=float)
    mag = sigma != 0
    if (
        mom is None
        or len(mom) != len(sigma)
        or np.any(np.abs(mom[mag]) < threshold)
    ):
        return False
    rel = np.sign(mom[mag]) * sigma[mag]
    return bool(np.all(rel == rel[0]))


def shell_design(bonds_cell, sigmas, n_shells, tol=0.03):
    ds = sorted({round(b[4], 2) for b in bonds_cell})
    groups = []
    for d in ds:
        if groups and d - groups[-1][-1] < tol:
            groups[-1].append(d)
        else:
            groups.append([d])
    groups = groups[:n_shells]
    X = np.zeros((len(sigmas), len(groups)))
    for s_idx, g in enumerate(groups):
        for i, j, _, _, d in bonds_cell:
            if any(abs(round(d, 2) - x) < 1e-9 for x in g):
                X[:, s_idx] += sigmas[:, i] * sigmas[:, j]
    return X, [float(np.mean(g)) for g in groups]


# -----------------------------------------------------------------------------


def batch_runs(batch_dir):
    """(sigmas, energies eV, labels) of the accepted runs of one batch."""
    man = json.loads((batch_dir / "manifest.json").read_text())
    rows = []
    for c in man["configs"]:
        cdir = batch_dir / c["dir"]
        e = vasp_energy(cdir)
        if e is None:
            continue
        mom = outcar_moments(cdir)
        sig = incar_sigma(cdir)
        if not pattern_kept(sig, mom):
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
        return man, []
    fm = [r for r in rows if r[2] == "FM"]
    if not fm:
        return man, []
    m_fm = fm[0][3]
    rows = [
        r for r in rows if abs(r[3] / m_fm - 1) < 0.35
    ]  # same spin state as FM
    return man, rows


def fit_compound(name, mlip):
    batches = []
    for d in (
        mlip / "dft_inputs" / name,
        mlip / "dft_round2" / "configs" / name,
    ):
        if (d / "manifest.json").exists():
            man, rows = batch_runs(d)
            if len(rows) >= 2:
                batches.append((man, rows))
    if not batches:
        return None
    reps = tuple(batches[0][0]["reps"])
    if any(tuple(m["reps"]) != reps for m, _ in batches):
        raise ValueError(f"{name}: batches use different cells")
    from ase.io import read

    n_prim = len(read(mlip / "results_ins" / name / "prim_mp.extxyz"))
    bonds = tile(
        load_bonds(mlip / "results_ins" / name / "model_bonds_prim.npy"),
        n_prim,
        reps,
    )
    sig = np.vstack([r[0] for _, rows in batches for r in rows])
    e = np.array([r[1] for _, rows in batches for r in rows]) * 1e3  # meV
    batch_id = np.concatenate(
        [[b] * len(rows) for b, (_, rows) in enumerate(batches)]
    )
    offsets = np.eye(len(batches))[batch_id]
    X_all, d_all = shell_design(bonds, sig, 12)
    keep = []
    for k in range(X_all.shape[1]):
        trial = keep + [k]
        A = np.hstack([offsets, X_all[:, trial]])
        if (
            A.shape[1] >= len(e)
            or np.linalg.matrix_rank(A) < A.shape[1]
            or np.linalg.cond(A) > 1e3
        ):
            continue
        keep = trial
    if not keep:
        return None
    A = np.hstack([offsets, X_all[:, keep]])
    coef, res, *_ = np.linalg.lstsq(A, e, rcond=None)
    dof = len(e) - A.shape[1]
    sigma2 = float(np.sum((A @ coef - e) ** 2) / dof) if dof > 0 else np.nan
    cov = (
        sigma2 * np.linalg.inv(A.T @ A)
        if dof > 0
        else np.full((A.shape[1],) * 2, np.nan)
    )
    J = coef[len(batches) :]
    err = np.sqrt(np.diag(cov))[len(batches) :]
    return {
        "name": name,
        "n_configs": int(len(e)),
        "n_batches": len(batches),
        "shells": [
            {
                "d": d_all[k],
                "dft": float(j),
                "err": float(s),
                "reliable": bool(
                    np.isfinite(s) and s < max(1.0, 0.3 * abs(j))
                ),
            }
            for k, j, s in zip(keep, J, err)
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mlip", required=True, help="Magnetic_MLIP directory")
    p.add_argument("--out", default="dft_exchange.json")
    args = p.parse_args()
    mlip = Path(args.mlip)
    names = sorted(
        {
            p.name
            for p in (mlip / "dft_round2" / "configs").iterdir()
            if p.is_dir()
        }
        | {
            p.name
            for p in (mlip / "dft_inputs").iterdir()
            if (p / "manifest.json").exists()
        }
    )
    results, flat = [], []
    for name in names:
        if not (mlip / "results_ins" / name / "model_bonds_prim.npy").exists():
            continue
        try:
            r = fit_compound(name, mlip)
        except Exception as err:  # report and continue with the next compound
            print(f"{name}: FAILED ({type(err).__name__}: {err})")
            continue
        if r is None:
            print(f"{name}: not enough accepted runs")
            continue
        results.append(r)
        flat += [
            {
                "name": name,
                "d": s["d"],
                "dft": s["dft"],
                "err": s["err"],
                "reliable": s["reliable"],
            }
            for s in r["shells"]
        ]
        print(
            f"{name}: {r['n_configs']} configs / {r['n_batches']} batches, shells "
            + ", ".join(
                f"{s['d']:.2f}:{s['dft']:+.2f}±{s['err']:.2f}{'' if s['reliable'] else '?'}"
                for s in r["shells"]
            )
        )
    Path(args.out).write_text(json.dumps(flat, indent=1))
    Path(args.out).with_suffix(".full.json").write_text(
        json.dumps(results, indent=1)
    )
    print(
        f"{len(results)} compounds, {len(flat)} shells, "
        f"{sum(s['reliable'] for s in flat)} reliable -> {args.out}"
    )


if __name__ == "__main__":
    main()
