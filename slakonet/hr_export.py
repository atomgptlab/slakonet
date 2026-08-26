"""Export a SlaKoNet Hamiltonian as ``wannier90_hr.dat``.

Usable as a library or as ``python -m slakonet.hr_export --jid ...``.

The file format assumes an *orthonormal* basis: Wannier functions have
S = I, so every consumer of it (WannierTools, WannierBerri, postw90)
solves H c = e c. SlaKoNet uses a non-orthogonal Slater--Koster basis,
so writing H(R) directly would silently discard S and produce wrong
bands -- plausible-looking ones, which is worse.

We therefore Lowdin-orthogonalise at each k before transforming to real
space,

    H'(k) = S(k)^{-1/2} H(k) S(k)^{-1/2},

which leaves the eigenvalues of H c = e S c exactly unchanged while
making S the identity. The Lowdin orbitals remain atom-centred and
symmetry-adapted, so orbital-resolved analysis downstream still means
what it usually means.

One caveat is worth checking per system and is reported by
``--check``: S(k)^{-1/2} is not short-ranged, so H'(R) can decay more
slowly than H(R). If the tail is not small at the edge of the R-star,
enlarge the mesh.
"""

from __future__ import annotations
import argparse
import itertools
import os
import numpy as np


def lowdin(Hk, Sk):
    """H'(k) = S^-1/2 H S^-1/2, via the eigendecomposition of S."""
    w, U = np.linalg.eigh(Sk)
    if w.min() <= 0:
        raise ValueError(
            f"overlap is not positive definite (min eigenvalue {w.min():.3e});"
            " this structure cannot be orthogonalised and its generalised"
            " eigenproblem is ill-posed to begin with"
        )
    X = U @ np.diag(w**-0.5) @ np.conj(U).T
    return X @ Hk @ X


def _base_mesh(atoms, cutoff):
    cell = np.asarray(atoms.get_cell()[:], dtype=float)
    V = abs(np.linalg.det(cell))
    d = [
        V / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
        for i in range(3)
    ]
    r = cutoff * 0.529177
    return [int(max(3, np.ceil(2 * r / di) + 1)) for di in d]


def hr_auto(
    atoms,
    model,
    cutoff=10.0,
    device="cpu",
    edge_tol=1e-4,
    max_steps=4,
    verbose=True,
):
    """Grow the mesh until H'(R) has actually decayed at the R-star edge.

    The mesh that makes H(R) exact is *not* enough for H'(R): Lowdin's
    S^{-1/2} is not short-ranged, so the orthogonalised blocks reach
    further than the Slater--Koster interaction does. On silicon the
    mesh that gives H(R) a 1e-7 eV reconstruction error leaves 6e-2 eV
    of H'(R) sitting on the edge of the star, and the exported bands are
    wrong by 5 meV. Roughly doubling the mesh fixes it.
    """
    mesh = _base_mesh(atoms, cutoff)
    for step in range(max_steps):
        HR, Rs, nw, mesh_used = hr_from_model(
            atoms, model, mesh=mesh, cutoff=cutoff, device=device
        )
        edge = float(
            np.abs(HR[np.abs(Rs).max(axis=1) == np.abs(Rs).max()]).max()
        )
        if verbose:
            print(f"  mesh {mesh_used}: edge |H'| = {edge:.3e} eV", flush=True)
        if edge <= edge_tol:
            return HR, Rs, nw, mesh_used, edge
        mesh = [min(21, n + 2) for n in mesh]
        if all(n >= 21 for n in mesh):
            break
    return HR, Rs, nw, mesh_used, edge


def hr_from_model(atoms, model, mesh=None, cutoff=10.0, device="cpu"):
    """Return (H'(R), R vectors, n_wann) in the Lowdin basis, in eV."""
    from slakonet.negf import _make_calc, hs_at_kpoints
    import torch

    c = _make_calc(atoms, model, cutoff=cutoff, device=device)
    if mesh is None:
        cell = np.asarray(atoms.get_cell()[:], dtype=float)
        V = abs(np.linalg.det(cell))
        d = [
            V / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
            for i in range(3)
        ]
        r = cutoff * 0.529177
        mesh = [int(min(12, max(3, np.ceil(2 * r / di) + 1))) for di in d]
    mesh = [int(x) for x in mesh]

    frac = np.array(
        list(itertools.product(*[np.arange(n) / n for n in mesh])), dtype=float
    )
    Hk, Sk = hs_at_kpoints(c, frac)
    Hk = np.asarray(Hk.detach().cpu().numpy())
    Sk = np.asarray(Sk.detach().cpu().numpy())
    if Hk.shape[-1] == len(frac):  # (nb, nb, nk) -> (nk, nb, nb)
        Hk = np.transpose(Hk, (2, 0, 1))
        Sk = np.transpose(Sk, (2, 0, 1))

    Hp = np.stack([lowdin(Hk[i], Sk[i]) for i in range(len(frac))])

    Rs = np.array(
        list(itertools.product(*[range(-(n // 2), n - n // 2) for n in mesh])),
        dtype=int,
    )
    phase = np.exp(-2j * np.pi * (frac @ Rs.T))  # (nk, nR)
    HR = np.einsum("kn,kab->nab", phase, Hp) / len(frac)
    return HR, Rs, Hk.shape[-1], mesh


def write_hr(path, HR, Rs, n_wann, header="written by slakonet"):
    """Write wannier90_hr.dat. Column index varies slowest, as in wannier90."""
    with open(path, "w") as f:
        f.write(f" {header}\n")
        f.write(f"{n_wann:12d}\n")
        f.write(f"{len(Rs):12d}\n")
        # all weights 1: we keep the full commensurate R box rather than a
        # Wigner-Seitz star, so no vector is shared between cells
        for i in range(len(Rs)):
            f.write("    1")
            if (i + 1) % 15 == 0 or i == len(Rs) - 1:
                f.write("\n")
        for n in range(len(Rs)):
            R = Rs[n]
            for jj in range(n_wann):  # column
                for ii in range(n_wann):  # row, fastest
                    v = HR[n, ii, jj]
                    f.write(
                        f"{R[0]:5d}{R[1]:5d}{R[2]:5d}"
                        f"{ii + 1:5d}{jj + 1:5d}"
                        f"{v.real:22.12f}{v.imag:22.12f}\n"
                    )


_ORB = {
    0: ["s"],
    1: ["py", "pz", "px"],
    2: ["dxy", "dyz", "dz2", "dxz", "dx2-y2"],
    3: [
        "fy(3x2-y2)",
        "fxyz",
        "fyz2",
        "fz3",
        "fxz2",
        "fz(x2-y2)",
        "fx(x2-3y2)",
    ],
}


def write_wt_in(path, atoms, model, n_wann, name=""):
    """Minimal WannierTools input matching the exported hr.dat.

    Orbitals within a shell are ordered m = -l .. +l in the real
    spherical-harmonic convention; m = 0 is pz / dz2, which we verified
    against the package's own bond-frame block (the sigma integral lands
    on the middle row). The +/-m labels follow the usual wannier90
    ordering but only matter if you use projected output -- band
    structures and topological invariants do not depend on them.
    """
    from slakonet.negf import _make_calc
    from ase.data import chemical_symbols

    c = _make_calc(atoms, model, cutoff=10.0, device="cpu")
    sd = c.shell_dict
    Z = atoms.get_atomic_numbers()
    cell = np.asarray(atoms.get_cell()[:], dtype=float)
    frac = atoms.get_scaled_positions()
    with open(path, "w") as f:
        f.write(
            f"&TB_FILE\nHrfile = 'wannier90_hr.dat'\nPackage = 'VASP'\n"
            f"/\n\n"
        )
        f.write(
            "!> exported from slakonet"
            + (f" ({name})" if name else "")
            + "; H is in the Lowdin-orthogonalised basis,\n"
            "!> so S = I as WannierTools assumes.\n\n"
        )
        f.write("LATTICE\nAngstrom\n")
        for v in cell:
            f.write(f"{v[0]:18.10f}{v[1]:18.10f}{v[2]:18.10f}\n")
        f.write(f"\nATOM_POSITIONS\n{len(Z)}\nDirect\n")
        for z, p in zip(Z, frac):
            f.write(
                f"{chemical_symbols[z]:4s}"
                f"{p[0]:16.10f}{p[1]:16.10f}{p[2]:16.10f}\n"
            )
        f.write("\nPROJECTORS\n")
        f.write(
            " ".join(str(sum(2 * el + 1 for el in sd[int(z)])) for z in Z)
            + "\n"
        )
        for z in Z:
            labels = [o for el in sd[int(z)] for o in _ORB[el]]
            f.write(f"{chemical_symbols[z]:4s}" + " ".join(labels) + "\n")
        f.write(
            f"\n&CONTROL\nBulkBand_calc = T\n/\n\n"
            f"&SYSTEM\nNumOccupied = {n_wann // 2}   "
            "!> CHECK THIS: set to your electron count\n"
            "SOC = 0\nE_FERMI = 0.0   !> set from the record's fermi\n/\n\n"
            "&PARAMETERS\nNk1 = 101\n/\n\n"
            "KPATH_BULK\n1\nG 0.0 0.0 0.0  X 0.5 0.0 0.5\n"
        )


def check(HR, Rs, atoms, model, cutoff=10.0, device="cpu", n_probe=4, seed=0):
    """Compare bands from the written H'(R) against the generalised solve."""
    from slakonet.negf import _make_calc, hs_at_kpoints

    rng = np.random.default_rng(seed)
    kp = rng.random((n_probe, 3))
    c = _make_calc(atoms, model, cutoff=cutoff, device=device)
    Hk, Sk = hs_at_kpoints(c, kp)
    Hk = np.asarray(Hk.detach().cpu().numpy())
    Sk = np.asarray(Sk.detach().cpu().numpy())
    if Hk.shape[-1] == len(kp):
        Hk = np.transpose(Hk, (2, 0, 1))
        Sk = np.transpose(Sk, (2, 0, 1))
    worst = 0.0
    for q in range(len(kp)):
        ref = np.linalg.eigvalsh(lowdin(Hk[q], Sk[q]))
        ph = np.exp(2j * np.pi * (kp[q] @ Rs.T))
        got = np.linalg.eigvalsh(np.einsum("n,nab->ab", ph, HR))
        worst = max(worst, np.abs(np.sort(ref) - np.sort(got)).max())
    edge = max(
        np.abs(HR[np.abs(Rs).max(axis=1) == np.abs(Rs).max()]).max(), 0.0
    )
    return worst, edge, float(np.abs(HR).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jid", help="JARVIS id, e.g. JVASP-1002")
    ap.add_argument("--poscar", help="structure file instead of --jid")
    ap.add_argument("--model", default="slakonet_v1a_full")
    ap.add_argument("--mesh", type=int, nargs=3, default=None)
    ap.add_argument("--out", default="wannier90_hr.dat")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--edge-tol",
        type=float,
        default=1e-4,
        help="grow the mesh until |H'| at the R-star edge is "
        "below this (eV); ignored if --mesh is given",
    )
    ap.add_argument(
        "--wt",
        action="store_true",
        help="also write a wt.in template for WannierTools",
    )
    a = ap.parse_args()

    from slakonet.optim import default_model
    from jarvis.core.atoms import Atoms

    if a.poscar:
        at = Atoms.from_poscar(a.poscar).ase_converter()
        name = os.path.basename(a.poscar)
    else:
        from jarvis.db.figshare import data

        d = {r["jid"]: r for r in data("dft_3d")}[a.jid]
        at = Atoms.from_dict(d["atoms"]).ase_converter()
        name = a.jid
    model = default_model(model_name=a.model).float()

    if a.mesh:
        HR, Rs, nw, mesh = hr_from_model(
            at, model, mesh=a.mesh, device=a.device
        )
        edge = float(
            np.abs(HR[np.abs(Rs).max(axis=1) == np.abs(Rs).max()]).max()
        )
        if edge > a.edge_tol:
            print(
                f"  WARNING: |H'| at the R-star edge is {edge:.3e} eV, "
                f"above {a.edge_tol:.0e}; the exported bands will be "
                "wrong. Drop --mesh to let it converge."
            )
    else:
        HR, Rs, nw, mesh, edge = hr_auto(
            at, model, device=a.device, edge_tol=a.edge_tol
        )
    write_hr(
        a.out,
        HR,
        Rs,
        nw,
        header=f"slakonet {a.model} {name} mesh {mesh} (Lowdin basis)",
    )
    print(f"wrote {a.out}: {nw} orbitals, {len(Rs)} R vectors, mesh {mesh}")
    if a.wt:
        wt = os.path.join(os.path.dirname(a.out) or ".", "wt.in")
        write_wt_in(wt, at, model, nw, name=name)
        print(f"wrote {wt}")
    if a.check:
        err, edge, big = check(HR, Rs, at, model, device=a.device)
        print(f"  eigenvalue agreement vs generalised solve: {err:.3e} eV")
        print(
            f"  largest |H'| on the R-star edge: {edge:.3e} eV "
            f"(max |H'| {big:.3e} eV)"
        )


if __name__ == "__main__":
    main()
