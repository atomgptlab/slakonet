"""Build one SlaKoNetDB record: real-space H(R)/S(R) plus bands, DOS, gap.

The archived object is H(R), S(R), not H(k): a uniform Gamma-centred mesh
is invertible, so the real-space blocks reproduce *any* k afterwards,
while a band path only ever reproduces itself. The mesh is sized per
structure from the Slater-Koster range,

    N_i >= 2 * cutoff / |a_i| + 1,

and the largest R-block at the Born-von Karman edge is stored as
`far_block`: if that is not ~0 the transform was not converged and the
record must not be trusted. On Si a 2^3 mesh gives 6 eV reconstruction
error and far_block 2.6; 4^3 gives 8e-7 and 1e-7.
"""

from __future__ import annotations
import itertools
import json
import os
import time

import numpy as np
import torch


def mesh_for(atoms, cutoff_bohr=10.0, cap=10):
    """Mesh that makes the inverse transform exact.

    The interaction range must fit inside the Born-von Karman cell along
    each direction, and the relevant width is the *interplanar spacing*
    d_i = V / |a_j x a_k|, not |a_i|. For an fcc primitive cell the two
    differ by sin(60 deg): using |a_i| picks 4 for Si where 5 is needed,
    and the resulting H(R) has 0.1 eV of weight left on the cell edge.
    """
    c = np.asarray(atoms.get_cell(), dtype=float)
    V = abs(np.linalg.det(c))
    d = np.array(
        [
            V / np.linalg.norm(np.cross(c[(i + 1) % 3], c[(i + 2) % 3]))
            for i in range(3)
        ]
    )
    r = cutoff_bohr * 0.529177
    return [int(min(cap, max(3, np.ceil(2 * r / di) + 1))) for di in d]


def build_record(
    atoms,
    model,
    cutoff=10.0,
    dos_sigma=0.1,
    n_dos=3000,
    line_density=15,
    device="cuda",
    hr_tol=1e-6,
    mu=None,
    recon_tol=1e-5,
    max_tries=4,
    mesh_cap=12,
    s_min_eig=1e-8,
):
    from slakonet.ase_calc import SlaKoNetCalculator
    from slakonet.negf import _make_calc, hs_at_kpoints
    from jarvis.core.atoms import ase_to_atoms
    from jarvis.core.kpoints import Kpoints3D
    from slakonet.optim import kpts_to_klines, default_mu

    if mu is None:
        mu = default_mu(model_name="slakonet_v1a_full")
    t0 = time.time()
    c = _make_calc(atoms, model, cutoff=cutoff, device=device)
    k_probe = np.array([0.137, 0.291, 0.412])  # on no mesh we will try
    H_direct = hs_at_kpoints(c, [k_probe])[0][..., 0].cpu().numpy()

    def transform(N):
        """H(R),S(R) from an N-mesh, with the reconstruction error."""
        ks = np.array(
            [
                (i / N[0], j / N[1], l / N[2])
                for i, j, l in itertools.product(*(range(n) for n in N))
            ]
        )
        Hk, Sk = hs_at_kpoints(c, ks)
        Hk = np.moveaxis(Hk.cpu().numpy(), -1, 0)
        Sk = np.moveaxis(Sk.cpu().numpy(), -1, 0)
        Rs = np.array(
            list(itertools.product(*(range(-(n // 2), n // 2 + 1) for n in N)))
        )
        ph = np.exp(-2j * np.pi * (ks @ Rs.T)) / len(ks)
        HR = np.einsum("kij,kr->rij", Hk, ph)
        SR = np.einsum("kij,kr->rij", Sk, ph)
        rebuilt = np.einsum(
            "rij,r->ij", HR, np.exp(2j * np.pi * (Rs @ k_probe))
        )
        # An overlap matrix is positive definite by construction. If S(k)
        # has a negative eigenvalue the generalised problem H c = e S c is
        # not a valid eigenproblem at that k, and the solver returns
        # nonsense -- PtRb3 reaches +-4.9e5 eV, which then propagates
        # silently into the total and formation energy.
        min_eig_S = float(
            min(np.linalg.eigvalsh(Sk[i]).min() for i in range(len(ks)))
        )
        return (HR, SR, Rs, float(np.abs(rebuilt - H_direct).max()), min_eig_S)

    # The geometric estimate is a lower bound, not a guarantee: it assumes
    # the interaction range is the SK cutoff, but a soft tail or an
    # anisotropic cell can push it further. Grow the mesh until the
    # reconstruction actually holds, and record what it took.
    N = mesh_for(atoms, cutoff)
    HR, SR, Rs, recon_err, min_eig_S = transform(N)
    tries = 1
    while recon_err > recon_tol and tries < max_tries:
        N = [min(n + 1, mesh_cap) for n in N]
        if all(n >= mesh_cap for n in N):
            break
        HR, SR, Rs, recon_err, min_eig_S = transform(N)
        tries += 1
    far = 0.0

    s_ok = min_eig_S > s_min_eig
    if not s_ok:
        formula = atoms.get_chemical_formula()
        print(
            f"\n*** REJECTED {formula}: overlap matrix is NOT positive "
            f"definite ***\n"
            f"    min eigenvalue of S(k) = {min_eig_S:.4e} "
            f"(must be > {s_min_eig:g})\n"
            f"    H c = e S c is not a valid eigenproblem at this k, so the\n"
            f"    eigenvalues, Fermi level, band gap and total energy from\n"
            f"    this structure are meaningless. The record is written with\n"
            f"    valid=False so it can be filtered, not silently trusted.\n"
            f"    This is a defect in the Slater-Koster overlap tables for\n"
            f"    this element combination, not a convergence problem.\n",
            flush=True,
        )

    # sparsify: most R blocks are numerically zero
    keep = np.array([np.abs(HR[r]).max() > hr_tol for r in range(len(Rs))])

    calc = SlaKoNetCalculator(
        model,
        kspacing=0.2,
        device=device,
        compute_forces=False,
        compute_stress=False,
    )
    at = atoms.copy()
    at.calc = calc
    energy = float(at.get_potential_energy())
    gap = calc.get_bandgap()
    ef = calc.get_fermi_level()
    # VBM/CBM from the converged mesh solve, same source as gap and E_F.
    vbm = calc.results.get("vbm")
    cbm = calc.results.get("cbm")

    kp = Kpoints3D().kpath(ase_to_atoms(atoms), line_density=line_density)
    bs = calc.band_structure(atoms, npoints=len(kp.kpts))
    e_dos, dos = calc.dos(atoms, num_points=n_dos, sigma=dos_sigma)

    # Formation energy costs nothing extra: the total energy is already
    # in hand and mu is a per-element table. It is only defined when every
    # species has a calibrated mu, so record why it is missing when it is.
    e_form, e_form_note = np.nan, ""
    try:
        syms = atoms.get_chemical_symbols()
        missing = sorted({s for s in syms if s not in mu})
        if missing:
            e_form_note = "no mu for " + ",".join(missing)
        else:
            e_form = (energy - sum(mu[s] for s in syms)) / len(syms)
    except Exception as exc:  # pragma: no cover
        e_form_note = f"{type(exc).__name__}: {exc}"

    return dict(
        HR=HR[keep].astype(np.complex64),
        SR=SR[keep].astype(np.complex64),
        Rvecs=Rs[keep].astype(np.int8),
        mesh=np.array(N),
        recon_err=recon_err,
        mesh_tries=tries,
        min_eig_S=min_eig_S,
        s_positive_definite=bool(s_ok),
        valid=bool(recon_err <= recon_tol and s_ok),
        n_R_kept=int(keep.sum()),
        n_R_total=len(Rs),
        bands=np.asarray(bs["energies"], dtype=np.float32),
        kpts=np.asarray(bs["kpts"], dtype=np.float32),
        dos_e=np.asarray(e_dos, dtype=np.float32),
        dos=np.asarray(dos, dtype=np.float32),
        gap=float(gap) if gap is not None else np.nan,
        vbm=float(vbm) if vbm is not None else np.nan,
        cbm=float(cbm) if cbm is not None else np.nan,
        # Band-path extrema are a *different* estimate: a path can miss an
        # extremum that lies off the high-symmetry lines, so the two gaps
        # are stored separately rather than reconciled.
        gap_path=float(bs.get("gap", np.nan)),
        vbm_path=float(bs.get("vbm", np.nan)),
        cbm_path=float(bs.get("cbm", np.nan)),
        fermi=float(ef),
        energy=energy,
        e_form=e_form,
        e_form_note=e_form_note,
        seconds=round(time.time() - t0, 1),
    )


def write_record(path, rec, meta):
    np.savez_compressed(
        path, meta=json.dumps(meta), **{k: v for k, v in rec.items()}
    )
