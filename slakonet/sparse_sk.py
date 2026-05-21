"""Sparse Slater-Koster Hamiltonian / overlap assembly (Gamma-only).

Prototype for step 1 of scaling slakonet to large finite systems: build
H and S directly as sparse COO from a neighbor list, never materializing
the dense Norb x Norb tensor (nor the dense Natom x Natom distance
matrix / dense basis index matrices used by `slaterkoster.hs_matrix`).

The neighbor list comes from the autograd-capable
`slakonet.neighborlist.torch_neighbor_list` (vendored from ALIGNN's
pure-torch graph builder, so ALIGNN is not a strict dependency), so
displacement vectors stay differentiable w.r.t. atomic positions -- the
bridge for a future hybrid slakonet/alignn model. The Slater-Koster
physics (radial integral interpolation + diatomic block rotation) is
reused verbatim from `slakonet.slaterkoster` so results match the
validated dense path bit-for-bit.

Scope of this prototype:
  * non-periodic finite systems (single Gamma point), no k-loop
  * real (non-SCC) H / S
  * single shell per azimuthal number per species (the sp3d5s*-style
    regime of the million-atom TB paper); same limitation as the dense
    path, which keys SK splines by l.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from slakonet.atoms import Geometry
from slakonet.basis import Basis
from slakonet.slaterkoster import hs_matrix, _gather_on_site, sub_block_rot

# torch-native, autograd-capable periodic neighbor list. Vendored into
# slakonet (see slakonet/neighborlist.py) so ALIGNN is not a strict
# dependency of the sparse Slater-Koster path.
from slakonet.neighborlist import torch_neighbor_list


def _squeeze_system(geometry):
    """Return (positions[N,3], atomic_numbers[N]) for a single system."""
    pos = geometry.positions
    z = geometry.atomic_numbers
    if pos.dim() == 3:  # batched (B, N, 3)
        if pos.shape[0] != 1:
            raise NotImplementedError(
                "hs_matrix_sparse handles a single system; got batch "
                f"of {pos.shape[0]}."
            )
        pos = pos[0]
        z = z[0]
    mask = z != 0  # drop padding atoms
    return pos[mask], z[mask]


def _atom_orbital_layout(z: Tensor, shell_dict: dict):
    """Per-atom orbital bookkeeping.

    Returns:
        atom_orb_start: (N,) global orbital offset of each atom.
        n_orb: total number of orbitals.
        shells: list (len N) of lists of (slot, l, local_orb_start) per atom,
            matching the orbital order used by the dense path / basis.on_atoms
            (atom-major, shells in shell_dict order, 2l+1 contiguous per shell).
    """
    n = z.shape[0]
    atom_orb_start = torch.zeros(n, dtype=torch.long, device=z.device)
    shells = []
    running = 0
    for a in range(n):
        za = int(z[a])
        ls = shell_dict[za]
        atom_orb_start[a] = running
        local = 0
        per_atom = []
        for slot, l in enumerate(ls):
            per_atom.append((slot, int(l), local))
            local += 2 * int(l) + 1
        shells.append(per_atom)
        running += local
    return atom_orb_start, running, shells


def _emit(rows_all, cols_all, vals_all, R, C, blk, periodic, ph):
    """Append a per-edge orbital block to the COO triplet lists.

    Periodic: H(k)[i,j] += blk * phase; the Hermitian partner comes from
    the reverse directed image edge (already in the neighbor list).
    Finite (non-periodic): emit blk at (R, C) AND at (C, R) with the
    SAME flat order (swap index arrays, keep blk values aligned) so the
    result is exactly symmetric.
    """
    if periodic:
        rows_all.append(R.reshape(-1))
        cols_all.append(C.reshape(-1))
        vals_all.append((blk * ph[:, None, None]).reshape(-1))
    else:
        rblk = blk.reshape(-1)
        rows_all.append(R.reshape(-1))
        cols_all.append(C.reshape(-1))
        vals_all.append(rblk)
        rows_all.append(C.reshape(-1))
        cols_all.append(R.reshape(-1))
        vals_all.append(rblk)


def hs_matrix_sparse(
    geometry,
    basis,
    sk_feed,
    cutoff: float = 10.0,
    coalesce: bool = True,
    kpoint=None,
    assembly: str = "direct",
) -> Tensor:
    """Assemble a sparse (Norb x Norb) H or S matrix.

    Finite (non-periodic) systems: a single real matrix.
    Periodic systems: the complex Bloch matrix H(k)/S(k) at one
    fractional ``kpoint`` (defaults to Gamma). Images are summed with
    phase ``exp(i 2*pi k . n_cell)`` -- the same convention as the dense
    `slaterkoster.hs_matrix` periodic branch.

    Args:
        geometry: `Geometry` (positions in Bohr). Periodic if it has a
            non-zero cell.
        basis: `Basis` for the system (only shell_dict / on-site used;
            no dense Norb^2 index matrices are touched).
        sk_feed: H-feed or S-feed (`SkFeed`).
        cutoff: interaction cutoff in Bohr (matches dense default).
        coalesce: coalesce the COO tensor before returning.
        kpoint: fractional k-point (len-3) for the periodic case.
        assembly: ``"direct"`` (default, vectorized shell-pair scatter,
            bit-exact and 10-75x faster) or ``"pairwise"`` (per-species-
            pair dense ``hs_matrix`` reuse, kept as a slow reference).

    Returns:
        torch.sparse_coo_tensor (Norb, Norb); real if finite, complex if
        periodic.
    """
    pos, z = _squeeze_system(geometry)
    device, dtype = pos.device, pos.dtype
    shell_dict = basis.shell_dict
    periodic = bool(geometry.is_periodic)

    atom_orb_start, n_orb, shells = _atom_orbital_layout(z, shell_dict)

    if periodic:
        cell = geometry.cell
        while cell.dim() > 2:
            cell = cell[0]
        cell = cell.to(device=device, dtype=dtype)
        if kpoint is None:
            kpoint = torch.zeros(3, device=device, dtype=dtype)
        else:
            kpoint = torch.as_tensor(
                kpoint, device=device, dtype=dtype
            ).flatten()
        lattice = cell
    else:
        # finite system => giant bounding box so no periodic image is
        # ever within `cutoff`.
        span = (pos.max(0).values - pos.min(0).values) + 3.0 * cutoff + 1.0
        lattice = torch.diag(span).to(device=device, dtype=dtype)

    # `r` stays a differentiable function of `pos` (and `lattice`).
    src, dst, shift, r = torch_neighbor_list(
        positions=pos,
        lattice=lattice,
        cutoff=float(cutoff),
        max_neighbors=None,
        atoms=None,
        use_matscipy_topology=False,  # keep pure-torch & differentiable
    )

    out_dtype = torch.complex128 if periodic else dtype
    if src.numel() == 0:
        idx = torch.empty(2, 0, dtype=torch.long, device=device)
        val = torch.empty(0, dtype=out_dtype, device=device)
        H = torch.sparse_coo_tensor(idx, val, (n_orb, n_orb))
        return H.coalesce() if coalesce else H

    if periodic:
        # process EVERY directed image edge; Hermiticity is provided by
        # the reverse (j,i,-S) edge (block^T, conjugate phase).
        ph_all = torch.exp(
            2j * torch.pi * (shift.to(dtype) @ kpoint)
        )  # (E,) exp(i 2pi k.n_cell)
    else:
        # one directed edge per unordered pair; emit B and B^T.
        keep = src < dst
        src, dst, r = src[keep], dst[keep], r[keep]
        ph_all = None

    zi = z[src].to(torch.long)
    zj = z[dst].to(torch.long)

    n_per_atom = torch.tensor(
        [sum(2 * int(l) + 1 for l in shell_dict[int(zz)]) for zz in z],
        device=device,
    )

    rows_all, cols_all, vals_all = [], [], []

    # Group edges by ordered species pair (Zi, Zj). Every pair in a group
    # has an identical 2-atom basis, so the diatomic block is obtained
    # with ONE batched call to the validated dense `hs_matrix` (two-center,
    # environment-free feed => the i-j block equals the full-system block
    # bit-for-bit). Only the off-diagonal atom0xatom1 sub-block is kept;
    # on-site (diagonal) is added once globally below.
    pair_key = zi * 200 + zj
    for pk in torch.unique(pair_key):
        em = pair_key == pk
        Zi = int(zi[em][0])
        Zj = int(zj[em][0])
        e_src = src[em]
        e_dst = dst[em]
        e_r = r[em]  # differentiable image-shifted displacement (Bohr)
        E = e_r.shape[0]

        if assembly == "pairwise":
            # ----- pairwise dense reuse -----
            # Bit-exact reference: one batched call to the validated dense
            # ``hs_matrix`` per species pair (Norb_pair x Norb_pair); we
            # keep only the off-diagonal atom0xatom1 sub-block. Robust but
            # carries large Python/Tensor overhead from Basis construction.
            n0 = int(n_per_atom[e_src][0])
            an_b = torch.tensor(
                [[Zi, Zj]], device=device, dtype=torch.long
            ).expand(E, 2).contiguous()
            zero = torch.zeros(E, 1, 3, device=device, dtype=dtype)
            pos_b = torch.cat([zero, e_r.view(E, 1, 3)], dim=1)
            geom_b = Geometry(an_b, pos_b, units="bohr")
            basis_b = Basis(an_b, shell_dict)
            mat = hs_matrix(geom_b, basis_b, sk_feed, cutoff=cutoff)
            if mat.dim() == 2:
                mat = mat.unsqueeze(0)
            blk = mat[:, :n0, n0:].to(out_dtype)
            ni, nj = blk.shape[1], blk.shape[2]
            gi = atom_orb_start[e_src]
            gj = atom_orb_start[e_dst]
            roff = torch.arange(ni, device=device)
            coff = torch.arange(nj, device=device)
            R = (gi[:, None, None] + roff[None, :, None]).expand(E, ni, nj)
            C = (gj[:, None, None] + coff[None, None, :]).expand(E, ni, nj)
            _emit(
                rows_all, cols_all, vals_all,
                R, C, blk, periodic,
                ph_all[em].to(out_dtype) if periodic else None,
            )

        elif assembly == "direct":
            # ----- direct vectorized SK -> COO -----
            # Bypass Basis/hs_matrix entirely. For each (slot_i, slot_j)
            # shell pair, gather radial integrals (one spline call per
            # canonical (lmin,lmax) key) and rotate via sub_block_rot on
            # the full edge batch. Constant work per (Zi,Zj) group is now
            # ~ (n_shells^2) small kernel launches, with the heavy E axis
            # vectorized.
            ls_i = shell_dict[Zi]
            ls_j = shell_dict[Zj]
            e_dist = e_r.norm(dim=1)
            e_uvec = e_r / e_dist.unsqueeze(-1)
            # local orbital offsets within each species
            loc_i_cum, t = [], 0
            for l in ls_i:
                loc_i_cum.append(t)
                t += 2 * int(l) + 1
            loc_j_cum, t = [], 0
            for l in ls_j:
                loc_j_cum.append(t)
                t += 2 * int(l) + 1
            gi_atom = atom_orb_start[e_src]
            gj_atom = atom_orb_start[e_dst]
            ph_group = (
                ph_all[em].to(out_dtype) if periodic else None
            )

            for slot_i, l1 in enumerate(ls_i):
                l1 = int(l1)
                ni = 2 * l1 + 1
                loc_i = loc_i_cum[slot_i]
                for slot_j, l2 in enumerate(ls_j):
                    l2 = int(l2)
                    nj = 2 * l2 + 1
                    loc_j = loc_j_cum[slot_j]
                    n_int = min(l1, l2) + 1
                    # SKF tables store l1<=l2 with the lmin-bearing atom
                    # first; swap atom order in the key when l1 > l2.
                    if l1 <= l2:
                        key = (Zi, Zj, l1, l2)
                    else:
                        key = (Zj, Zi, l2, l1)
                    splines = sk_feed.off_site_dict.get(key)
                    if splines is None:
                        continue
                    integrals = splines(e_dist)
                    if not torch.is_tensor(integrals):
                        integrals = torch.as_tensor(
                            integrals, dtype=dtype, device=device
                        )
                    else:
                        integrals = integrals.to(
                            dtype=dtype, device=device
                        )
                    if integrals.dim() == 1:
                        integrals = integrals.unsqueeze(-1)
                    integrals = integrals[..., :n_int]

                    lp = torch.tensor([l1, l2], device=device)
                    if l1 == 0 and l2 == 0:
                        block = integrals.view(-1, 1, 1)
                    else:
                        block = sub_block_rot(lp, e_uvec, integrals)
                        block = block.reshape(-1, ni, nj)
                    block = block.to(out_dtype)

                    gi_loc = gi_atom + loc_i
                    gj_loc = gj_atom + loc_j
                    roff_l = torch.arange(ni, device=device)
                    coff_l = torch.arange(nj, device=device)
                    Rl = (
                        gi_loc[:, None, None] + roff_l[None, :, None]
                    ).expand(E, ni, nj)
                    Cl = (
                        gj_loc[:, None, None] + coff_l[None, None, :]
                    ).expand(E, ni, nj)
                    _emit(
                        rows_all, cols_all, vals_all,
                        Rl, Cl, block, periodic, ph_group,
                    )

        else:
            raise ValueError(
                f"assembly must be 'pairwise' or 'direct', got {assembly!r}"
            )

    # on-site (diagonal) reuses the dense helper (S=0 self term, phase 1).
    onsite = _gather_on_site(geometry, basis, sk_feed).reshape(-1)[:n_orb]
    diag_idx = torch.arange(n_orb, device=device)
    rows_all.append(diag_idx)
    cols_all.append(diag_idx)
    vals_all.append(onsite.to(dtype=out_dtype, device=device))

    rows = torch.cat(rows_all)
    cols = torch.cat(cols_all)
    vals = torch.cat(vals_all)
    H = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, (n_orb, n_orb)
    )
    return H.coalesce() if coalesce else H


def _coo_to_scipy_csr(t: Tensor):
    """torch sparse_coo (2D, real or complex) -> scipy csr_matrix."""
    import numpy as np
    from scipy.sparse import coo_matrix

    t = t.coalesce().cpu()
    idx = t.indices().numpy()
    v = t.values()
    if v.is_complex():
        val = v.to(torch.complex128).numpy()
        np_dtype = np.complex128
    else:
        val = v.to(torch.float64).numpy()
        np_dtype = np.float64
    n, m = t.shape
    return coo_matrix(
        (val, (idx[0], idx[1])), shape=(n, m), dtype=np_dtype
    ).tocsr()


def solve_near_gap(
    H,
    S,
    k: int,
    sigma: float,
    return_vectors: bool = False,
):
    """A few interior generalized eigenpairs near energy ``sigma``.

    Solves ``H c = E S c`` for the ``k`` eigenvalues closest to ``sigma``
    using shift-invert Lanczos (ARPACK via scipy ``eigsh``) on the sparse
    operators -- the regime used by the million-atom TB paper (a handful
    of states near the gap, never a full diagonalization).

    Args:
        H, S: sparse Hamiltonian / overlap (torch sparse_coo or scipy
            sparse). ``S`` must be symmetric positive definite.
        k: number of eigenpairs to return.
        sigma: target energy (Hartree) -- e.g. a gap-interior estimate.
        return_vectors: also return eigenvectors.

    Returns:
        evals (sorted) and, if requested, evecs (columns).
    """
    from scipy.sparse.linalg import eigsh

    if torch.is_tensor(H):
        H = _coo_to_scipy_csr(H)
    if torch.is_tensor(S):
        S = _coo_to_scipy_csr(S)

    # shift-invert: returns the k eigenvalues nearest sigma. `M=S` makes it
    # the generalized problem; ARPACK factorizes (H - sigma*S) once. That
    # factorization is singular if sigma coincides with an eigenvalue, so
    # nudge sigma by a tiny relative amount and retry (the result is still
    # the eigenvalues nearest the requested energy).
    sigma = float(sigma)
    scale = abs(sigma) + 1.0
    w = v = None
    for j in range(6):
        s = sigma if j == 0 else sigma + (1e-9 * scale) * (2 ** j) * (
            -1 if j % 2 else 1
        )
        try:
            w, v = eigsh(
                H, k=k, M=S, sigma=s, which="LM", mode="normal"
            )
            break
        except RuntimeError as e:
            if "singular" not in str(e).lower() or j == 5:
                raise
    order = w.argsort()
    w = w[order]
    if return_vectors:
        return w, v[:, order]
    return w


def sparse_bands(
    geometry,
    basis,
    h_feed,
    s_feed,
    kpoints,
    k: int,
    sigma: float,
    cutoff: float = 10.0,
):
    """Near-gap band energies along a list of fractional k-points.

    For each k: assemble sparse complex H(k)/S(k) and pull the ``k``
    eigenvalues nearest ``sigma`` via shift-invert Lanczos. Returns an
    array of shape ``(n_kpoints, k)`` (sorted per k). This is the
    million-atom-paper regime applied to a band path.
    """
    import numpy as np

    kpoints = np.asarray(kpoints, dtype=float).reshape(-1, 3)
    bands = []
    for kp in kpoints:
        Hk = hs_matrix_sparse(
            geometry, basis, h_feed, cutoff=cutoff, kpoint=kp
        )
        Sk = hs_matrix_sparse(
            geometry, basis, s_feed, cutoff=cutoff, kpoint=kp
        )
        bands.append(solve_near_gap(Hk, Sk, k=k, sigma=sigma))
    return np.stack(bands, axis=0)
