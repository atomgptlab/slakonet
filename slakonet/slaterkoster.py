# -*- coding: utf-8 -*-
"""Code associated with performing Slater-Koster transformations.

This houses the code responsible for constructing & applying the Slater-Koster
transformations that rotate matrix elements from the reference frame of the
parametrisation set, [0,0,1], into that required by the calculation.

Hamiltonian & overlap matrices can be constructed via calls to `sk_hs_matrix`.
"""
from typing import Dict, Tuple
import numpy as np
import torch
from torch.nn.functional import normalize
from torch import Tensor, stack
from slakonet.atoms import Geometry
from slakonet.skfeed import SkfFeed
from slakonet.skfeed import _SkFeed as SkFeed
from slakonet.atoms import Geometry, Periodic
from slakonet.basis import Basis
from slakonet.utils import pack, split_by_size

# from tbmalt import Basis, Periodic
# from tbmalt.common import split_by_size
# from tbmalt.common.batch import pack
# from tbmalt.ml.skfeeds import SkFeed
import torch.nn.functional as F

# Static module-level constants (used for SK transformation operations)
_SQR3, _SQR6, _SQR10, _SQR15 = np.sqrt(
    np.array([3.0, 6.0, 10.0, 15.0])
).tolist()
_HSQR3 = 0.5 * np.sqrt(3.0)


def fermi(eigenvalue: Tensor, nelectron: Tensor, kT=0.0, spin=None):
    """
    Fully vectorized fermi function that avoids indexing issues.
    Uses broadcasting and masks instead of loops.
    """
    assert torch.all(torch.ge(nelectron, 1))

    batch_size = nelectron.shape[0]
    n_states = eigenvalue.shape[-1]
    device = eigenvalue.device
    dtype = eigenvalue.dtype

    if kT != 0:
        raise NotImplementedError("Smearing not implemented.")

    if spin is None:
        # Non-spin case: electrons fill in pairs

        # Calculate number of electron pairs and singles
        n_pairs = torch.div(
            nelectron, 2, rounding_mode="floor"
        )  # [batch_size]
        n_singles = nelectron % 2  # [batch_size]

        # Create state indices on the same device: [batch_size, n_states]
        state_indices = (
            torch.arange(n_states, device=device, dtype=nelectron.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Create masks for doubly and singly occupied states
        # Doubly occupied: state index < n_pairs
        doubly_occupied_mask = state_indices < n_pairs.unsqueeze(
            1
        )  # [batch_size, n_states]

        # Singly occupied: state index == n_pairs AND there's a single electron
        singly_occupied_mask = (state_indices == n_pairs.unsqueeze(1)) & (
            n_singles.unsqueeze(1) > 0
        )

        # Create occupancy tensor
        occ = torch.zeros(batch_size, n_states, dtype=dtype, device=device)
        occ[doubly_occupied_mask] = 2.0
        occ[singly_occupied_mask] = 1.0

        # Calculate number of occupied states
        nocc = torch.div(nelectron.float(), 2.0).ceil()

        return occ, nocc

    else:
        # Spin-polarized case: one electron per state

        # Get number of electrons for this spin
        n_elec_spin = nelectron[:, spin]  # [batch_size]

        # Create state indices on the same device
        state_indices = (
            torch.arange(n_states, device=device, dtype=nelectron.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Create mask for occupied states
        occupied_mask = state_indices < n_elec_spin.unsqueeze(1)

        # Create occupancy tensor
        occ = torch.zeros(batch_size, n_states, dtype=dtype, device=device)
        occ[occupied_mask] = 1.0

        # Calculate number of occupied states
        nocc = torch.div(nelectron.float(), 2.0).ceil()

        return occ, nocc


def fffffermi(eigenvalue: Tensor, nelectron: Tensor, kT=0.0, spin=None):
    """Fermi-Dirac distributions without smearing.
    Arguments:
        eigenvalue: Eigen-energies.
        nelectron: number of electrons.
        kT: temperature.
    Returns
        occ: occupancies of electrons
    """
    # make sure each system has at least one electron
    assert False not in torch.ge(nelectron, 1)
    if spin is None:
        # the number of full occupied state
        electron_pair = torch.true_divide(nelectron.clone().detach(), 2).int()
        # the left single electron
        electron_single = (nelectron.clone().detach() % 2).unsqueeze(1)

        # zero temperature
        if kT != 0:
            raise NotImplementedError("not implement smearing method.")

        # occupied state for batch, if full occupied, occupied will be 2
        # with unpaired electron, return 1
        occ_ = pack(
            [
                torch.cat(
                    (torch.ones(electron_pair[i]) * 2, electron_single[i]), 0
                )
                for i in range(nelectron.shape[0])
            ]
        )

        # pad the rest unoccupied states with 0
        occ = F.pad(
            input=occ_, pad=(0, eigenvalue.shape[-1] - occ_.shape[-1]), value=0
        )

        # all occupied states (include full and not full occupied)
        nocc = (nelectron.clone().detach() / 2.0).ceil()

        return occ, nocc
    else:
        electron_pair = torch.true_divide(
            nelectron[..., spin].clone().detach(), 1
        ).int()
        # the left single electron
        electron_single = (
            nelectron[..., spin].clone().detach() % 1
        ).unsqueeze(1)

        # zero temperature
        if kT != 0:
            raise NotImplementedError("not implement smearing method.")

        # occupied state for batch, if full occupied, occupied will be 2
        # with unpaired electron, return 1
        occ_ = pack(
            [
                torch.cat(
                    (torch.ones(electron_pair[i]), electron_single[i]), 0
                )
                for i in range(nelectron.shape[0])
            ]
        )

        # pad the rest unoccupied states with 0
        occ = F.pad(
            input=occ_, pad=(0, eigenvalue.shape[-1] - occ_.shape[-1]), value=0
        )

        # all occupied states (include full and not full occupied)
        nocc = (nelectron.clone().detach() / 2.0).ceil()

        return occ, nocc


def hs_matrix(
    geometry: Geometry,
    basis: Basis,
    sk_feed: SkFeed,
    train_onsite=None,
    ml_onsite=None,
    scale_dict=None,
    orbital_resolved=False,
    cutoff=10.0,
    multi_varible=None,
    # geometry: Geometry, basis: Basis, sk_feed: SkFeed, **kwargs
) -> Tensor:
    """Build nueral network Hamiltonian or overlap dictionary.

    Constructs the Hamiltonian or overlap matrix for the target system(s)
    through the application of Slater-Koster transformations to the integrals
    provided by the ``sk_feed``.

    Arguments:
        geometry: `Geometry` or `Periodic` instances associated with the target system(s).
        basis: `Shell` instance associated with the target system(s).
        sk_feed: The Slater-Koster feed entity responsible for providing the
            requisite Slater Koster integrals and on-site terms.

    Keyword Arguments:
        kwargs: `kwargs` are passed into all calls made to the ``sk_feed``
            object's `off_site` & `on_site` methods. This permits additional
            information, such as that needed for environmental dependent feeds,
            to be provided to these methods without having to add custom call
            signatures for each Slater-Koster feed type.

    Returns:
        mat: Hamiltonian or overlap matrices for the target systems.

    Warnings:
        There is no tolerance unit test for this function; one will be created
        once a production grade standard `SkFeed` object has been implemented
        & a set of dummy skf files have been created. However, tests carried
        out during development have shown the method to be within acceptable
        tolerances. Please don't modify this function until a tolerance test
        has been implemented.

    Todo:
        - Create tolerance unit test once a production level standard `SkFeed`
          object has been implemented and a dummy set of skf files have been
          created (which include f-orbitals).

    """

    assert geometry.positions.dtype is torch.float64, "dtype should be float64"
    is_periodic = geometry.is_periodic
    # train_onsite = kwargs.get("train_onsite", None)
    # ml_onsite = kwargs.get("ml_onsite", None)
    # scale_dict = kwargs.get("scale_dict", None)
    # orbital_resolved = kwargs.get("orbital_resolved", False)
    # cutoff = kwargs.get("cutoff", 10.0)
    shape_orbs = basis.orbital_matrix_shape
    g_var = None

    if not is_periodic:
        mat = torch.zeros(
            shape_orbs,  # <- Results matrix
            device=geometry.positions.device,
            dtype=torch.float64,
        )
    else:
        # n_kpoints = kwargs.get("n_kpoints")
        n_kpoints = geometry.n_kpoints
        phase = geometry.phase
        assert n_kpoints is not None, "Please set n_kpoints if PBC is True"
        # assert phase is not None, "Please set phase if PBC is True"
        if isinstance(n_kpoints, Tensor):
            n_kpoints = torch.max(n_kpoints)
        mat = torch.zeros(
            *shape_orbs,
            n_kpoints,
            device=geometry.positions.device,
            dtype=torch.complex128 if phase is not None else torch.float64,
        )

    # The multi_varible offer multi-dimensional interpolation and gather of
    # integrals, the default will be 1D interpolation only with distances
    # multi_varible = kwargs.get("multi_varible", None)

    # Matrix Initialisation, matrix indice for full, block, or shell ...
    # include indice belong to which batch, which the 1st, 2nd atoms are ...
    if not is_periodic:
        l_mat_f = basis.azimuthal_matrix(mask_diag=True, mask_on_site=True)
        l_mat_s = basis.azimuthal_matrix("shell", mask_on_site=True)
    else:
        l_mat_f = basis.azimuthal_matrix(mask_diag=False, mask_on_site=False)
        l_mat_s = basis.azimuthal_matrix("shell", mask_on_site=False)

    i_mat_s = basis.index_matrix("shell")
    i_mat_f = basis.index_matrix("full")
    an_mat_a = basis.atomic_number_matrix("atomic")

    dist_mat_a = geometry.distances
    vec_mat_a = -normalize(geometry.distance_vectors, 2, -1)  # Unit vectors

    # Build mask for l-like distances matrix to select atomic pairs
    _, mask_dist_l, mask_dist_s = basis.mask(geometry, cutoff)

    # Loop over each azimuthal-pair interaction (max ℓ=3 (f))
    l_pairs = [
        torch.tensor([i, j]) for i in range(4) for j in range(4) if i <= j
    ]
    for l_pair in l_pairs:

        # Mask identifying indices associated with the current l_pair target
        if l_pair[0] == l_pair[1]:  # only calculate upper triangle
            index_mask_s = torch.nonzero(
                (l_mat_s == l_pair).all(dim=-1)
                * (i_mat_s[..., 0] <= i_mat_s[..., 1])
                * mask_dist_s
            ).T
        else:
            # Calculate only for l1 < l2, use transpose to get all
            index_mask_s = torch.nonzero(
                (l_mat_s == l_pair).all(dim=-1) * mask_dist_s
            ).T

        if len(index_mask_s[0]) == 0:  # Skip if no l_pair blocks are found
            continue

        # Gather from i_mat_s to get the atom index mask.
        index_mask_a = index_mask_s.clone()
        index_mask_a[-2:] = i_mat_s[[*index_mask_s]].T

        # Gather the atomic numbers, distances, and unit vectors.
        atom_pairs_l = an_mat_a[[*index_mask_a]]

        if not is_periodic:
            g_dist = dist_mat_a[[*index_mask_a]]
            g_vecs = vec_mat_a[[*index_mask_a]]
        else:
            g_dist = dist_mat_a[[*index_mask_a]].T
            g_dist[g_dist.eq(0)] = 99999
            g_vecs = vec_mat_a.permute(0, 2, 3, 4, 1)[[*index_mask_a]].permute(
                2, 0, 1
            )
            atom_pairs_l = atom_pairs_l.repeat(g_dist.shape[0], 1, 1)
            if multi_varible is not None:
                g_var = multi_varible.repeat(
                    g_dist.shape[0], 1, 1, 1, 1
                ).permute(1, 2, 3, 0, -1)
                g_var = g_var[[*index_mask_a]].transpose(0, 1)
            else:
                g_var = None

            if scale_dict is not None:
                scale = scale_dict[tuple(l_pair.tolist())][[*index_mask_a]].T

        # Mask the distances to avoid unnecessary memory use
        mask_dist = g_dist.lt(cutoff) * g_dist.gt(0)
        if not mask_dist.any():
            continue

        g_dist = g_dist[mask_dist]
        g_vecs = g_vecs[mask_dist]
        atom_pairs_l = atom_pairs_l[mask_dist]
        if multi_varible is not None and is_periodic:
            g_var = g_var[mask_dist]

        if scale_dict is not None:
            scale = scale[mask_dist].unsqueeze(-1)

        # gather multi_varible
        if not is_periodic and multi_varible is not None:
            g_var = _gether_var(multi_varible, index_mask_a)

        # Get off-site integrals from the sk_feed, passing on any kwargs
        # provided by the user. If the SK-feed is environmentally dependent,
        # then it will need the indices of the atoms; as this data cannot be
        # provided by the user it must be explicitly added to the kwargs here.
        integrals = _gather_off_site(
            atom_pairs_l,
            l_pair,
            g_dist,
            sk_feed,
            shell_dict=basis.shell_dict,
            g_var=g_var,
            # **kwargs,
        )
        if scale_dict is not None:
            integrals = integrals * scale

        # Make a call to the relevant Slater-Koster function to get the sk-block
        if (l_pair == torch.tensor([0, 0])).all():
            sk_data = integrals.unsqueeze(-2)
        else:
            sk_data = sub_block_rot(l_pair, g_vecs, integrals)

        # Generate SK data in various K-points
        if is_periodic:
            mask_img_dist = torch.nonzero(mask_dist)
            mask_batch = index_mask_a[0][mask_img_dist[..., 1]]
            mask_atm1 = index_mask_a[1][mask_img_dist[..., 1]]
            mask_atm2 = index_mask_a[2][mask_img_dist[..., 1]]

            # .  .  .  .  .  .  batch .  . which k-points .  . atom 1  .  atom 2  .
            mask_img_dist = (
                mask_batch,
                mask_img_dist[..., 0],
                mask_atm1,
                mask_atm2,
            )
            sk_data = _pe_sk_data2(sk_data, phase, mask_img_dist)

        # the indices to bridge the gap between blocks and flatten data
        if l_pair.ne(0).all():
            # size of block, row/col
            nr, nc = l_pair * 2 + 1

            # Get total orbital index
            idx_mask_u, nl = index_mask_s[:2].unique(
                dim=-1, return_counts=True
            )
            n_tot = torch.arange(int(nr * len(index_mask_s[-1]))) + 1

            # Get flatted indices which corresponds to the following a_mask
            r = pack(torch.split(n_tot, tuple(nl * nr))).reshape(
                len(nl), -1, nr
            )
            r = r.transpose(-1, -2).flatten()
            r = r[r.ne(0)] - 1

            # Perform the reordering
            if not is_periodic:
                sk_data = sk_data.reshape(-1, nc)[r]
            else:
                sk_data = sk_data.reshape(-1, nc, sk_data.shape[-1])[r]

        if not is_periodic:
            sk_data = sk_data.flatten()
        elif l_pair[0] == 0 or l_pair[1] == 0:
            sk_data = sk_data.flatten(0, 2)
        else:
            sk_data = sk_data.flatten(0, 1)

        # Create the full sized index mask and assign the results.
        if l_pair[0] == l_pair[1]:
            a_mask = torch.nonzero(
                # orbital mask, only atomic pairs has such pairs
                (l_mat_f == l_pair).all(-1)
                # upper triangle with diagonal, reduce matrix calculations
                # With diagonal mainly due to neighbouring images in PBC
                * (i_mat_f[..., 0] <= i_mat_f[..., 1])
                # orbital like distance mask, reduce matrix calculations
                * mask_dist_l
            ).T

            # This mask is used to reproduce the lower triangle integrals
            # As long as PyTorch has a stable eigen solver for half matrix
            # This will be deleted
            a_mask1 = torch.nonzero(
                (l_mat_f == l_pair).all(-1)
                * mask_dist_l
                * (i_mat_f[..., 0] < i_mat_f[..., 1])
            ).T
        else:
            # Example: sp orbital
            # ss0 sp0 sp1 sp2
            # sp0 ...
            # sp1 ...
            # sp2 ...
            # Only calculate row, and use the following code to get all if PBC:
            # mat.transpose(-2, -3)[[*a_mask1]] = ...
            a_mask = torch.nonzero((l_mat_f == l_pair).all(-1) * mask_dist_l).T

        mat[[*a_mask]] = sk_data  # (ℓ_1, ℓ_2) blocks, i.e. the row blocks
        if not is_periodic:
            # (ℓ_2, ℓ_1) column-wise
            mat.transpose(-1, -2)[[*a_mask]] = sk_data
        else:
            if l_pair[0] == l_pair[1]:
                mat.transpose(-2, -3)[[*a_mask1]] = torch.conj(mat[[*a_mask1]])
            else:
                mat.transpose(-2, -3)[[*a_mask]] = torch.conj(sk_data)

    # Set the onsite terms (diagonal)
    if not train_onsite or train_onsite == "global" or ml_onsite is None:
        _onsite = _gather_on_site(geometry, basis, sk_feed)
        # _onsite = _gather_on_site(geometry, basis, sk_feed, **kwargs)
    elif train_onsite == "local":
        if not orbital_resolved:
            _onsite = ml_onsite

            # Repeat p, d orbitals from 1 to 3, 5...
            _onsite = torch.repeat_interleave(
                _onsite, basis.orbs_per_shell[basis.orbs_per_shell.ne(0)]
            )

            # Pack results if necessary (code has no effect on single systems)
            c = torch.unique_consecutive(
                (basis.on_atoms != -1).nonzero().T[0], return_counts=True
            )[1]
            _onsite = pack(torch.split(_onsite, tuple(c))).view(
                basis.orbital_matrix_shape[:-1]
            )
        else:
            _onsite = (
                sk_feed.on_site_dict["ml_onsite"]
                if ml_onsite is None
                else ml_onsite
            )

    if not is_periodic:
        mat.diagonal(0, -2, -1)[:] = mat.diagonal(0, -2, -1)[:] + _onsite
    else:
        # REVISE, ONSITE in different k-space
        _onsite = _onsite.repeat(n_kpoints, 1, 1).permute(1, 0, 2)
        mat.diagonal(0, -2, -3)[:] = mat.diagonal(0, -2, -3)[:] + _onsite

    return mat


def hs_matrix_nn(geometry, basis: Basis, sk_feed: SkFeed, **kwargs) -> Tuple:

    # If add all neighbouring H ans S to central cell
    mat_dict, onsite_dict, idx_dict = {}, {}, {}

    # If True, return integrals of unique (atom, orbital)
    is_periodic = geometry.is_periodic

    cutoff = kwargs.get("cutoff", 10.0)
    if not is_periodic:
        l_mat_s = basis.azimuthal_matrix("shell", mask_on_site=True)
    else:
        l_mat_s = basis.azimuthal_matrix("shell", mask_on_site=False)

    i_mat_s = basis.index_matrix("shell")
    an_mat_a = basis.atomic_number_matrix("atomic")

    dist_mat_a = geometry.distances

    # Build mask for l-like distances matrix to select atomic pairs
    _, mask_dist_l, mask_dist_s = basis.mask(geometry, cutoff)

    # The multi_varible offer multi-dimensional interpolation and gather of
    # integrals, the default will be 1D interpolation only with distances
    multi_varible = kwargs.get("multi_varible", None)

    # Loop over each azimuthal-pair interaction (max ℓ=3 (f))
    l_pairs = torch.tensor(
        [[i, j] for i in range(4) for j in range(4) if i <= j]
    )

    for l_pair in l_pairs:

        if l_pair[0] == l_pair[1]:  # only calculate upper triangle
            index_mask_s = torch.nonzero(
                (l_mat_s == l_pair).all(dim=-1)
                * (i_mat_s[..., 0] <= i_mat_s[..., 1])
                * mask_dist_s
            ).T
        else:
            # Calculate only for l1 < l2, use transpose to get all
            index_mask_s = torch.nonzero(
                (l_mat_s == l_pair).all(dim=-1) * mask_dist_s
            ).T

        if len(index_mask_s[0]) == 0:  # Skip if no l_pair blocks are found
            continue

        # Gather from i_mat_s to get the atom index mask.
        index_mask_a = index_mask_s.clone()
        index_mask_a[-2:] = i_mat_s[[*index_mask_s]].T

        # Gather the atomic numbers, distances, and unit vectors.
        atom_pairs_l = an_mat_a[[*index_mask_a]]
        if not is_periodic:
            g_dist = dist_mat_a[[*index_mask_a]]
        else:
            g_dist = dist_mat_a[[*index_mask_a]].T
            g_dist[g_dist.eq(0)] = 99999
            atom_pairs_l = atom_pairs_l.repeat(g_dist.shape[0], 1, 1)

        # Mask the distances to avoid unnecessary memory use
        mask_dist = g_dist.lt(cutoff) * g_dist.gt(0)
        if not mask_dist.any():
            continue

        g_dist = g_dist[mask_dist]
        atom_pairs_l = atom_pairs_l[mask_dist]

        # gather multi_varible
        g_var = _gether_var(multi_varible, index_mask_a)

        integrals = _gather_off_site(
            atom_pairs_l,
            l_pair,
            g_dist,
            sk_feed,
            shell_dict=basis.shell_dict,
            g_var=g_var,
            **kwargs,
        )

        mat_dict.update({tuple(l_pair.tolist()): integrals})

        # This code works to generate integrals mask of atomic pairs
        u_atom_pair_l = torch.unique(atom_pairs_l, dim=0)
        for pair in u_atom_pair_l:
            mask = (pair == atom_pairs_l).all(1)
            idx_dict.update(
                {
                    tuple(pair.tolist())
                    + tuple(l_pair.tolist()): (index_mask_a, mask_dist, mask)
                }
            )

    return mat_dict, idx_dict, sk_feed.on_site_dict


def add_kpoint(
    hs_pred_dict: Dict[tuple, Tensor],
    h_index_dict,
    geometry,
    shell_dict,
    basis: Basis,
    sk_feed: SkFeed = None,
    hs_onsite: dict = {},
    orbital_resolve=False,
    **kwargs,
):
    """
    Arguments:
        hs_pred_dict: Size of each is [n_batch, n_atom, n_atom, n_kpoint, m],
            n_kpoint is for periodic system, m equals the min(l).
    """
    hs_dict = kwargs.get("hs_dict", None)
    hs_merge_pred_dict = (
        {}
    )  # Merge the same orbitals in different atomic pairs
    number = torch.unique(geometry.atomic_numbers)
    unique_number = number[number.ne(0)]

    assert geometry.positions.dtype is torch.float64, "dtype should be float64"
    is_periodic = geometry.is_periodic
    train_onsite = kwargs.get("train_onsite", True)
    cutoff = kwargs.get("cutoff", 10.0)
    shape_orbs = basis.orbital_matrix_shape

    n_kpoints = geometry.n_kpoints
    phase = geometry.phase
    assert n_kpoints is not None, "Please set n_kpoints if PBC is True"
    # assert phase is not None, "Please set phase if PBC is True"
    if isinstance(n_kpoints, Tensor):
        n_kpoints = torch.max(n_kpoints)
    matc = torch.zeros(
        *shape_orbs,
        n_kpoints,
        device=geometry.positions.device,
        dtype=torch.complex128 if phase is not None else torch.float64,
    )

    # The multi_varible offer multi-dimensional interpolation and gather of
    # integrals, the default will be 1D interpolation only with distances
    multi_varible = kwargs.get("multi_varible", None)

    # Matrix Initialisation, matrix indice for full, block, or shell ...
    # include indice belong to which batch, which the 1st, 2nd atoms are ...
    if not is_periodic:
        l_mat_f = basis.azimuthal_matrix(mask_diag=True, mask_on_site=True)
        l_mat_s = basis.azimuthal_matrix("shell", mask_on_site=True)
    else:
        l_mat_f = basis.azimuthal_matrix(mask_diag=False, mask_on_site=False)
        l_mat_s = basis.azimuthal_matrix("shell", mask_on_site=False)

    i_mat_s = basis.index_matrix("shell")
    i_mat_f = basis.index_matrix("full")
    an_mat_a = basis.atomic_number_matrix("atomic")
    dist_mat_a = geometry.distances
    vec_mat_a = -normalize(geometry.distance_vectors, 2, -1)  # Unit vectors

    # Build mask for l-like distances matrix to select atomic pairs
    _, mask_dist_l, mask_dist_s = basis.mask(geometry, cutoff)

    # Loop over each azimuthal-pair interaction (max ℓ=3 (f))
    l_pairs = [
        torch.tensor([i, j]) for i in range(4) for j in range(4) if i <= j
    ]

    for l_pair in l_pairs:

        # Mask identifying indices associated with the current l_pair target
        if l_pair[0] == l_pair[1]:  # only calculate upper triangle
            index_mask_s = torch.nonzero(
                (l_mat_s == l_pair).all(dim=-1)
                * (i_mat_s[..., 0] <= i_mat_s[..., 1])
                * mask_dist_s
            ).T
        else:
            # Calculate only for l1 < l2, use transpose to get all
            index_mask_s = torch.nonzero(
                (l_mat_s == l_pair).all(dim=-1) * mask_dist_s
            ).T

        if len(index_mask_s[0]) == 0:  # Skip if no l_pair blocks are found
            continue

        # Gather from i_mat_s to get the atom index mask.
        index_mask_a = index_mask_s.clone()
        index_mask_a[-2:] = i_mat_s[[*index_mask_s]].T

        # Gather the atomic numbers, distances, and unit vectors.
        atom_pairs_l = an_mat_a[[*index_mask_a]]

        g_dist = dist_mat_a[[*index_mask_a]].T
        g_dist[g_dist.eq(0)] = 99999
        g_vecs = vec_mat_a.permute(0, 2, 3, 4, 1)[[*index_mask_a]].permute(
            2, 0, 1
        )
        atom_pairs_l = atom_pairs_l.repeat(g_dist.shape[0], 1, 1)

        # Mask the distances to avoid unnecessary memory use
        mask_dist = g_dist.lt(cutoff) * g_dist.gt(0)
        if not mask_dist.any():
            continue

        g_vecs = g_vecs[mask_dist]
        atom_pairs_l = atom_pairs_l[mask_dist]

        # Generate SK data in various K-points
        u_atom_pair_l = torch.unique(atom_pairs_l, dim=0)
        mat = torch.zeros(hs_dict[tuple(l_pair.tolist())].shape)
        key_l = tuple(l_pair.tolist())
        for pair in u_atom_pair_l:
            key = tuple(pair.tolist()) + tuple(l_pair.tolist())
            mask = h_index_dict[key][2]
            mat[mask] = hs_pred_dict[key][0] * hs_dict[key_l][mask]

        sk_data = sub_block_rot(l_pair, g_vecs, mat)

        mask_img_dist = torch.nonzero(mask_dist)
        mask_batch = index_mask_a[0][mask_img_dist[..., 1]]
        mask_atm1 = index_mask_a[1][mask_img_dist[..., 1]]
        mask_atm2 = index_mask_a[2][mask_img_dist[..., 1]]

        # .  .  .  .  .  .  batch .  . which k-points .  . atom 1  .  atom 2  .
        mask_img_dist = (
            mask_batch,
            mask_img_dist[..., 0],
            mask_atm1,
            mask_atm2,
        )
        sk_data = _pe_sk_data2(sk_data, phase, mask_img_dist)

        if l_pair.ne(0).all():
            # size of block, row/col
            nr, nc = l_pair * 2 + 1

            # Get total orbital index
            idx_mask_u, nl = index_mask_s[:2].unique(
                dim=-1, return_counts=True
            )
            n_tot = torch.arange(int(nr * len(index_mask_s[-1]))) + 1

            # Get flatted indices which corresponds to the following a_mask
            r = pack(torch.split(n_tot, tuple(nl * nr))).reshape(
                len(nl), -1, nr
            )
            r = r.transpose(-1, -2).flatten()
            r = r[r.ne(0)] - 1

            # Perform the reordering
            if not is_periodic:
                sk_data = sk_data.reshape(-1, nc)[r]
            else:
                sk_data = sk_data.reshape(-1, nc, sk_data.shape[-1])[r]

        if l_pair[0] == 0:
            sk_data = sk_data.flatten(0, 2)
        else:
            sk_data = sk_data.flatten(0, 1)

        # Create the full sized index mask and assign the results.
        if l_pair[0] == l_pair[1]:
            a_mask = torch.nonzero(
                # orbital mask, only atomic pairs has such pairs
                (l_mat_f == l_pair).all(-1)
                # upper triangle with diagonal, reduce matrix calculations
                # With diagonal mainly due to neighbouring images in PBC
                * (i_mat_f[..., 0] <= i_mat_f[..., 1])
                # orbital like distance mask, reduce matrix calculations
                * mask_dist_l
            ).T

            # This mask is used to reproduce the lower triangle integrals
            # As long as PyTorch has a stable eigen solver for half matrix
            # This will be deleted
            a_mask1 = torch.nonzero(
                (l_mat_f == l_pair).all(-1)
                * mask_dist_l
                * (i_mat_f[..., 0] < i_mat_f[..., 1])
            ).T
        else:
            # Example: sp orbital
            # ss0 sp0 sp1 sp2
            # sp0 ...
            # sp1 ...
            # sp2 ...
            # Only calculate row, and use the following code to get all if PBC:
            # mat.transpose(-2, -3)[[*a_mask1]] = ...
            a_mask = torch.nonzero((l_mat_f == l_pair).all(-1) * mask_dist_l).T

        matc[[*a_mask]] = sk_data  # (ℓ_1, ℓ_2) blocks, i.e. the row blocks

        if l_pair[0] == l_pair[1]:
            matc.transpose(-2, -3)[[*a_mask1]] = torch.conj(matc[[*a_mask1]])
        else:
            matc.transpose(-2, -3)[[*a_mask]] = torch.conj(sk_data)

    # Set the onsite terms (diagonal)
    if not train_onsite:
        _onsite = _gather_on_site(geometry, basis, sk_feed, **kwargs)
    else:
        ind_mat = geometry.atomic_numbers.flatten().repeat_interleave(
            basis.orbs_per_atom.flatten()
        )
        _onsite = torch.zeros(ind_mat.shape)
        # if not orbital_resolve:
        #     for iatom in unique_number.tolist():
        #         _on_tmp = torch.cat([hs_onsite[(iatom, il)]
        #                              for il in shell_dict[iatom]], -1).flatten()
        #         _onsite[ind_mat == iatom] = _on_tmp
        #         _onsite[ind_mat == iatom] = on
        # else:
        for iatom in unique_number.tolist():
            _on_tmp = torch.cat(
                [hs_onsite[(iatom, il)] for il in shell_dict[iatom]], -1
            ).flatten()
            _onsite[ind_mat == iatom] = _on_tmp

        c = torch.unique_consecutive(
            (basis.on_atoms != -1).nonzero().T[0], return_counts=True
        )[1]

        _onsite = pack(torch.split(_onsite, tuple(c))).view(
            basis.orbital_matrix_shape[:-1]
        )

    if not is_periodic:
        matc.diagonal(0, -2, -1)[:] = matc.diagonal(0, -2, -1)[:] + _onsite
    else:
        # REVISE, ONSITE in different k-space
        _onsite = _onsite.repeat(n_kpoints, 1, 1).permute(1, 0, 2)

        matc.diagonal(0, -2, -3)[:] = matc.diagonal(0, -2, -3)[:] + _onsite

    return matc


def _gather_on_site(
    geometry: Geometry, basis: Basis, sk_feed: SkFeed, **kwargs
) -> Tensor:
    """Retrieves on site terms from a target feed in a batch-wise manner.

    This is a convenience function for retrieving on-site terms from an SKFeed
    object.

    Arguments:
        geometry: `Geometry` instance associated with the target system(s).
        basis: `Basis` instance associated with the target system(s).
        sk_feed: The Slater-Koster feed entity responsible for providing the
            requisite Slater Koster integrals and on-site terms.

    Keyword Arguments:
        kwargs: `kwargs` are passed into calls made to the ``sk_feed``
            object's `off_site` method.

    Returns:
        on_site_values: On-site values associated with the specified systems.

    Notes:
        Unlike `_gather_of_site`, this function does not require the keyword
        argument ``atom_indices`` as it can be constructed internally.
    """
    an = geometry.atomic_numbers
    a_shape = basis.atomic_matrix_shape[:-1]
    o_shape = basis.orbital_matrix_shape[:-1]

    # Get the onsite values for all non-padding elements & pass on the indices
    # of the atoms just in case they are needed by the SkFeed
    mask = an.nonzero(as_tuple=True)

    if "atom_indices" not in kwargs:
        kwargs["atom_indices"] = torch.arange(geometry.n_atoms.max()).expand(
            a_shape
        )

    os_flat = torch.cat(sk_feed.on_site(atomic_numbers=an[mask]))
    # os_flat = torch.cat(sk_feed.on_site(atomic_numbers=an[mask], **kwargs))

    if not sk_feed.orbital_resolve:
        os_flat = torch.repeat_interleave(
            os_flat, basis.orbs_per_shell[basis.orbs_per_shell.ne(0)]
        )

    # Pack results if necessary (code has no effect on single systems)
    c = torch.unique_consecutive(
        (basis.on_atoms != -1).nonzero().T[0], return_counts=True
    )[1]
    return pack(split_by_size(os_flat, c)).view(o_shape)


def _gather_off_site(
    atom_pairs: Tensor,
    shell_pairs: Tensor,
    distances: Tensor,
    sk_feed: SkFeed,
    shell_dict: dict = None,
    g_var: Tensor = None,
    **kwargs,
) -> Tensor:
    """Retrieves integrals from a target feed in a batch-wise manner.

    This convenience function mediates the integral retrieval operation by
    splitting requests into batches of like types permitting fast batch-
    wise retrieval.

    Arguments:
        atom_pairs: Atomic numbers of each atom pair.
        shell_pairs: Shell numbers associated with each interaction. Note that
            all shells must correspond to identical azimuthal numbers.
        distances: Distances between the atom pairs.
        sk_feed: The Slater-Koster feed entity responsible for providing the
            requisite Slater Koster integrals and on-site terms.
        is_periodic:

    Keyword Arguments:
        kwargs: Surplus `kwargs` are passed into calls made to the ``sk_feed``
            object's `off_site` method.
        atom_indices: Tensor: Indices of the atoms for which the integrals are
            being evaluated. For a single system this should be a tensor of
            size 2xN where the first & second row specify the indices of the
            first and second atoms respectively. For a batch of systems an
            extra row is appended to the start specifying which system the
            atom pair is associated with.

    Returns:
        integrals: The relevant integral values evaluated at the specified
            distances.

    Notes:
        Any kwargs specified will be passed through to the `integral_feed`
        during function calls. Integrals can only be evaluated for a single
        azimuthal pair at a time.

    Warnings:
        All shells specified in ``shell_pairs`` must have a common azimuthal
        number / angular momentum. This is because shells with azimuthal
        quantum numbers will return a different number of integrals, which
        will cause size mismatch issues.

    """
    n_shell = kwargs.get("n_shell", False)
    # Block the passing of vectors, which can cause hard to diagnose issues
    if distances.ndim > 2:
        raise ValueError(
            'Argument "distances" must be a 1d or 2d torch.tensor.'
        )

    integrals = None

    # Identify all unique [atom|atom|shell|shell] sets.
    as_pairs = torch.cat(
        (atom_pairs, shell_pairs.repeat(atom_pairs.shape[0], 1)), -1
    )
    as_pairs_u = as_pairs.unique(dim=0)

    # Loop over each of the unique atom_pairs
    for as_pair in as_pairs_u:
        # Construct an index mask for gather & scatter operations
        mask = torch.where((as_pairs == as_pair).all(1))[0]

        # Retrieve the integrals & assign them to the "integrals" tensor. The
        # SkFeed class requires all arguments to be passed in as keywords.
        if n_shell:
            shell_pair = [
                shell_dict[as_pair[0].tolist()][as_pair[2]],
                shell_dict[as_pair[1].tolist()][as_pair[3]],
            ]
        else:
            shell_pair = as_pair[..., -2:]
        var = None if g_var is None else g_var[mask]

        off_sites = sk_feed.off_site(
            atom_pair=as_pair[..., :-2],
            shell_pair=shell_pair,
            distances=distances[mask],
            # variables=var,
            # atom_indices=ai_select,
            # **kwargs,
        )

        # The result tensor's shape cannot be *safely* identified prior to the
        # first sk_feed call, thus it must be instantiated in the first loop.
        if integrals is None:
            integrals = torch.zeros(
                (len(as_pairs), off_sites.shape[-1]),
                dtype=distances.dtype,
                device=distances.device,
            )

        # If shells with differing angular momenta are provided then a shape
        # mismatch error will be raised. However, the message given is not
        # exactly useful thus the exception's message needs to be modified.
        try:
            integrals[mask] = off_sites
        except RuntimeError as e:
            if str(e).startswith("shape mismatch"):
                raise type(e)(
                    f"{e!s}. This could be due to shells with mismatching "
                    "angular momenta being provided."
                )

    # Return the resulting integrals
    return integrals


def _gether_var(multi_varible, index_mask_a):
    if multi_varible is None:
        return None
    elif multi_varible.dim() == 2:
        return torch.stack(
            [
                multi_varible[index_mask_a[0], index_mask_a[1]],
                multi_varible[index_mask_a[0], index_mask_a[2]],
            ]
        ).T
    elif multi_varible.dim() == 3:
        # [param1+atom1; param1+atom2; param2+atom1; param2+atom2;]
        return torch.stack(
            [
                multi_varible[..., 0][index_mask_a[0], index_mask_a[1]],
                multi_varible[..., 0][index_mask_a[0], index_mask_a[2]],
                multi_varible[..., 1][index_mask_a[0], index_mask_a[1]],
                multi_varible[..., 1][index_mask_a[0], index_mask_a[2]],
            ]
        ).T


def _pe_sk_data(geometry, sk_data, dist_mat_a, mask, **kwargs):
    """Reshape periodic Hamiltonian and overlap data."""
    neigh_size = dist_mat_a.shape[-1]

    # mask[0] is how many distances in each geometry
    phase = geometry.phase[:, mask[0]].transpose(-1, -2).flatten(1, 2)

    # repeat if there are p, d orbitals, since the total integrals size is not
    # equal to atom pairs size, the last 2 dims of sk_data are n_integrals
    phase = phase.unsqueeze(-1).repeat_interleave(sk_data.shape[-2], -1)
    phase = phase.unsqueeze(-1).repeat_interleave(sk_data.shape[-1], -1)

    sk_data = sk_data.unsqueeze(0).repeat_interleave(phase.shape[0], 0)

    # Reshape data: [n_kpoints, n_atoms, n_cellvec, n_integrals, n_integrals]
    # and then sum over cell vectors dimension
    sk_data = (
        (phase * sk_data)
        .reshape(sk_data.shape[0], neigh_size, -1, *sk_data.shape[2:])
        .sum(1)
    )

    # make the kpoints as last dimension for batch
    return sk_data.permute(1, 2, 3, 0)


def _pe_sk_data2(
    sk_data,
    phase,
    mask_img_dist: tuple,
):

    if phase is not None:
        sk_data = sk_data * phase[
            ..., mask_img_dist[0], mask_img_dist[1]
        ].unsqueeze(-1).unsqueeze(-1)
    else:
        # Only Gamma point
        sk_data = sk_data.unsqueeze(0)

    # .  .  .  .  .  .  .  . n-batch .  .  .  .  . atom 1 .  .  .  . atom 2 .
    mask = torch.stack(
        [mask_img_dist[0], mask_img_dist[-2], mask_img_dist[-1]]
    )
    index_mask_a, origin_idx = mask.unique(dim=-1, return_inverse=True)
    shape = torch.Size(
        (
            sk_data.shape[0],
            index_mask_a.shape[1],
            sk_data.shape[2],
            sk_data.shape[3],
        )
    )

    # Get index for which image it belongs to
    origin_idx = origin_idx.repeat(
        sk_data.shape[0], sk_data.shape[-2], sk_data.shape[-1], 1
    ).permute(0, -1, 1, 2)

    sk_data = (
        torch.zeros(shape, dtype=sk_data.dtype)
        .scatter_add_(1, origin_idx, sk_data)
        .permute(1, 2, 3, 0)
    )

    return sk_data


def sub_block_rot(l_pair: Tensor, u_vec: Tensor, integrals: Tensor) -> Tensor:
    """Diatomic sub-block rotated into the reference frame of the system.

    This takes the unit distance vector and slater-koster integrals between a
    pair of orbitals and constructs the associated diatomic block which has
    been rotated into the reference frame of the system.

    Args:
        l_pair: Azimuthal quantum numbers associated with the orbitals.
        u_vec: Unit distance vector between the orbitals.
        integrals: Slater-Koster integrals between the orbitals, in order of
            σ, π, δ, γ, etc.

    Returns:
        block: Diatomic block(s)

    """
    if u_vec.device != integrals.device:
        raise RuntimeError(  # Better to throw this exception manually
            f"Expected u_vec({u_vec.device}) & integrals({integrals.device}) "
            "to be on the same device!"
        )

    # If smallest is ℓ first the matrix multiplication complexity is reduced
    l1, l2 = int(min(l_pair)), int(max(l_pair))

    # Tensor in which to place the results.
    block = torch.zeros(
        len(u_vec) if u_vec.ndim > 1 else 1,
        2 * l1 + 1,
        2 * l2 + 1,
        device=integrals.device,
    )

    # Integral matrix block (in the reference frame of the parameter set)
    i_mat = sub_block_ref(l_pair.sort()[0], integrals)

    # Identify which vectors must use yz type rotations & which must use xy.
    rot_mask = torch.gt(u_vec[..., -2].abs(), u_vec[..., -1].abs())

    # Perform transformation operation (must do yz & xy type rotations)
    for rots, mask in zip((_sk_yz_rots, _sk_xy_rots), (rot_mask, ~rot_mask)):
        if len(u_vec_selected := u_vec[mask].squeeze()) > 0:
            rot_a = rots[l1](u_vec_selected)
            rot_b = rots[l2](u_vec_selected)
            block[mask] = torch.einsum(
                "...ji,...ik,...ck->...jc", rot_a, i_mat[mask], rot_b
            )

    # The masking operation converts single instances into batches of size 1,
    # therefore a catch is added to undo this.
    if u_vec.dim() == 1:
        block = block.squeeze(1)

    # Transpose if ℓ₁>ℓ₂ and flip the sign as needed.
    if l_pair[0] > l_pair[1]:
        sign = (-1) ** (l1 + l2)
        block = sign * block.transpose(-1, -2)

    return block


def sub_block_ref(l_pair: Tensor, integrals: Tensor):
    """Diatomic sub-block in the Slater-Koster integrals' reference frame.

    This yields the tensor that is multiplied with the transformation matrices
    to produce the diatomic sub-block in the atomic reference frame.

    Args:
        l_pair: Angular momenta of the two systems.
        integrals: Slater-Koster integrals between orbitals with the specified
            angular momenta, in order of σ, π, δ, γ, etc.

    Returns:
        block: Diatomic sub-block in the reference frame of the integrals.

    Notes:
        Each row of ``integrals`` should represent a separate system; i.e.
        a 3x1 matrix would indicate a batch of size 3, each with one integral.
        Whereas a matrix of size 1x3 or a vector of size 3 would indicate one
        system with three integral values.
    """
    l1, l2 = min(l_pair), max(l_pair)

    # Test for anticipated number of integrals to ensure `integrals` is in the
    # correct shape.
    if (m := integrals.shape[-1]) != (n := l1 + 1):
        raise ValueError(
            f"Expected {n} integrals per-system (l_min={l1}), but found {m}"
        )

    # Generate integral reference frame block; extending its dimensionality if
    # working on multiple systems.
    block = torch.zeros(2 * l1 + 1, 2 * l2 + 1, device=integrals.device)
    if integrals.dim() == 2:
        block = block.expand(len(integrals), -1, -1).clone()

    # Fetch the block's diagonal and assign the integrals to it like so
    #      ┌               ┐
    #      │ i_1, 0.0, 0.0 │  Where i_0 and i_1 are the first and second
    #      │ 0.0, i_0, 0.0 │  integral values respectively.
    #      │ 0.0, 0.0, i_1 │
    #      └               ┘
    # While this method is a little messy it is faster than alternate methods
    diag = block.diagonal(offset=l2 - l1, dim1=-2, dim2=-1)
    size = integrals.shape[-1]
    diag[..., -size:] = integrals
    diag[..., : size - 1] = integrals[..., 1:].flip(-1)
    # Return the results; a transpose s required if l1 > l2
    return block if l1 == l_pair[0] else block.transpose(-1, -2)


#################################
# Slater-Koster Transformations #
#################################
# Note that the internal slater-koster transformation functions "_skt_*" are
# able to handle batches of systems, not just one system at a time.
def _rot_yz_s(unit_vector: Tensor) -> Tensor:
    r"""s-block transformation matrix rotating about the y and z axes.

    Transformation matrix for rotating s-orbital blocks, about the y & z axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produces simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Notes:
        This function acts as a dummy subroutine as s integrals do not require
        require transformation operations. This exists primarily to maintain
        functional consistency.
    """
    # Using `norm()` rather than `ones()` allows for backpropagation
    return torch.linalg.norm(unit_vector, dim=-1).view(
        (-1, *[1] * unit_vector.ndim)
    )


def _rot_xy_s(unit_vector: Tensor) -> Tensor:
    r"""s-block transformation matrix rotating about the x and y axes.

    Transformation matrix for rotating s-orbital blocks, about the x & y axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produces simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Notes:
        This function acts as a dummy subroutine as s integrals do not require
        require transformation operations. This exists primarily to maintain
        functional consistency.
    """
    # Using `norm()` rather than `ones()` allows for backpropagation
    return torch.linalg.norm(unit_vector, dim=-1).view(
        (-1, *[1] * unit_vector.ndim)
    )


def _rot_yz_p(unit_vector: Tensor) -> Tensor:
    r"""p-block transformation matrix rotating about the y and z axes.

    Transformation matrix for rotating p-orbital blocks, about the y & z axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produces simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill defined
        when z≈1.
    """
    x, y, z = unit_vector.T
    zeros = torch.zeros_like(x)
    alpha = torch.sqrt(1.0 - z * z)
    rot = stack(
        [
            stack([x / alpha, zeros, -y / alpha], -1),
            stack([y, z, x], -1),
            stack([y * z / alpha, -alpha, x * z / alpha], -1),
        ],
        -1,
    )
    return rot


def _rot_xy_p(unit_vector: Tensor) -> Tensor:
    r"""p-block transformation matrix rotating about the x and y axes.

    Transformation matrix for rotating p-orbital blocks, about the x & y axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produces simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill defined
        when y≈1.
    """
    x, y, z = unit_vector.T
    zeros = torch.zeros_like(x)
    alpha = torch.sqrt(1.0 - y * y)
    rot = stack(
        [
            stack([alpha, -y * z / alpha, -x * y / alpha], -1),
            stack([y, z, x], -1),
            stack([zeros, -x / alpha, z / alpha], -1),
        ],
        -1,
    )
    return rot


def _rot_yz_d(unit_vector: Tensor) -> Tensor:
    r"""d-block transformation matrix rotating about the y and z axes.

    Transformation matrix for rotating d-orbital blocks, about the y & z axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produces simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill defined
        when z≈1.
    """
    x, y, z = unit_vector.T
    zeros = torch.zeros_like(x)
    a = 1.0 - z * z
    b = torch.sqrt(a)
    xz, xy, yz = unit_vector.T * unit_vector.roll(1, -1).T
    xyz = x * yz
    x2 = x * x
    rot = stack(
        [
            stack([-z + 2.0 * x2 * z / a, -x, zeros, y, -2.0 * xyz / a], -1),
            stack(
                [-b + 2.0 * x2 / b, xz / b, zeros, -yz / b, -2.0 * xy / b], -1
            ),
            stack(
                [
                    xy * _SQR3,
                    yz * _SQR3,
                    1.0 - 1.5 * a,
                    xz * _SQR3,
                    _SQR3 * (-0.5 * a + x2),
                ],
                -1,
            ),
            stack(
                [
                    2.0 * xyz / b,
                    -2.0 * y * b + y / b,
                    -_SQR3 * z * b,
                    -2.0 * x * b + x / b,
                    -z * b + 2.0 * x2 * z / b,
                ],
                -1,
            ),
            stack(
                [
                    -xy + 2.0 * xy / a,
                    -yz,
                    0.5 * _SQR3 * a,
                    -xz,
                    0.5 * a - 1.0 + x2 * (-1.0 + 2.0 / a),
                ],
                -1,
            ),
        ],
        -1,
    )
    return rot


def _rot_xy_d(unit_vector: Tensor) -> Tensor:
    r"""d-block transformation matrix rotating about the x and y axes.

    Transformation matrix for rotating d-orbital blocks, about the x & y axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produces simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill defined
        when y≈1.
    """
    x, y, z = unit_vector.T
    a = 1.0 - y * y
    b = torch.sqrt(a)
    xz, xy, yz = unit_vector.T * unit_vector.roll(1, -1).T
    xyz = x * yz
    z2 = z * z
    rot = stack(
        [
            stack(
                [z, -x, xyz * _SQR3 / a, y * (1 - 2 * z2 / a), -xyz / a], -1
            ),
            stack(
                [
                    x * (2 * b - 1 / b),
                    z * (2 * b - 1.0 / b),
                    -y * z2 * _SQR3 / b,
                    -2 * xyz / b,
                    y * (-2 * b + z2 / b),
                ],
                -1,
            ),
            stack(
                [
                    xy * _SQR3,
                    yz * _SQR3,
                    1.5 * z2 - 0.5,
                    xz * _SQR3,
                    0.5 * _SQR3 * (2 * a - z2 - 1),
                ],
                -1,
            ),
            stack(
                [yz / b, -xy / b, -xz * _SQR3 / b, -b + 2 * z2 / b, xz / b], -1
            ),
            stack(
                [
                    xy,
                    yz,
                    _SQR3 * (0.5 * (z2 + 1) - z2 / a),
                    xz - 2 * xz / a,
                    a - 0.5 * z2 - 0.5 + z2 / a,
                ],
                -1,
            ),
        ],
        -1,
    )
    return rot


def _rot_yz_f(unit_vector: Tensor) -> Tensor:
    r"""f-block transformation matrix rotating about the y and z axes.

    Transformation matrix for rotating f-orbital blocks, about the y & z axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produces simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill defined
        when z≈1.
    """
    x, y, z = unit_vector.T
    xz, xy, yz = unit_vector.T * unit_vector.roll(1, -1).T
    xyz = x * yz
    zeros = torch.zeros_like(x)
    a = 1.0 - z * z
    b = torch.sqrt(a)
    c = b**3
    x2 = x * x
    rot = stack(
        [
            stack(
                [
                    x * (2.25 * b - 3 * (x2 + 1) / b + 4 * x2 / c),
                    _SQR6 * z * (0.5 * b - x2 / b),
                    0.25 * _SQR15 * x * b,
                    zeros,
                    -0.25 * _SQR15 * y * b,
                    _SQR6 * xyz / b,
                    y * (-0.75 * b + 0.25 * (12 * x2 + 4) / b - 4 * x2 / c),
                ],
                -1,
            ),
            stack(
                [
                    _SQR6 * xz * (-1.5 + 2 * x2 / a),
                    2 * a - 1 + x2 * (-4 + 2 / a),
                    -0.5 * _SQR10 * x * z,
                    zeros,
                    0.5 * _SQR10 * y * z,
                    xy * (4 - 2 / a),
                    _SQR6 * yz * (0.5 - 2 * x2 / a),
                ],
                -1,
            ),
            stack(
                [
                    _SQR15 * x * (-0.75 * b + x2 / b),
                    _SQR10 * z * (-0.5 * b + x2 / b),
                    x * (-1.25 * b + 1 / b),
                    zeros,
                    y * (1.25 * b - 1 / b),
                    -_SQR10 * xyz / b,
                    _SQR15 * y * (0.25 * b - x2 / b),
                ],
                -1,
            ),
            stack(
                [
                    y * _SQR10 * (-0.25 * a + x2),
                    xyz * _SQR15,
                    _SQR6 * y * (-1.25 * a + 1),
                    z * (-2.5 * a + 1),
                    _SQR6 * x * (-1.25 * a + 1),
                    _SQR15 * z * (-0.5 * a + x2),
                    _SQR10 * x * (-0.75 * a + x2),
                ],
                -1,
            ),
            stack(
                [
                    _SQR15 * yz * (-0.25 * b + x2 / b),
                    _SQR10 * xy * (-1.5 * b + 1 / b),
                    yz * (-3.75 * b + 1 / b),
                    _SQR6 * (1.25 * c - b),
                    xz * (-3.75 * b + 1 / b),
                    _SQR10 * (0.75 * c - 0.25 * (6.0 * x2 + 2) * b + x2 / b),
                    _SQR15 * xz * (-0.75 * b + x2 / b),
                ],
                -1,
            ),
            stack(
                [
                    _SQR6 * y * (0.25 * a - 0.25 * (4 * x2 + 2) + 2 * x2 / a),
                    xyz * (-3 + 2 / a),
                    _SQR10 * y * (0.75 * a - 0.5),
                    0.5 * _SQR15 * a * z,
                    _SQR10 * x * (0.75 * a - 0.5),
                    z * (1.5 * a - 0.5 * (6.0 * x2 + 2) + 2 * x2 / a),
                    _SQR6
                    * x
                    * (0.75 * a - 0.25 * (4 * x2 + 6.0) + 2 * x2 / a),
                ],
                -1,
            ),
            stack(
                [
                    yz * (0.25 * b - (x2 + 1) / b + 4 * x2 / c),
                    _SQR6 * xy * (0.5 * b - 1 / b),
                    0.25 * _SQR15 * yz * b,
                    -0.25 * _SQR10 * c,
                    0.25 * _SQR15 * xz * b,
                    _SQR6 * (-0.25 * c + 0.25 * (2 * x2 + 2) * b - x2 / b),
                    xz * (0.75 * b - 0.25 * (4 * x2 + 12) / b + 4 * x2 / c),
                ],
                -1,
            ),
        ],
        -1,
    )
    return rot


def _rot_xy_f(unit_vector: Tensor) -> Tensor:
    r"""f-block transformation matrix rotating about the x and y axes.

    Transformation matrix for rotating f-orbital blocks, about the x & y axes,
    from the integration frame to the molecular frame. Multiple transformation
    matrices can be produces simultaneously.

    Arguments:
        unit_vector: Unit vector(s) between pair(s) of orbitals.

    Returns:
        rot: Transformation matrix.

    Warnings:
        The resulting transformation matrix becomes numerically ill defined
        when y≈1.
    """
    x, y, z = unit_vector.T
    xz, xy, yz = unit_vector.T * unit_vector.roll(1, -1).T
    xyz = x * yz
    a = 1.0 - y * y
    b = torch.sqrt(a)
    c = b**3
    z2 = z * z
    rot = stack(
        [
            stack(
                [
                    c + (-0.75 * z2 - 0.75) * b + 1.5 * z2 / b,
                    _SQR6 * xz * (0.5 * b - 1 / b),
                    _SQR15 * (0.25 * (z2 + 1) * b - 0.5 * z2 / b),
                    _SQR10 * yz * (-0.25 * (z2 + 3) / b + z2 / c),
                    _SQR15 * xy * (-0.25 * (z2 + 1) / b + z2 / c),
                    _SQR6 * yz * (-0.5 * b + (0.25 * z2 + 0.75) / b - z2 / c),
                    xy * (-b + (0.25 * z2 + 0.25) / b - z2 / c),
                ],
                -1,
            ),
            stack(
                [
                    _SQR6 * xz * (1 - 0.5 / a),
                    -2 * a + 4 * z2 + 1 - 2 * z2 / a,
                    _SQR10 * xz * (-1 + 0.5 / a),
                    _SQR15 * xy * z2 / a,
                    _SQR10 * yz * (1 - 1.5 * z2 / a),
                    xy * (2 - 3 * z2 / a),
                    _SQR6 * yz * (-1 + 0.5 * z2 / a),
                ],
                -1,
            ),
            stack(
                [
                    _SQR15 * (c - 0.75 * (z2 + 1) * b + 0.5 * z2 / b),
                    _SQR10 * xz * (1.5 * b - 1 / b),
                    (3.75 * z2 - 0.25) * b - 2.5 * z2 / b,
                    -0.25 * _SQR6 * yz * (5 * z2 - 1) / b,
                    -0.25 * xy * (15 * z2 - 1) / b,
                    _SQR10 * yz * (-1.5 * b + (0.75 * z2 + 0.25) / b),
                    _SQR15 * xy * (-b + (0.25 * z2 + 0.25) / b),
                ],
                -1,
            ),
            stack(
                [
                    _SQR10 * y * (a - 0.75 * z2 - 0.25),
                    _SQR15 * xy * z,
                    0.25 * _SQR6 * (5 * z2 - 1) * y,
                    z * (2.5 * z2 - 1.5),
                    _SQR6 * x * (1.25 * z2 - 0.25),
                    _SQR15 * z * (a - 0.5 * z2 - 0.5),
                    _SQR10 * x * (a - 0.25 * z2 - 0.75),
                ],
                -1,
            ),
            stack(
                [
                    0.5 * _SQR15 * xyz / b,
                    _SQR10 * y * (-0.5 * b + z2 / b),
                    -2.5 * xyz / b,
                    -0.25 * _SQR6 * x * (5 * z2 - 1) / b,
                    z * (-2.5 * b - 0.25 * (-15 * z2 + 1) / b),
                    _SQR10 * x * (-0.5 * b + (0.75 * z2 + 0.25) / b),
                    _SQR15 * z * (0.5 * b - (0.25 * z2 + 0.25) / b),
                ],
                -1,
            ),
            stack(
                [
                    _SQR6 * y * (a - 0.25 * (3 * z2 + 1) + 0.5 * z2 / a),
                    xyz * (3 - 2 / a),
                    _SQR10 * y * (0.25 * (3 * z2 + 1) - 0.5 * z2 / a),
                    _SQR15 * z * (0.5 * (z2 + 1) - z2 / a),
                    _SQR10 * x * (0.75 * z2 + 0.25 - 1.5 * z2 / a),
                    z * (3 * a - 1.5 * z2 - 3.5 + 3 * z2 / a),
                    _SQR6 * x * (a - 0.25 * z2 - 0.75 + 0.5 * z2 / a),
                ],
                -1,
            ),
            stack(
                [
                    1.5 * xyz / b,
                    _SQR6 * y * (-0.5 * b + z2 / b),
                    -0.5 * _SQR15 * xyz / b,
                    _SQR10 * x * (-0.25 * (3 * z2 + 1) / b + z2 / c),
                    _SQR15 * z * (-0.5 * b + (0.75 * z2 + 0.75) / b - z2 / c),
                    _SQR6 * x * (-0.5 * b + (0.75 * z2 + 0.25) / b - z2 / c),
                    z * (1.5 * b - (0.75 * z2 + 0.75) / b + z2 / c),
                ],
                -1,
            ),
        ],
        -1,
    )

    return rot


_sk_yz_rots = {0: _rot_yz_s, 1: _rot_yz_p, 2: _rot_yz_d, 3: _rot_yz_f}
_sk_xy_rots = {0: _rot_xy_s, 1: _rot_xy_p, 2: _rot_xy_d, 3: _rot_xy_f}
