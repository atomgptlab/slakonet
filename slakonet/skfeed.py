# -*- coding: utf-8 -*-
"""Slater-Koster integral feed objects.

This contains all Slater-Koster integral feed objects. These objects are
responsible for generating the Slater-Koster integrals used in constructing
Hamiltonian and overlap matrices. The on-site and off-site terms are yielded
by the `on_site` and `off_site` class methods respectively.
"""
import os
from typing import Union, Tuple, Literal, Optional, List, Dict
from abc import ABC
from inspect import getfullargspec
from warnings import warn
import numpy as np
import torch
from torch import Tensor
from scipy.interpolate import CubicSpline
from slakonet.atoms import Geometry
from slakonet.basis import Basis
from slakonet.utils import pack
from slakonet.atoms import (
    unique_atom_pairs,
    batch_chemical_symbols,
)
from slakonet.interpolation import (
    PolyInterpU,
)
from slakonet.skf import Skf
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class _SkFeed(ABC):
    """ABC for objects responsible for supplying Slater-Koster integrals.

    Subclasses of the this abstract base class are responsible for supplying
    the Slater-Koster integrals needed to construct the Hamiltonian & overlap
    matrices.

    Arguments:
        device: Device on which the `SkFeed` object and its contents resides.
        dtype: Floating point dtype used by `SkFeed` object.

    Developers Notes:
        This class provides a common fabric upon which all Slater-Koster
        integral feed objects are built. As the `_SkFeed` class is in its
        infancy it is subject to change; e.g. the addition of an `update`
        method which allows relevant model variables to be updated via a
        single call during backpropagation.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        # These are static, private variables and must NEVER be modified!
        self.__device = device
        self.__dtype = dtype

    def __init_subclass__(cls, check_sig: bool = True):
        """Check the signature of subclasses' methods.

        Issues non-fatal warnings if invalid signatures are detected in
        subclasses' `off_site` or `on_site` methods. Both methods must accept
        an arbitrary number of keyword arguments, i.e. `**kwargs`. The
        `off_site` & `on_site` method must take the keyword arguments
        (atom_pair, shell_pair, distances) and (atomic_numbers) respectively.

        This behaviour is enforced to maintain consistency between the various
        subclasses of `_SkFeed`'; which is necessary as the various subclasses
        will likely differ significantly from one another & may become quite
        complex.

        Arguments:
            check_sig: Signature check not performed if ``check_sig = False``.
                This offers a way to override these warnings if needed.
        """

        def check(func, has_args):
            sig = getfullargspec(func)
            name = func.__qualname__
            if check_sig:  # This check can be skipped
                missing = ", ".join(has_args - set(sig.args))
                if len(missing) != 0:
                    warn(
                        f'Signature Warning: keyword argument(s) "{missing}"'
                        f' missing from method "{name}"',
                        stacklevel=4,
                    )

            if sig.varkw is None:  # This check cannot be skipped
                warn(
                    f'Signature Warning: method "{name}" must accept an '
                    f"arbitrary keyword arguments, i.e. **kwargs.",
                    stacklevel=4,
                )

        check(cls.off_site, {"atom_pair", "shell_pair", "distances"})
        check(cls.on_site, {"atomic_numbers"})

    @property
    def device(self) -> torch.device:
        """The device on which the geometry object resides."""
        return self.__device

    @device.setter
    def device(self, value):
        # Instruct users to use the ".to" method if wanting to change device.
        name = self.__class__.__name__
        raise AttributeError(
            f"{name} object's dtype can only be modified "
            'via the ".to" method.'
        )

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by geometry object."""
        return self.__dtype

    # @abstractmethod
    def off_site(
        self,
        atom_pair: Tensor,
        shell_pair: Tensor,
        distances: Tensor,
        # **kwargs,
    ) -> Tensor:
        """Evaluate the selected off-site Slater-Koster integrals.

        This evaluates & returns the off-site Slater-Koster integrals between
        orbitals `l_pair` on atoms `atom_pair` at the distances specified by
        `distances`. Note that only one `atom_pair` & `shell_pair` can be
        evaluated at at time. The dimensionality of the the returned tensor
        depends on the number of distances evaluated & the number of bonding
        integrals associated with the interaction.

        Arguments:
            atom_pair: Atomic numbers of the associated atoms.
            shell_pair: Shell numbers associated with the interaction.
            distances: Distances between the atoms pairs.

        Keyword Arguments:
            atom_indices: Tensor: The indices of the atoms associated with the
                 ``distances`` specified. This is automatically passed in by
                 the Slater-Koster transformation code.

        Return:
            integrals: Off-site integrals between orbitals ``shell_pair`` on
                atoms ``atom_pair`` at the specified distances.

        Developers Notes:
            The Slater-Koster transformation passes "atom_pair", "shell_pair",
            & "distances" as keyword arguments. This avoids having to change
            the Slater-Koster transformation code every time a new feed is
            created. These four arguments were made default as they will be
            required by most Slater-Koster feed implementations. A warning
            will be issued if a `_SkFeed` subclass is found to be missing any
            of these arguments. However, this behaviour can be suppressed by
            adding the class argument `check_sig=False`.

            It is imperative that this method accepts an arbitrary number of
            keyword arguments, i.e. has a `**kwarg` argument. This allows for
            additional data to be passed in. By default the Slater-Koster
            transformation code will add the keyword argument "atom_indices".
            This specifies the indices of the atoms involved, which is useful
            if the feed takes into account environmental dependency.

            Any number of additional arguments can be added to this method.
            However, to get the Slater-Koster transform code to pass this
            information through one must pass the requisite data as keyword
            arguments to the Slater-Koster transform function itself. As it
            will pass through any keyword arguments it encounters.

        """
        pass

    # @abstractmethod
    def on_site(self, atomic_numbers: Tensor) -> Tuple[Tensor, ...]:
        # def on_site(self, atomic_numbers: Tensor, **kwargs) -> Tuple[Tensor, ...]:
        """Returns the specified on-site terms.

        Arguments:
            atomic_numbers: Atomic numbers for which on-site terms should be
                returned.

        Keyword Arguments:
            atom_indices: Tensor: The indices of the atoms associated with the
                 ``distances`` specified. This is automatically passed in by
                 the Slater-Koster transformation code.

        Returns:
            on_sites: Tuple of on-site term tensors, one for each atom in
                ``atomic_numbers``.

        Developers Notes:
            See the documentation for the _SkFeed.off_site method for
            more information.

        """
        pass

    # @abstractmethod
    def to(self, device: torch.device) -> "SkFeed":
        """Returns a copy of the `SkFeed` instance on the specified device.
        This method creates and returns a new copy of the `SkFeed` instance
        on the specified device "``device``".
        Arguments:
            device: Device on which the clone should be placed.
        Returns:
            sk_feed: A copy of the `SkFeed` instance placed on the specified
                device.
        Notes:
            If the `SkFeed` instance is already on the desired device then
            `self` will be returned.
        """
        pass

    def gen_onsite(
        self, geometry: Geometry, basis: Basis, orbital_resolved=False
    ):
        self.on_site_dict["ml_onsite"] = _expand_onsite(
            self.on_site_dict, geometry, basis, orbital_resolved
        )
        self.orbital_resolved = orbital_resolved


class SkfFeed(_SkFeed):
    """This is the standardard method to supply Slater-Koster integral feeds.

    The standard suggests that the feeds are similar to the method in DFTB+.
    The `from_dir` function can be used to read normal Slater-Koster files
    and return Hamiltonian and overlap feeds separatedly in default.

    Arguments:
        off_site_dict: Collections of off-site data as off-site feeds of
            Hamiltonian or overlap.
        on_site_dict: Collections of on-site data as on-site feeds of
            Hamiltonian or overlap.

    Attributes:
        off_site_dict: Collections of off-site data as off-site feeds of
            Hamiltonian or overlap.
        on_site_dict: Collections of on-site data as on-site feeds of
            Hamiltonian or overlap.

    """

    def __init__(
        self,
        off_site_dict: dict,
        on_site_dict: dict,
        shell_dict: dict,
        orbital_resolve=False,
        # **kwargs, #TODO
    ):
        self.off_site_dict = off_site_dict
        self.on_site_dict = on_site_dict
        self.shell_dict = shell_dict
        self.orbital_resolve = (
            orbital_resolve  # kwargs.get("orbital_resolve", False)
        )

    @classmethod
    def from_dict(self, info):
        off_site_dict = info["off_site_dict"]  # info['hs_dict']
        on_site_hs_dict = info["on_site_dict"]
        shell_dict = info["shell_dict"]
        orbital_resolve = info["orbital_resolve"]
        print(info)
        return SkfFeed(
            off_site_dict,
            on_site_hs_dict,
            shell_dict,
            orbital_resolve=orbital_resolve,
        )  # , **kwargs)

    def to_dict(self):
        info = {}
        info["shell_dict"] = self.shell_dict
        info["on_site_dict"] = self.on_site_dict
        info["off_site_dict"] = self.off_site_dict
        info["orbital_resolve"] = self.orbital_resolve
        return info

    def off_site(
        self,
        atom_pair: Tensor,
        shell_pair: Tensor,
        distances: Tensor,
        g_compr=None,
        # **kwargs,
    ) -> Tensor:
        """Get integrals for given geometrys.

        Arguments:
            distances: distances of single & multi systems.
            atom_pair: skf files type. Support normal skf, h5py binary skf.
            l_pair: The quantum number ℓ pairs.
            ski_type: Type of integral, H or S.

        Keyword Args:
            orbital_resolve: If each orbital is resolved for U.
            abcd: abcd parameters in cubic spline method.

        Returns:
            integral: Getting integral in SKF tables with given atom pair, ℓ
                number pair, distance, or compression radii pair.
        """
        # if kwargs.get("orbital_resolve", False):
        #    raise NotImplementedError("Not implement orbital resolved U.")
        # g_compr = kwargs.get("g_compr", None)

        splines = self.off_site_dict[
            (*atom_pair.tolist(), *shell_pair.tolist())
        ]

        # call the interpolator
        if g_compr is None:
            integral = splines(distances)
        else:
            integral = splines(g_compr[0], g_compr[0], distances)

        if isinstance(integral[0], np.ndarray):
            integral = torch.from_numpy(integral)

        return integral

    def on_site(self, atomic_numbers: Tensor) -> List[Tensor]:
        # def on_site(self, atomic_numbers: Tensor, **kwargs) -> List[Tensor]:
        """Returns the specified on-site terms.

        In sktable dictionary, s, p, d and f orbitals of original on-site
        from Slater-Koster files have been expanded one, three, five and
        seven times in default. The expansion of on-site could be controlled
        by `orbital_expand` when loading Slater-Koster integral tables. The
        output on-site size relies on the defined `max_ls`.

        Arguments:
            atomic_numbers: Atomic numbers for which on-site terms should be
                returned.
        max_ls: A dictionary specifying the maximum permitted angular momentum
            associated with a each atomic number. keys must be integers not
            torch tensors.

        Returns:
            on_sites: Tuple of on-site term tensors, one for each atom in
                `atomic_numbers`.

        """
        if not self.orbital_resolve:
            return [
                self.on_site_dict[(ian.tolist())] for ian in atomic_numbers
            ]
        else:
            return [
                self.on_site_dict[(ian.tolist(), il)]
                for ian in atomic_numbers
                for il in range(max(self.shell_dict[int(ian)]) + 1)
            ]


class SkfParamFeed:
    """This is a standard Slater-Koster feed except Hamiltonian and overlap.

    The standard suggests that the feeds are similar to the method in DFTB+.
    The `from_dir` function can be used to read normal Slater-Koster files
    and return repulsive and the first two or three lines data in Slater-Koster
    files.

    Arguments:
        sktable_dict: Dictionary contains SKF integral tables.
        geometry: Geometry object.

    Attributes:
        sktable_dict: Dictionary contains SKF integral tables.
        geometry: Geometry object.

    """

    def __init__(self, sktable_dict: dict, geometry: Geometry, elements):
        self.sktable_dict = sktable_dict
        self.geometry = geometry
        self.elements = elements

    @property
    def U(self) -> Tensor:
        """Return Hubbart U for current geometry."""
        U = torch.zeros(self.geometry.atomic_numbers.shape)
        for inum in self.geometry.unique_atomic_numbers():
            mask = self.geometry.atomic_numbers == inum
            U[mask] = self.sktable_dict[(inum.tolist(), "U")]
        return U

    @property
    def mass(self):
        """Return atomic mass for each atom in current geometry."""
        mass = torch.zeros(self.geometry.atomic_numbers.shape)
        for inum in self.geometry.unique_atomic_numbers():
            mask = self.geometry.atomic_numbers == inum
            mass[mask] = self.sktable_dict[(inum.tolist(), "mass")]
        return mass

    @property
    def qzero(self):
        """Return atomic charge for each atom in current geometry."""
        qzero = torch.zeros(self.geometry.atomic_numbers.shape)
        for inum in self.geometry.unique_atomic_numbers():
            mask = self.geometry.atomic_numbers == inum
            qzero[mask] = self.sktable_dict[
                (inum.tolist(), "occupations")
            ].sum()
        return qzero

    @property
    def cutoff(self):
        """Return max cutoff for each atom pair in Slater-Koster tables."""
        if self.geometry is not None:
            element_pair = unique_atom_pairs(geometry=self.geometry)
        else:
            element_pair = unique_atom_pairs(elements=self.elements)

        hs_cut = 0.0
        for ii, ielement in enumerate(element_pair):
            icut = self.sktable_dict[(*ielement.tolist(), "hs_cut")]
            hs_cut = icut if icut > hs_cut else hs_cut
        return hs_cut


def _path_to_skf(path, element, is_dir):
    """Return the joint path to the skf file or binary file."""
    if not is_dir:
        return path
    else:
        return os.path.join(path, element[0] + "-" + element[1] + ".skf")


def _get_homo_dict(
    sktable_dict: dict, skf: object, orbital_resolve=False
) -> dict:
    # def _get_homo_dict(sktable_dict: dict, skf: object, **kwargs) -> dict:
    """Write onsite, Hubbert U and other homo parameters into dict."""
    # if kwargs.get("orbital_resolve", False):
    #    raise NotImplementedError("Not implement orbital resolved Hubbert U.")

    assert skf.atom_pair[0] == skf.atom_pair[1]

    # return Hubbert U without orbital resolve
    sktable_dict[(skf.atom_pair[0].tolist(), "U")] = skf.hubbard_us.unsqueeze(
        1
    )[-1]
    sktable_dict[(skf.atom_pair[0].tolist(), "occupations")] = skf.occupations

    return sktable_dict


def _get_dict(sk_dict: dict, skf: Skf, repulsive) -> Tuple[dict, dict]:
    """Get Hamiltonian or overlap tables for each orbital interaction.

    Arguments:
        sk_dict: Dictionary for all other data except Hamiltonian or
            overlap.
        skf: Object with original SKF integrals data.

    Returns:
        h_dict: Dictionary with updated Hamiltonian tables.
        s_dict: Dictionary with updated overlap tables.

    """
    if repulsive:
        sk_dict[(*skf.atom_pair.tolist(), "grid")] = skf.r_spline.grid
        sk_dict[(*skf.atom_pair.tolist(), "long_grid")] = torch.stack(
            [skf.r_spline.grid[-1], skf.r_spline.cutoff]
        )
        sk_dict[(*skf.atom_pair.tolist(), "rep_cut")] = skf.r_spline.cutoff
        sk_dict[(*skf.atom_pair.tolist(), "spline_coef")] = (
            skf.r_spline.spline_coef
        )
        sk_dict[(*skf.atom_pair.tolist(), "exp_coef")] = skf.r_spline.exp_coef
        sk_dict[(*skf.atom_pair.tolist(), "tail_coef")] = (
            skf.r_spline.tail_coef
        )

    if skf.atom_pair[0] == skf.atom_pair[1]:
        sk_dict[(skf.atom_pair[0].tolist(), "occupations")] = skf.occupations
        sk_dict[(skf.atom_pair[0].tolist(), "hubbard_us")] = skf.hubbard_us
        sk_dict[(skf.atom_pair[0].tolist(), "mass")] = skf.mass

    return sk_dict


def _get_hs_dict(
    hs_dict: dict,
    interpolator: object,
    skf: object,
    skf_type: str,
    vcr=None,
    tvcr=None,
    pair=None,
    build_abcd=False,
    # **kwargs,
) -> Tuple[dict, dict]:
    """Get Hamiltonian or overlap tables for each orbital interaction.

    Arguments:
        h_dict: Hamiltonian tables dictionary.
        s_dict: Overlap tables dictionary.
        interpolator: Slater-Koster interpolation method.
        interactions: Orbital interactions, e.g. (0, 0, 0) for ss0 orbital.
        skf: Object with original SKF integrals data.

    Returns:
        h_dict: Dictionary with updated Hamiltonian tables.
        s_dict: Dictionary with updated overlap tables.

    """
    # build_abcd = kwargs.get("build_abcd", False)
    hs_data = (
        getattr(skf, "hamiltonian")
        if skf_type == "H"
        else getattr(skf, "overlap")
    )

    for ii, interaction in enumerate(hs_data.keys()):

        # Standard interpolation
        if vcr is None and tvcr is None:
            hs_dict[(*skf.atom_pair.tolist(), *interaction)] = interpolator(
                skf.grid, hs_data[interaction].T
            )

            # write spline parameters into list
            if build_abcd:
                hs_dict["variable"].append(
                    hs_dict[
                        (*skf.atom_pair.tolist(), *interaction)
                    ].abcd.requires_grad_(build_abcd)
                )

        # with one extra varible for each element atom
        elif vcr is not None:
            val = torch.stack([vcr[pair[0].tolist()], vcr[pair[1].tolist()]]).T
            hs_dict[(*skf.atom_pair.tolist(), *interaction)] = interpolator(
                val, hs_data[interaction], skf.grid
            )

        # with two extra varibles for each element atom
        elif tvcr is not None:
            hs_dict[(*skf.atom_pair.tolist(), *interaction)] = interpolator(
                hs_data[interaction].permute(-2, 0, 1, 2, 3, -1),
                tvcr,
                tvcr,
                tvcr,
                tvcr,
                skf.grid,
            )

    return hs_dict


def _get_hs_dict(
    hs_dict: dict,
    interpolator: object,
    skf: object,
    skf_type: str,
    vcr=None,
    tvcr=None,
    pair=None,
    build_abcd=False,
    # **kwargs,
) -> Tuple[dict, dict]:
    """Get Hamiltonian or overlap tables for each orbital interaction.

    Arguments:
        h_dict: Hamiltonian tables dictionary.
        s_dict: Overlap tables dictionary.
        interpolator: Slater-Koster interpolation method.
        interactions: Orbital interactions, e.g. (0, 0, 0) for ss0 orbital.
        skf: Object with original SKF integrals data.

    Returns:
        h_dict: Dictionary with updated Hamiltonian tables.
        s_dict: Dictionary with updated overlap tables.

    """
    # build_abcd = kwargs.get("build_abcd", False)
    hs_data = (
        getattr(skf, "hamiltonian")
        if skf_type == "H"
        else getattr(skf, "overlap")
    )

    for ii, interaction in enumerate(hs_data.keys()):

        # Standard interpolation
        if vcr is None and tvcr is None:
            hs_dict[(*skf.atom_pair.tolist(), *interaction)] = interpolator(
                skf.grid, hs_data[interaction].T
            )

            # write spline parameters into list
            if build_abcd:
                hs_dict["variable"].append(
                    hs_dict[
                        (*skf.atom_pair.tolist(), *interaction)
                    ].abcd.requires_grad_(build_abcd)
                )

        # with one extra varible for each element atom
        elif vcr is not None:
            val = torch.stack([vcr[pair[0].tolist()], vcr[pair[1].tolist()]]).T
            hs_dict[(*skf.atom_pair.tolist(), *interaction)] = interpolator(
                val, hs_data[interaction], skf.grid
            )

        # with two extra varibles for each element atom
        elif tvcr is not None:
            hs_dict[(*skf.atom_pair.tolist(), *interaction)] = interpolator(
                hs_data[interaction].permute(-2, 0, 1, 2, 3, -1),
                tvcr,
                tvcr,
                tvcr,
                tvcr,
                skf.grid,
            )

    return hs_dict


def _get_onsite_dict(
    onsite_hs_dict: dict,
    skf: object,
    shell_dict,
    integral_type,
    orbital_resolve=False,
    # onsite_hs_dict: dict, skf: object, shell_dict, integral_type, **kwargs
) -> Tuple[dict, dict]:
    """Write on-site of Hamiltonian or overlap.

    In default, the on-site has been expanded according to orbital shell.
    If maximum orbital from SKF fikes is d, the original s, p and d on-site
    will repeat one, three and five times.

    Arguments:
        onsite_h_dict: Hamiltonian onsite dictionary.
        onsite_s_dict: Overlap onsite dictionary.
        skf: Object with original SKF data.
        max_l: An integer specifying the maximum permitted angular momentum
            associated with the specific atomic number in this function.
        onsite_h_feed: If True, return Hamiltonian onsite, else return dict
            `onsite_h_dict` with any changes.
        onsite_s_feed: If True, return overlap onsite, else return dict
            `onsite_s_dict` with any changes.

    Returns:
        onsite_h_dict: Hamiltonian onsite dictionary.
        onsite_s_dict: Overlap onsite dictionary.

    """
    # get index to expand homo parameters according to the orbitals
    max_l = max(shell_dict[int(skf.atom_pair[0])])
    # orbital_resolve = kwargs.get("orbital_resolve", False)
    if orbital_resolve:
        orb_index = [(ii + 1) ** 2 - ii**2 for ii in range(max_l + 1)]
    else:
        orb_index = [1] * (max_l + 1)

    # flip make sure the order is along s, p ...
    if integral_type == "H" and not orbital_resolve:
        onsite_hs_dict[(skf.atom_pair[0].tolist())] = torch.cat(
            [
                isk.repeat(ioi)
                for ioi, isk in zip(orb_index, skf.on_sites[: max_l + 1])
            ]
        )

    elif integral_type == "H" and orbital_resolve:
        for il, isk, ioi in zip(
            range(max_l + 1), skf.on_sites[: max_l + 1], orb_index
        ):
            onsite_hs_dict[(skf.atom_pair[0].tolist(), il)] = isk.repeat(ioi)

    elif integral_type == "S" and not orbital_resolve:
        onsite_hs_dict[(skf.atom_pair[0].tolist())] = torch.cat(
            [torch.ones(ioi) for ioi in orb_index]
        )

    elif integral_type == "S" and orbital_resolve:
        for il in range(max_l + 1):
            onsite_hs_dict[(skf.atom_pair[0].tolist(), il)] = torch.ones(
                int(2 * il + 1)
            )

    return onsite_hs_dict


# Type alias to improve PEP484 readability
SkFeed = _SkFeed
