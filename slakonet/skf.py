"""Methods for reading from and writing to skf and associated files."""

# Adopted from tbmalt
import re
import warnings
from dataclasses import dataclass
from os.path import isfile, split, splitext, basename
from time import time
from typing import List, Tuple, Union, Dict, Sequence, Any, Optional
from itertools import product
import h5py
import numpy as np
import torch
from h5py import Group
from torch import Tensor
from slakonet.elements import chemical_symbols
from slakonet.elements import atomic_numbers
from slakonet.utils import pack, triangular_root, tetrahedral_root
from slakonet.interpolation import poly_to_zero, vcr_poly_to_zero

OptTens = Optional[Tensor]
SkDict = Dict[Tuple[int, int], Tensor]


class Skf:
    r"""Slater-Koster file parser.

    This class handles the parsing of DFTB+ skf formatted Slater-Koster files,
    and their binary analogs. Data can be read from and saved to files using
    the `read` & `write` methods. Reading a file will return an `Skf` instance
    holding all stored data.

    Arguments:
         atom_pair: Atomic numbers of the elements associated with the
            interaction.
         hamiltonian: Dictionary keyed by azimuthal-number-pairs (ℓ₁, ℓ₂) and
            valued by m×d Hamiltonian integral tensors; where m and d iterate
            over bond-order (σ, π, etc.) and distances respectively.
         overlap: Dictionary storing the overlap integrals in a manner akin to
            ``hamiltonian``.
         grid: Distances at which the ``hamiltonian`` & ``overlap`` elements
            were evaluated.
        hs_cutoff:
         r_spline: A :class:`.RSpline` object detailing the repulsive
            spline. [DEFAULT=None]
         r_poly: A :class:`.RPoly` object detailing the repulsive
            polynomial. [DEFAULT=None]
         on_sites: On site terms, homo-atomic systems only. [DEFAULT=None]
         hubbard_us: Hubbard U terms, homo-atomic systems only. [DEFAULT=None]
         mass: Atomic mass, homo-atomic systems only. [DEFAULT=None]
         occupations: Occupations of the orbitals, homo-atomic systems only.
            [DEFAULT=None]

    Attributes:
        atomic: True if the system contains atomic data, only relevant to the
            homo-atomic cases.

    .. _Notes:
    Notes:
        HOMO atomic systems commonly, but not always, include additional
        "atomic" data; namely atomic mass, on-site terms, occupations, and
        the Hubbard-U values. These can be optionally specified using the
        ``mass``, ``on_sites``, ``occupations``, and ``hubbard_us`` attributes
        respectively. However, these attributes are mutually inclusive, i.e.
        either all are specified or none are. Furthermore, values contained
        within such tensors should be ordered from lowest azimuthal number
        to highest, where applicable.

        Further information regarding the skf file format specification can be
        found in the document: "`Format of the v1.0 Slater-Koster Files`_".

    Warnings:
        This may fail to parse files which do not strictly adhere to the skf
        file format. Some skf files, such as those from the "pbc" parameter
        set, contain non-trivial errors in them, e.g. incorrectly specified
        number of grid points. Such files require fixing before they can be
        read in.

        The ``atom_pair`` argument is order sensitive, i.e. [6, 7] ≠ [7, 6].
        For example, the p-orbital of the s-p-σ interaction would be located
        on N when ``atom_pair`` is [6, 7] but on C when it is [7, 6].

    Raises:
        ValueError: if some but not all atomic attributes are specified. See
            the :ref:`Notes` section for more details.

    .. _Format of the v1.0 Slater-Koster Files:
        https://dftb.org/fileadmin/DFTB/public/misc/slakoformat.pdf

    """

    # Used to reorder hamiltonian and overlap data read in from skf files.
    _sorter = [9, 8, 7, 5, 6, 3, 4, 0, 1, 2]
    _sorter_e = [
        19,
        18,
        17,
        16,
        14,
        15,
        12,
        13,
        10,
        11,
        7,
        8,
        9,
        4,
        5,
        6,
        0,
        1,
        2,
        3,
    ]

    # Dataclasses for holding the repulsive interaction data.
    @dataclass
    class RPoly:
        """Dataclass container for the repulsive polynomial.

        Arguments:
            cutoff: Cutoff radius of the repulsive interaction.
            coef: The eight polynomial coefficients (c2-c9).
        """

        cutoff: Tensor
        coef: Tensor

    @dataclass
    class RSpline:
        """Dataclass container for the repulsive spline.

        Arguments:
            grid: Distance for the primary spline segments.
            cutoff: Cutoff radius for the tail spline.
            spline_coef: The primary spline's Coefficients (four per segment).
            exp_coef: The exponential expression's coefficients a1, a2 & a3.
            tail_coef: The six coefficients of the terminal tail spline.

        """

        grid: Tensor
        cutoff: Tensor
        spline_coef: Tensor
        exp_coef: Tensor
        tail_coef: Tensor

    # HDF5-SK version number. Updated when introducing a change that would
    # break backwards compatibility with previously created HDF5-skf file.
    version = "0.1"

    def __init__(
        self,
        atom_pair: Tensor,
        hamiltonian: SkDict,
        overlap: SkDict,
        grid: Tensor,
        hs_cut,
        r_spline: Optional[RSpline] = None,
        r_poly: Optional[RPoly] = None,
        hubbard_us: OptTens = None,
        on_sites: OptTens = None,
        occupations: OptTens = None,
        mass: OptTens = None,
    ):

        self.atom_pair = atom_pair

        # SkDict attributes
        self.hamiltonian = hamiltonian
        self.overlap = overlap
        self.grid = grid
        self.hs_cutoff = hs_cut

        # Ensure grid is uniformly spaced
        if not (grid.diff().diff().abs() < 1e-5).all():
            raise ValueError("Electronic integral grid spacing is not uniform")

        # Repulsive attributes
        self.r_spline = r_spline
        self.r_poly = r_poly

        # Either the system contains atomic information or it does not; it is
        # illogical to have some atomic attributes but not others.
        check = [
            i is not None for i in [on_sites, hubbard_us, occupations, mass]
        ]
        if all(check) != any(check):
            raise ValueError(
                "Either all or no atomic attributes must be supplied:"
                "\n\t- on_sites\n\t- hubbard_us\n\t- mass\n\t- occupations"
            )

        # Atomic attributes
        self.atomic: bool = all(check)
        self.on_sites = on_sites
        self.hubbard_us = hubbard_us
        self.mass = mass
        self.occupations = occupations

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Skf instance to a dictionary representation.

        Returns:
            dict: Dictionary containing all the Skf data that can be serialized.
        """

        def tensor_to_serializable(obj):
            """Convert tensors to serializable format (lists)."""
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            elif isinstance(obj, dict):
                return {k: tensor_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [tensor_to_serializable(item) for item in obj]
            else:
                return obj

        # Build the base dictionary
        data = {
            "atom_pair": tensor_to_serializable(self.atom_pair),
            "hamiltonian": {
                f"{k[0]}-{k[1]}": tensor_to_serializable(v)
                for k, v in self.hamiltonian.items()
            },
            "overlap": {
                f"{k[0]}-{k[1]}": tensor_to_serializable(v)
                for k, v in self.overlap.items()
            },
            "grid": tensor_to_serializable(self.grid),
            "hs_cutoff": tensor_to_serializable(self.hs_cutoff),
            "atomic": self.atomic,
        }

        # Add repulsive spline data if present
        if self.r_spline is not None:
            data["r_spline"] = {
                "grid": tensor_to_serializable(self.r_spline.grid),
                "cutoff": tensor_to_serializable(self.r_spline.cutoff),
                "spline_coef": tensor_to_serializable(
                    self.r_spline.spline_coef
                ),
                "exp_coef": tensor_to_serializable(self.r_spline.exp_coef),
                "tail_coef": tensor_to_serializable(self.r_spline.tail_coef),
            }

        # Add repulsive polynomial data if present
        if self.r_poly is not None:
            data["r_poly"] = {
                "cutoff": tensor_to_serializable(self.r_poly.cutoff),
                "coef": tensor_to_serializable(self.r_poly.coef),
            }

        # Add atomic data if present
        if self.atomic:
            data["atomic_data"] = {
                "on_sites": tensor_to_serializable(self.on_sites),
                "hubbard_us": tensor_to_serializable(self.hubbard_us),
                "mass": tensor_to_serializable(self.mass),
                "occupations": tensor_to_serializable(self.occupations),
            }

        return data

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "Skf":
        """Create an Skf instance from a dictionary representation.

        Arguments:
            data: Dictionary containing the Skf data.
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]
            device: Device on which to place tensors. [DEFAULT=None]

        Returns:
            skf: The resulting Skf object.
        """
        dd = {"dtype": dtype, "device": device}

        def to_tensor(obj):
            """Convert serializable data back to tensors."""
            if isinstance(obj, list):
                return torch.tensor(obj, **dd)
            else:
                return obj

        # Parse basic attributes
        atom_pair = torch.tensor(data["atom_pair"], dtype=torch.long)

        # Parse hamiltonian and overlap dictionaries
        hamiltonian = {
            tuple(map(int, k.split("-"))): to_tensor(v)
            for k, v in data["hamiltonian"].items()
        }
        overlap = {
            tuple(map(int, k.split("-"))): to_tensor(v)
            for k, v in data["overlap"].items()
        }

        grid = to_tensor(data["grid"])
        hs_cutoff = (
            to_tensor(data["hs_cutoff"])
            if isinstance(data["hs_cutoff"], list)
            else data["hs_cutoff"]
        )

        # Initialize kwargs for constructor
        init_kwargs = {}

        # Parse repulsive spline if present
        if "r_spline" in data:
            rs_data = data["r_spline"]
            init_kwargs["r_spline"] = cls.RSpline(
                grid=to_tensor(rs_data["grid"]),
                cutoff=to_tensor(rs_data["cutoff"]),
                spline_coef=to_tensor(rs_data["spline_coef"]),
                exp_coef=to_tensor(rs_data["exp_coef"]),
                tail_coef=to_tensor(rs_data["tail_coef"]),
            )

        # Parse repulsive polynomial if present
        if "r_poly" in data:
            rp_data = data["r_poly"]
            init_kwargs["r_poly"] = cls.RPoly(
                cutoff=to_tensor(rp_data["cutoff"]),
                coef=to_tensor(rp_data["coef"]),
            )

        # Parse atomic data if present
        if data.get("atomic", False) and "atomic_data" in data:
            atomic_data = data["atomic_data"]
            init_kwargs.update(
                {
                    "on_sites": to_tensor(atomic_data["on_sites"]),
                    "hubbard_us": to_tensor(atomic_data["hubbard_us"]),
                    "mass": to_tensor(atomic_data["mass"]),
                    "occupations": to_tensor(atomic_data["occupations"]),
                }
            )

        return cls(
            atom_pair=atom_pair,
            hamiltonian=hamiltonian,
            overlap=overlap,
            grid=grid,
            hs_cut=hs_cutoff,
            **init_kwargs,
        )

    @classmethod
    def read(
        cls, path: str, atom_pair: Sequence[int] = None, **kwargs
    ) -> "Skf":
        """Parse Slater-Koster data from skf files and their binary analogs.

        Arguments:
            path: Path to the file that is to be read (.skf or .hdf5).
            atom_pair: Atomic numbers of the element pair. This is only used
                when reading from an HDF5 file with more than one SK entry.
                [DEFAULT=None]

        Keyword Arguments:
            device: Device on which to place tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]

        Returns:
            skf: `Skf` object contain all data parsed from the specified file.
        """
        if not isfile(path):  # Verify the target file exists
            raise FileNotFoundError(f"Could not find: {path}")

        if "sk" in splitext(path)[1].lower():  # If path points to an skf file
            # Issue a waring if the user specifies `atom_pair` for an skf file
            if atom_pair is not None:
                warnings.warn(
                    '"atom_pair" argument is only used when reading'
                    "from HDF5 files with multiple SK entries."
                )

            return cls.from_skf(path, **kwargs)

        with h5py.File(path, "r") as db:  # Otherwise must be an hdf5 database
            # If atom_pair is specified use this to identify the target
            # name = '-'.join([chemical_symbols[int[i]] for i in atom_pair])
            if atom_pair is not None:
                name = "-".join([chemical_symbols[int(i)] for i in atom_pair])
            else:
                # Otherwise scan for valid entries: if only 1 SK entry exists
                # then assume it's the target; if multiple entries exist then
                # it's impossible to know which the user wanted.
                e = "[A-Z][a-z]*"
                entries = [k for k in db if re.fullmatch(f"{e}-{e}", k)]
                if len(entries) == 1:
                    name = entries[0]
                else:
                    raise ValueError(
                        "Use atom_pair when database have "
                        f"more than one entry: {basename(path)}"
                    )

            return cls.from_hdf5(db[name], **kwargs)

    @classmethod
    def from_skf(
        cls,
        path: str,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        smooth_to_zero: bool = True,
        **kwargs,
    ) -> "Skf":
        """Parse and skf file into an `Skf` instance.

        File names should follow the naming convention X-Y.skf where X & Y are
        the chemical symbols of the associated elements. However, any file
        which **ends** in X.Y will be successfully parsed (where "." is any
        character (including no character)).

        Arguments:
            path: Path to the target skf file.
            device: Device on which to place tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]

        Returns:
            skf: Return the arguments in `Skf` object.

        """
        dd = {"dtype": dtype, "device": device}
        kwargs_in = {}
        if not smooth_to_zero:
            warnings.warn(
                "`smooth_to_zero` is set as Flase, and there is no"
                " tail smoothing in interpolation, please make sure"
                " you understand what you have done."
            )

        # Identify the elements involved according to the file name
        e = "[A-Z][a-z]?"
        try:
            atom_pair = torch.tensor(
                [
                    atomic_numbers[i]
                    for i in re.findall(
                        e,
                        re.search(rf"{e}.?{e}(?=.)", split(path)[-1]).group(0),
                    )
                ]
            )
        except AttributeError as error:
            raise ValueError(
                "Could not parse element names form file."
            ) from error

        lines = open(path, "r").readlines()

        # Remove the comment line if present
        lines = lines[1:] if lines[0].startswith("@") else lines

        # 0th line, grid distance and grid points number
        g_step, n_grids = lines[0].replace(",", " ").split()[:2]
        g_step, n_grids = float(g_step), int(n_grids)
        grid = torch.arange(1, n_grids + 1, **dd) * g_step
        hs_cut = n_grids * g_step - g_step

        # Determine if this is the homo/atomic case (from the file's contents)
        atomic = len(atom_ln := _s2t(_esr(lines[1]), **dd)) in [10, 13]

        # Read in the mass and polynomial repulsion coefficients
        mass, r_poly, r_cut = _s2t(_esr(lines[1 + atomic]), **dd)[:10].split(
            [1, 8, 1]
        )

        # If polynomial coefficients are valid, create an r_poly object
        if (r_poly != 0).any():
            kwargs_in["r_poly"] = cls.RPoly(r_cut, r_poly)

        # Parse hamiltonian/overlap integrals.
        h_data, s_data = (
            _s2t(
                _esr("  ".join(lines[2 + atomic : 2 + atomic + n_grids])), **dd
            )
            .view(n_grids, -1)
            .chunk(2, 1)
        )

        # H/S tables are reordered so the lowest l comes first, broken up into
        # into shell-pair chunks, e.g. ss, sp, sd, pp, etc, before finally
        # being placed into dictionaries.
        count = h_data.shape[-1]
        sort = cls._sorter if count == 10 else cls._sorter_e  # ◂──────┐
        max_l = round(tetrahedral_root(count) - 1)  # ◂─f-orbital catch┘

        # Sort, segmentation and parse the tables into a pair of dictionaries
        l_pairs = torch.triu_indices(max_l + 1, max_l + 1).T
        h_data, s_data = [
            {
                tuple(l_pair.tolist()): integral
                for l_pair, integral in
                #            |   ↓ Sorting ↓   |    ↓ Segmentation by ℓ pair ↓    |
                zip(
                    l_pairs,
                    integrals.T[sort].split((l_pairs[:, 0] + 1).tolist()),
                )
                if not (integral == 0.0).all()
            }  # ← Ignore any dummy interactions
            for integrals in [h_data, s_data]
        ]

        # if smooth_to_zero:
        #     h_data = {ikey: poly_to_zero(
        #         grid, h_data[ikey]) for ikey in h_data.keys()}
        #     s_data = {ikey: poly_to_zero(
        #         grid, s_data[ikey]) for ikey in s_data.keys()}

        if (
            atomic
        ):  # Parse homo data; on-site/Hubbard-U/occupations. (skip spe)
            n = int((len(atom_ln) - 1) / 3)  # -> Number of shells specified
            occs, hubb_u, _, on_site = atom_ln.flip(0).split([n, n, 1, n])
            # If integrals were culled; atomic data must be too.
            max_l = int(triangular_root(len(h_data)) - 1) + 1
            kwargs_in.update(
                {
                    "mass": mass,
                    "occupations": occs[: max_l + 1],
                    "on_sites": on_site[:max_l],
                    "hubbard_us": hubb_u[:max_l],
                }
            )

        # Parse repulsive spline (if present)
        # if "Spline" in lines:
        if "Spline\n" in lines:
            # ln = lines.index("Spline") + 2
            ln = lines.index("Spline\n") + 2
            n_int, r_cutoff = lines[ln - 1].split()
            r_tab = _s2t(lines[ln + 1 : ln + int(n_int)], **dd).view(-1, 6)
            r_grid = torch.cat((r_tab[:, 0], r_tab[None, -1, 1]))
            kwargs_in["r_spline"] = cls.RSpline(
                # Repulsive grid, cutoff & repulsive spline coefficients.
                r_grid,
                torch.tensor(float(r_cutoff), **dd),
                r_tab[:, 2:],
                # The exponential and tail spline's coefficients.
                _s2t(lines[ln], **dd),
                _s2t(lines[ln + int(n_int)], **dd)[2:],
            )

        return cls(atom_pair, h_data, s_data, grid, hs_cut, **kwargs_in)

    @classmethod
    def from_hdf5(
        cls,
        source: Group,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "Skf":
        """Instantiate a `Skf` instances from an HDF5 group.

        Arguments:
            source: An HDF5 group containing slater-koster data.
            device: Device on which to place tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]

        Returns:
            skf: The resulting `Skf` object.
        """

        def tt(group, name):
            """Convenience function to convert to data into tensors"""
            return torch.tensor(group[name][()], **dd)

        dd = {"dtype": dtype, "device": device}

        # Check that the version is compatible
        if float(source.attrs["version"]) > float(cls.version):
            warnings.warn("HDF5-skf file Version exceeds local code version.")

        kwargs = {}
        atom_pair = torch.tensor(source.attrs["atoms"])

        # Retrieve integral data
        ints = source["integrals"]
        # Convert structured numpy arrays into a dictionary of tensors
        H, S = (
            {
                # Convert name-string > Tuple[int, int] & np.array > torch.tensor
                tuple(map(int, n.split("-"))): torch.tensor(i[n], **dd)
                # Loop over field names (strings like "ℓ₁-ℓ₂") & their numpy arrays
                for n in i.dtype.fields.keys()
            }
            for i in [ints["H"], ints["S"]]
        )

        grid = (torch.arange(0, ints["count"][()], **dd) + 1) * ints["step"][
            ()
        ]
        hs_cut = grid[-1] - ints["step"][()]

        if source.attrs["has_r_spline"]:  # Repulsive spline data
            r = source["r_spline"]
            kwargs["r_spline"] = cls.RSpline(
                tt(r, "grid"),
                tt(r, "cutoff"),
                tt(r, "spline_coef"),
                tt(r, "exp_coef"),
                tt(r, "tail_coef"),
            )

        if source.attrs["has_r_poly"]:  # Repulsive polynomial
            r = source["r_poly"]
            kwargs["r_poly"] = cls.RPoly(tt(r, "cutoff"), tt(r, "coef"))

        if source.attrs["is_atomic"]:  # Atomic data
            a = source["atomic"]
            kwargs.update(
                {
                    "on_sites": tt(a, "on_sites"),
                    "hubbard_us": tt(a, "hubbard_us"),
                    "mass": tt(a, "mass"),
                    "occupations": tt(a, "occupations"),
                }
            )

        return cls(atom_pair, H, S, grid, hs_cut, **kwargs)

    def write(self, path: str, overwrite: Optional[bool] = False):
        """Save the Slater-Koster data to a file.

        The target file can be either an skf file or an hdf5 database. Desired
        file format will be inferred from the file's name.

        Arguments:
            path: path to the file in which the data is to be saved.
            overwrite: Existing skf-files/HDF5-groups can only be overwritten
                when ``overwrite`` is True. [DEFAULT=False]

        """
        if "sk" in splitext(path)[1].lower():  # If path points to an skf file
            if isfile(path) and not overwrite:
                raise FileExistsError(
                    'File already exists; use "overwrite" to permit '
                    "overwriting."
                )
            self.to_skf(path)

        else:  # Otherwise it must be an HDF5 file
            with h5py.File(path, "a") as db:  # Create/open the HDF5 file
                name = "-".join(
                    [chemical_symbols[int(i)] for i in self.atom_pair]
                )
                if name in db:  # If an entry already exists in this database
                    if not overwrite:  # Then raise an exception
                        raise FileExistsError(
                            f'Entry {name} already exists; use "overwrite" '
                            "to permit overwriting."
                        )
                    else:  # Unless told to overwrite it
                        del db[name]
                # Create the HDF5 entry & fill it with data via `to_hdf5`.
                self.to_hdf5(db.create_group(name))

    def to_skf(self, path: str):

        def t2a(t):
            """Converts a torch tensor to numpy array."""
            return t.detach().cpu().numpy()

        def a2s(a, f):
            """Converts a numpy array into a formatted string."""
            # Slow but easy way to convert array to string
            if a.ndim == 1:
                return "".join(f"{j:{f}}" for j in a)
            else:
                return "\n".join([a2s(j, f) for j in a])

        # Used for working out array lengths later on
        max_l = max(max(self.hamiltonian)[0], 2)

        # Build the first line defining the integral data's grid.
        # Format: {grid step size} {number of grid points}
        grid_n = len(self.grid)
        grid_step = self.grid.diff()[0]
        output = f"{grid_step:<12.8f}{grid_n:>5}"

        # Parse the atomic data into a string.
        # Format: {on site terms} {SPE} {hubbard u values} {occupancies}
        if self.atomic:
            # Care must be taken when parsing atomic data ase some elements of
            # these arrays may have been culled at read time.
            homo = np.zeros((max_l + 1) * 3)  # Parse in standard atomic data
            for n, i in enumerate(
                [self.occupations, self.hubbard_us, self.on_sites]
            ):
                homo[(start := (max_l + 1) * n) : start + len(i)] = t2a(i)
            # Add dummy SPE value and reverse the array's order
            homo = np.flip(np.insert(homo, (max_l + 1) * 2, 0.0))
            # Finally append the homo data to the output string
            output += "\n" + a2s(homo, ">21.12E")

        # Generate the repulsive polynomial line.
        # Format {mass} {coefficients} {cutoff} {ZEROS}
        coef = np.zeros(7) if self.r_poly is None else t2a(self.r_poly.coef)
        r = np.zeros(1) if self.r_poly is None else t2a(self.r_poly.cutoff)
        mass = t2a(self.mass) if self.atomic else np.zeros(1)
        r_poly_data = np.hstack((mass, coef, r, np.zeros(10)))

        output += "\n" + a2s(r_poly_data, ">21.12E")

        # Build HS data
        ls = range(max_l, -1, -1)
        lps = [i for i in product(ls, ls) if i[0] <= i[1]]
        hs_data = np.hstack(
            [  # Concatenate H & S matrices.
                np.hstack(  # Collate each integral, adding dummy data as needed.
                    [
                        t2a(
                            torch.atleast_2d(
                                i.get(l, torch.zeros(l[0], grid_n))
                            )
                        ).T
                        for l in lps
                    ]
                )
                for i in [self.hamiltonian, self.overlap]
            ]
        )
        output += "\n" + a2s(hs_data, ">21.12E")

        # Append the repulsive spline data, is present.
        if (rs_data := self.r_spline) is not None:
            grid = rs_data.grid
            # Header
            output += "\nSpline"
            # Grid data: {number of grid points} {cutoff}
            output += f"\n{len(grid):<5} {rs_data.cutoff:>12.8f}"
            # Exponential: {coefficients}
            output += "\n" + a2s(rs_data.exp_coef, ">21.12E")
            # Primary spline: {from} {to} {coefficients}
            s_data = t2a(
                torch.cat(
                    (
                        grid[:-1].view(-1, 1),
                        grid[1:].view(-1, 1),
                        rs_data.spline_coef,
                    ),
                    -1,
                )
            )
            output += "\n" + a2s(s_data, ">21.12E")
            # Spline tail: {from} {to} {coefficients}
            tail = t2a(
                torch.cat((grid[-1:], rs_data.cutoff[None], rs_data.tail_coef))
            )
            output += "\n" + a2s(tail, ">21.12E")

        # Write the results to the target file
        open(path, "w").write(output)

    def to_hdf5(self, target: Group):
        """Saves the `Skf` instance into a target HDF5 Group.

        Arguments:
            target: The hdf5 group to which the skf data should be saved.

        Notes:
            This function does not create its own group as it expects that
            ``target`` is the group into which data should be writen.
        """

        def t2n(t: Tensor) -> np.ndarray:
            """Convert torch tensor to a numpy array."""
            return t.detach().cpu().numpy()

        def add_data(entities: Dict[str, Any], to: str):
            """Create a new group and add multiple datasets to it."""
            to = target.create_group(to)
            for name, data in entities.items():
                # Convert any torch tensor into numpy arrays.
                data = t2n(data) if isinstance(data, Tensor) else data
                to.create_dataset(name, data=data)

        # Attributes
        target.attrs.update(
            {
                "atoms": t2n(self.atom_pair),
                "version": self.version,
                "has_r_poly": self.r_poly is not None,
                "is_atomic": self.atomic,
                "has_r_spline": self.r_spline is not None,
            }
        )

        # Convert electronic integral matrices into structured numpy arrays.
        dtype = np.dtype(
            [
                ("%s-%s" % k, np.float64, tuple(v.shape))
                for k, v in self.hamiltonian.items()
            ]
        )
        h_data, s_data = [
            np.array(tuple(t2n(i) for i in j.values()), dtype)
            for j in [self.hamiltonian, self.overlap]
        ]

        # SkDict component
        add_data(
            {
                "H": h_data,
                "count": len(self.grid),
                "S": s_data,
                "step": self.grid.diff()[0],
            },
            "integrals",
        )

        if (p := self.r_poly) is not None:  # Repulsive polynomial
            add_data({"coef": p.coef, "cutoff": p.cutoff}, "r_poly")

        if (s := self.r_spline) is not None:  # Repulsive spline
            add_data(
                {
                    "grid": s.grid,
                    "cutoff": s.cutoff,
                    "exp_coef": s.exp_coef,
                    "step": s.grid.diff()[0],
                    "tail_coef": s.tail_coef,
                    "spline_coef": s.spline_coef,
                },
                "r_spline",
            )

        if self.atomic:  # Atomic
            add_data(
                {
                    "on_sites": self.on_sites,
                    "hubbard_us": self.hubbard_us,
                    "occupations": self.occupations,
                    "mass": self.mass,
                },
                "atomic",
            )

        # Metadata
        add_data({"time_created": time()}, "metadata")

    def __str__(self) -> str:
        """Returns a string representing the `Skf` object."""
        cls_name = self.__class__.__name__
        name = "-".join([chemical_symbols[int(i)] for i in self.atom_pair])
        r_spline = "No" if self.r_spline is None else "Yes"
        r_poly = "No" if self.r_poly is None else "Yes"
        atomic = "No" if self.atomic is None else "Yes"
        return (
            f"{cls_name}({name}, r-spline: {r_spline}, r-poly: {r_poly}, "
            f"atomic-data: {atomic})"
        )

    def __repr__(self) -> str:
        """Returns a simple string representation of the `Skf` object."""
        cls_name = self.__class__.__name__
        name = "-".join([chemical_symbols[int(i)] for i in self.atom_pair])
        return f"{cls_name}({name})"


class VcrSkf(Skf):
    """Variable compression radii Slater-Koster file parser.

    Arguments:
         atom_pair: Atomic numbers of the elements associated with the
            interaction.
         hamiltonian: Dictionary keyed by azimuthal-number-pairs (ℓ₁, ℓ₂) and
            valued by m×d Hamiltonian integral tensors; where m and d iterate
            over bond-order (σ, π, etc.) and distances respectively.
         overlap: Dictionary storing the overlap integrals in a manner akin to
            ``hamiltonian``.
         grid: Distances at which the ``hamiltonian`` & ``overlap`` elements
            were evaluated.
         r_spline: A :class:`.RSpline` object detailing the repulsive
            spline. [DEFAULT=None]
         r_poly: A :class:`.RPoly` object detailing the repulsive
            polynomial. [DEFAULT=None]
         on_sites: On site terms, homo-atomic systems only. [DEFAULT=None]
         hubbard_us: Hubbard U terms, homo-atomic systems only. [DEFAULT=None]
         mass: Atomic mass, homo-atomic systems only. [DEFAULT=None]
         occupations: Occupations of the orbitals, homo-atomic systems only.
            [DEFAULT=None]

        Attributes:
        atom_pair: True if the system contains atomic data, only relevant to
            the homo-atomic cases.
        hamiltonian:
        overlap:
        hs_cut:

    """

    RSpline = Skf.RSpline
    RPoly = Skf.RPoly

    def __init__(
        self,
        atom_pair: Tensor,
        hamiltonian: SkDict,
        overlap: SkDict,
        grid: Tensor,
        hs_cutoff,
        r_spline: Optional[RSpline] = None,
        r_poly: Optional[RPoly] = None,
        hubbard_us: OptTens = None,
        on_sites: OptTens = None,
        occupations: OptTens = None,
        mass: OptTens = None,
    ):
        super().__init__(
            atom_pair,
            hamiltonian,
            overlap,
            grid,
            hs_cutoff,
            r_spline,
            r_poly,
            hubbard_us,
            on_sites,
            occupations,
            mass,
        )

    @classmethod
    def read(
        cls,
        path: Union[List[str], str],
        atom_pair: Sequence[int] = None,
        path_homo: Union[List[str], str] = None,
        smooth_to_zero: bool = True,
        **kwargs,
    ) -> List["Skf"]:
        """Parse Slater-Koster data from skf files and their binary analogs.

        `path` includes Slater-Koster data with various compression radii,
        therefore there is no `on_sites` or `hubbard_us`. In this case,
        `path_homo`, which refers to standard Slater-Koster data, is added
        to generate `on_sites` or `hubbard_us`. The example of Slater-Koster
        files name style, C-H_02.00_02.00.skf, C-C_10.00_02.00.skf, where the
        first, second numbers correspond to the compression radii of first,
        second elements.

        Arguments:
            path: Path to the binary file or list of SKF files, the tail of
                SKF files should satisfy `.skf.nn.nn.nn.nn` type, where `n`
                is integer number.
            atom_pair: Atomic numbers of the element pair. This is only used
                when reading from an HDF5 file with more than one SK entry.
                [DEFAULT=None]
            smooth_to_zero: Boolean parameter to decide if smooth the tail
                for various SKF with different compression radii.

        Keyword Arguments:
            device: Device on which to place tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]
            path_homo: Get the atomic data

        Returns:
            skf: A list of `Skf` object contain all data parsed from the
                specified file.
        """
        # Read a list of standard Slater-Koster files
        if isinstance(path, list):
            return cls._read_skf(
                path, path_homo, atom_pair, smooth_to_zero, **kwargs
            )
        else:  # -> binary h5 SK files
            return super().read(path, atom_pair, **kwargs)

    def to_hdf5(self, target: Group):
        """Saves the `Skf` instance into a target HDF5 Group."""
        super().to_hdf5(target)

    def write(self, path: str, overwrite: Optional[bool] = False):
        """Writes the `Skf` instance into a target HDF5 Group."""
        super().write(path, overwrite)

    @classmethod
    def _read_skf(
        cls,
        path: Union[List[str], str],
        path_homo: Union[List[str], str] = None,
        atom_pair: Sequence[int] = None,
        smooth_to_zero: bool = True,
        **kwargs,
    ):
        """Read a list Slater-Koster files with various variables."""
        # loop over all compression radii
        # element_name_list_atom = _split_name_r(path_homo, is_tvcr=False)
        _iskf, _compr, kwargs_in = [], [], {}

        for ipath in path:
            _iskf.append(super().read(ipath, atom_pair, **kwargs))

        # merge data with various compression radii
        atom_pair = _iskf[-1].atom_pair
        nr = int(np.sqrt(len(path)))  # -> Number of compression radii
        h_data = {
            ikey: pack([isk.hamiltonian[ikey] for isk in _iskf])
            for ikey in _iskf[-1].hamiltonian.keys()
        }
        s_data = {
            ikey: pack([isk.overlap[ikey] for isk in _iskf])
            for ikey in _iskf[-1].overlap.keys()
        }
        grid = pack([isk.grid for isk in _iskf])

        _cut = [isk.hs_cutoff for isk in _iskf]
        assert max(_cut) == min(_cut), (
            "Please set all the cutoff as the same, get max value"
            + f"{max(_cut)} and min value {min(_cut)}"
        )
        hs_cut = _cut[0]

        # smooth the tail of HS in case the size is different
        if smooth_to_zero:
            n_grid = torch.tensor([len(isk.grid) for isk in _iskf])
            h_data = {
                ikey: vcr_poly_to_zero(grid, h_data[ikey], n_grid)
                for ikey in h_data.keys()
            }
            s_data = {
                ikey: vcr_poly_to_zero(grid, s_data[ikey], n_grid)
                for ikey in s_data.keys()
            }

        # reshape and transpose the dim as: vcr1, vcr2, distances, n_orb_pairs
        h_data = {
            ikey: h_data[ikey]
            .reshape(nr, nr, h_data[ikey].shape[1], -1)
            .transpose(-1, -2)
            for ikey in h_data.keys()
        }
        s_data = {
            ikey: s_data[ikey]
            .reshape(nr, nr, s_data[ikey].shape[1], -1)
            .transpose(-1, -2)
            for ikey in s_data.keys()
        }

        # update grid after smooth, the grid equals to max grid
        step = _iskf[-1].grid[..., -1] - _iskf[-1].grid[..., -2]
        grid = (
            torch.arange(1, h_data[list(h_data.keys())[0]].shape[-2] + 2)
            * step
        )

        if _iskf[-1].r_poly is not None:
            kwargs_in["r_poly"] = pack([isk.r_poly for isk in _iskf]).reshape(
                nr, nr, -1
            )

        # atomic data from path_homo
        if atom_pair[0] == atom_pair[1]:
            # ind = element_name_list_atom.index(atom_pair)
            _homo = super().read(path_homo, atom_pair, **kwargs)
            for ih in ["mass", "occupations", "on_sites", "hubbard_us"]:
                kwargs_in[ih] = getattr(_homo, ih)
            r_grid = _homo.r_spline.grid
            r_cutoff = _homo.r_spline.cutoff
            r_spline_coef = _homo.r_spline.spline_coef
            r_exp_coef = _homo.r_spline.exp_coef
            r_tail_coef = _homo.r_spline.tail_coef
            kwargs_in["r_spline"] = cls.RSpline(
                # Repulsive grid, cutoff & repulsive spline coefficients.
                r_grid,
                r_cutoff,
                r_spline_coef,
                # The exponential and tail spline's coefficients.
                r_exp_coef,
                r_tail_coef,
            )

        # if _iskf[-1].r_spline is not None:
        #     r_grid = pack([ii.r_spline.grid for ii in _iskf])
        #     r_cutoff = pack([ii.r_spline.cutoff.unsqueeze(0) for ii in _iskf])
        #     r_spline_coef = pack([ii.r_spline.spline_coef for ii in _iskf])
        #     r_exp_coef = pack([ii.r_spline.exp_coef for ii in _iskf])
        #     r_tail_coef = pack([ii.r_spline.tail_coef for ii in _iskf])
        #     kwargs_in['r_spline'] = cls.RSpline(
        #         # Repulsive grid, cutoff & repulsive spline coefficients.
        #         r_grid, r_cutoff, r_spline_coef,
        #         # The exponential and tail spline's coefficients.
        #         r_exp_coef, r_tail_coef)

        return cls(atom_pair, h_data, s_data, grid, hs_cut, **kwargs_in)


class TvcrSkf(Skf):
    """Two variable compression radii Slater-Koster file parser.

    Arguments:
         atom_pair: Atomic numbers of the elements associated with the
            interaction.
         hamiltonian: Dictionary keyed by azimuthal-number-pairs (ℓ₁, ℓ₂) and
            valued by m×d Hamiltonian integral tensors; where m and d iterate
            over bond-order (σ, π, etc.) and distances respectively.
         overlap: Dictionary storing the overlap integrals in a manner akin to
            ``hamiltonian``.
         grid: Distances at which the ``hamiltonverlap`` elements
            were evaluated.
         r_spline: A :class:`.RSpline` object detailing the repulsive
            spline. [DEFAULT=None]
         r_poly: A :class:`.RPoly` object detailing the repulsive
            polynomial. [DEFAULT=None]
         on_sites: On site terms, homo-atomic systems only. [DEFAULT=None]
         hubbard_us: Hubbard U terms, homo-atomic systems only. [DEFAULT=None]
         mass: Atomic mass, homo-atomic systems only. [DEFAULT=None]
         occupations: Occupations of the orbitals, homo-atomic systems only.
            [DEFAULT=None]

        Attributes:
        atomic: True if the system contains atomic data, only relevant to the
            homo-atomic cases.

    """

    RSpline = Skf.RSpline
    RPoly = Skf.RPoly

    def __init__(
        self,
        atom_pair: Tensor,
        hamiltonian: SkDict,
        overlap: SkDict,
        grid: Tensor,
        hs_cut,
        r_spline: Optional[RSpline] = None,
        r_poly: Optional[RPoly] = None,
        hubbard_us: OptTens = None,
        on_sites: OptTens = None,
        occupations: OptTens = None,
        mass: OptTens = None,
    ):
        super().__init__(
            atom_pair,
            hamiltonian,
            overlap,
            grid,
            hs_cut,
            r_spline,
            r_poly,
            hubbard_us,
            on_sites,
            occupations,
            mass,
        )

    @classmethod
    def read(
        cls,
        path: Union[List[str], str],
        atom_pair: Sequence[int] = None,
        path_homo: Union[List[str], str] = None,
        smooth_to_zero: bool = True,
        **kwargs,
    ) -> List["Skf"]:
        """Parse Slater-Koster data from skf files and their binary analogs.

        `path` includes Slater-Koster data with various compression radii,
        therefore there is no `on_sites` or `hubbard_us`. In this case,
        `path_homo`, which refers to standard Slater-Koster data, is added
        to generate `on_sites` or `hubbard_us`. The example of Slater-Koster
        files name style, C-H_02.00_02.00.skf, C-C_10.00_02.00.skf, where the
        first, second numbers correspond to the compression radii of first,
        second elements.

        Arguments:
            path: Path to the binary file or list of SKF files, the tail of
                SKF files should satisfy `.skf.nn.nn.nn.nn` type, where `n`
                is integer number.
            atom_pair: Atomic numbers of the element pair. This is only used
                when reading from an HDF5 file with more than one SK entry.
                [DEFAULT=None]
            smooth_to_zero: Boolean parameter to decide if smooth the tail
                for various SKF with different compression radii.

        Keyword Arguments:
            device: Device on which to place tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]
            path_homo: Get the atomic data

        Returns:
            skf: A list of `Skf` object contain all data parsed from the
                specified file.
        """
        if isinstance(path, list):
            return cls._read_skf(
                path, path_homo, atom_pair, smooth_to_zero, **kwargs
            )
        else:  # -> binary h5 SK files
            return super().read(path, atom_pair, **kwargs)

    def to_hdf5(self, target: Group):
        """Saves the `Skf` instance into a target HDF5 Group."""
        super().to_hdf5(target)

    def write(self, path: list, overwrite: Optional[bool] = False):
        """Writes the `Skf` instance into a target HDF5 Group."""
        super().write(path, overwrite)

    @classmethod
    def _read_skf(
        cls,
        path: Union[List[str], str],
        path_homo: Union[List[str], str] = None,
        atom_pair: Sequence[int] = None,
        smooth_to_zero: bool = True,
        **kwargs,
    ):
        assert path_homo is not None, (
            "`path_homo` is needed to read" + "on-site data, but get None."
        )
        _iskf, _compr, kwargs_in = [], [], {}

        for ipath in path:
            _iskf.append(super().read(ipath, atom_pair, **kwargs))

        # merge data with various compression radii
        atom_pair = _iskf[-1].atom_pair
        nr = int(len(path) ** 0.25)  # -> Number of compression radii
        h_data = {
            ikey: pack([isk.hamiltonian[ikey] for isk in _iskf])
            for ikey in _iskf[-1].hamiltonian.keys()
        }
        s_data = {
            ikey: pack([isk.overlap[ikey] for isk in _iskf])
            for ikey in _iskf[-1].overlap.keys()
        }
        grid = pack([isk.grid for isk in _iskf])

        _cut = [isk.hs_cutoff for isk in _iskf]
        assert max(_cut) == min(_cut), (
            "Please set all the cutoff as the same, get max value"
            + f"{max(_cut)} and min value {min(_cut)}"
        )
        hs_cut = _cut[0]

        # smooth the tail of HS in case the size is different
        if smooth_to_zero:
            n_grid = torch.tensor([len(isk.grid) for isk in _iskf])
            h_data = {
                ikey: vcr_poly_to_zero(grid, h_data[ikey], n_grid)
                for ikey in h_data.keys()
            }
            s_data = {
                ikey: vcr_poly_to_zero(grid, s_data[ikey], n_grid)
                for ikey in s_data.keys()
            }
        h_data = {
            ikey: h_data[ikey].reshape(
                nr, nr, nr, nr, h_data[ikey].shape[1], -1
            )
            for ikey in h_data.keys()
        }
        s_data = {
            ikey: s_data[ikey].reshape(
                nr, nr, nr, nr, s_data[ikey].shape[1], -1
            )
            for ikey in s_data.keys()
        }

        # update grid after smooth, the grid equals to max grid
        step = _iskf[-1].grid[..., -1] - _iskf[-1].grid[..., -2]
        grid = (
            torch.arange(1, h_data[list(h_data.keys())[0]].shape[-1] + 1)
            * step
        )

        if _iskf[-1].r_poly is not None:
            kwargs_in["r_poly"] = pack([isk.r_poly for isk in _iskf]).reshape(
                nr, nr, nr, nr, -1
            )

        # atomic data from path_homo
        if atom_pair[0] == atom_pair[1]:
            # ind = element_name_list_atom.index(ie)
            _homo = super().read(path_homo, atom_pair, **kwargs)
            for ih in ["mass", "occupations", "on_sites", "hubbard_us"]:
                kwargs_in[ih] = getattr(_homo, ih)

        if _iskf[-1].r_spline is not None:
            r_grid = pack([ii.r_spline.grid for ii in _iskf])
            r_cutoff = pack([ii.r_spline.cutoff.unsqueeze(0) for ii in _iskf])
            r_spline_coef = pack([ii.r_spline.spline_coef for ii in _iskf])
            r_exp_coef = pack([ii.r_spline.exp_coef for ii in _iskf])
            r_tail_coef = pack([ii.r_spline.tail_coef for ii in _iskf])

            kwargs_in["r_spline"] = cls.RSpline(
                # Repulsive grid, cutoff & repulsive spline coefficients.
                r_grid,
                r_cutoff,
                r_spline_coef,
                # The exponential and tail spline's coefficients.
                r_exp_coef,
                r_tail_coef,
            )

        this_cls = cls(atom_pair, h_data, s_data, grid, hs_cut, **kwargs_in)

        # if write:
        #     this_cls.write(output, overwrite)

        return this_cls


#########################
# Convenience Functions #
#########################
def _s2t(text: Union[str, List[str]], sep: str = " \t", **kwargs) -> Tensor:
    """Converts string to tensor.

    This uses the `np.fromstring` method to quickly convert blocks of text
    into arrays, which are then converted into tensors.

    Arguments:
        text: string to extract the tensor from. If a list of strings is
            supplied then they will be joined prior to tensor extraction.
        sep: possible delimiters. [DEFAULT=' \t']

    Keyword Arguments:
        kwargs: these will be passed into the `torch.tensor` call.

    """
    text = sep.join(text) if isinstance(text, list) else text
    return torch.tensor(
        np.fromstring(text, sep=sep, dtype=np.float64), **kwargs
    )


def _esr(text: str) -> str:
    """Expand stared number representations.

    This is primarily used to resolve the skf file specification violations
    which are found in some of the early skf files. Specifically the user of
    started notations like `10*1.0` to represent a value of one repeated ten
    times, or the mixed use of spaces, tabs and commas.

    Arguments:
        text: string to be rectified.

    Returns:
        r_text: rectified string.

    Notes:
        This finds strings like `3*.0` & `10*1` and replaces them with
        `.0 .0 .0` & `1 1 1 1 1 1 1 1 1 1` respectively.
    """
    # Strip out unnecessary commas
    text = text.replace(",", " ")
    if "*" in text:
        for i in set(re.findall(r"[0-9]+\*[0-9|.]+", text)):
            n, val = i.strip(",").split("*")
            text = text.replace(i, f"{' '.join([val] * int(n))}")
    return text


def _split_name_r(
    path: list, is_vcr: bool = False, is_tvcr: bool = False
) -> Tuple[str, str, Tensor]:
    """Split skf name with compression radii tail."""
    e = "[A-Z][a-z]?"
    element_name = [
        "".join(
            re.findall(e, re.search(rf"{e}.?{e}(?=.)", split(ip)[-1]).group(0))
        )
        for ip in path
    ]

    if is_vcr:
        vcr = [
            torch.tensor([float(ip[-15:-10]), float(ip[-9:-4])]) for ip in path
        ]
        return vcr, element_name
    elif is_tvcr:
        vcr = [
            torch.tensor(
                [
                    float(ip[-27:-22]),
                    float(ip[-21:-16]),
                    float(ip[-15:-10]),
                    float(ip[-9:-4]),
                ]
            )
            for ip in path
        ]
        return vcr, element_name

    else:
        return element_name


if __name__ == "__main__":
    skf_path = "tests/Si-Si.skf"
    sk = Skf.from_skf(skf_path)
    print(sk)
    dd = sk.to_dict()
    print(sk.to_dict())
    sk = Skf.from_dict(dd)
    print(sk)
