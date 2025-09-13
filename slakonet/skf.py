"""Methods for reading from and writing to skf and associated files."""

# Adopted from tbmalt
import re
import warnings
from dataclasses import dataclass
from os.path import isfile, split, splitext, basename
from time import time
from typing import List, Tuple, Union, Dict, Sequence, Any, Optional
from itertools import product
import numpy as np
import torch
from torch import Tensor
from slakonet.elements import chemical_symbols
from slakonet.elements import atomic_numbers
from slakonet.utils import pack, triangular_root, tetrahedral_root

# from slakonet.interpolation import poly_to_zero, vcr_poly_to_zero

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


if __name__ == "__main__":
    skf_path = "tests/Si-Si.skf"
    sk = Skf.from_skf(skf_path)
    print(sk)
    dd = sk.to_dict()
    print(sk.to_dict())
    sk = Skf.from_dict(dd)
    print(sk)
