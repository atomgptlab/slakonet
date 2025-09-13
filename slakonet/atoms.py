"""A container to hold data associated with a chemical system's structure.

This module provides the `Geometry` data structure class and its associated
code. The `Geometry` class is intended to hold any & all data needed to fully
describe a chemical system's structure.
"""

from typing import Union, List, Optional
from operator import itemgetter
import torch
import numpy as np
from ase import Atoms
import ase.io as io
from slakonet.utils import pack, merge
from slakonet.units import length_units
from slakonet.elements import chemical_symbols
from slakonet.elements import atomic_numbers as data_numbers

Tensor = torch.Tensor


class Periodic:
    """Calculate the translation vectors for cells for 3D periodic boundary condition.

    Arguments:
        latvec: Lattice vector describing the geometry of periodic geometry,
            with Bohr as unit.
        cutoff: Interaction cutoff distance for reading SK table.

    Keyword Args:
        distance_extention: Extention of cutoff in SK tables to smooth the tail.
        positive_extention: Extension for the positive lattice vectors.
        negative_extention: Extension for the negative lattice vectors.

    Return:
        cutoff: Global cutoff for the diatomic interactions, unit is Bohr.
        cellvol: Volume of the unit cell.
        reccellvol: Volume of the reciprocal lattice unit cell.
        cellvec: Cell translation vectors in relative coordinates.
        rcellvec: Cell translation vectors in absolute units.
        ncell: Number of lattice cells.

    Examples:
        >>> from tbmalt import Periodic
        >>> import torch
    """

    def __init__(
        self,
        geometry: object,
        latvec: Tensor,
        cutoff: Union[Tensor, float],
        **kwargs,
    ):
        self.geometry = geometry
        self.is_periodic = geometry.is_periodic
        self._n_batch = (
            geometry._n_batch if geometry._n_batch is not None else 1
        )
        self.n_atoms = geometry.n_atoms

        # mask for periodic and non-periodic systems
        self.mask_pe = self.geometry.periodic_list
        self.latvec, self.cutoff = self._check(latvec, cutoff, **kwargs)

        if self.geometry.frac_list.any():
            self._positions_check()
        dim = self.geometry.atomic_numbers.dim()
        self.atomic_numbers = (
            self.geometry.atomic_numbers.unsqueeze(0)
            if dim == 1
            else self.geometry.atomic_numbers
        )
        self.positions = (
            self.geometry.positions.unsqueeze(0)
            if dim == 1
            else self.geometry.positions
        )

        dist_ext = kwargs.get("distance_extention", 1.0)
        return_distance = kwargs.get("return_distance", True)

        # Global cutoff for the diatomic interactions
        self.cutoff = self.cutoff + dist_ext

        self.invlatvec, self.mask_zero = self._inverse_lattice()

        self.recvec = self._reciprocal_lattice()

        # Unit cell volume
        self.cellvol = abs(torch.det(self.latvec))

        self.cellvec, self.rcellvec, self.ncell = self.get_cell_translations(
            **kwargs
        )

        # K-sampling
        self.kpoints, self.n_kpoints, self.k_weights = self._kpoints(**kwargs)

        if return_distance is True:
            (
                self.mask_central_cell,
                self.positions_pe,
                self.positions_vec,
                self.periodic_distances,
            ) = self._periodic_distance()
            self.neighbour_pos, self.neighbour_vec, self.neighbour_dis = (
                self._neighbourlist()
            )

    def _check(self, latvec, cutoff, **kwargs):
        """Check dimension, type of lattice vector and cutoff."""
        # Default lattice vector is from geometry, therefore default unit is bohr
        unit = kwargs.get("unit", "bohr")

        # Molecule will be padding with zeros, here select latvec for solid
        if isinstance(latvec, list):
            latvec = pack(latvec)
        elif not isinstance(latvec, Tensor):
            raise TypeError("Lattice vector is tensor or list of tensor.")

        if latvec.dim() == 2:
            latvec = latvec.unsqueeze(0)
        elif latvec.dim() != 3:
            raise ValueError("lattice vector dimension should be 2 or 3")

        if isinstance(cutoff, float):
            cutoff = torch.tensor([cutoff])
        elif not isinstance(cutoff, Tensor):
            raise TypeError(
                f"cutoff should be float or Tensor, but get {type(cutoff)}"
            )
        if cutoff.dim() == 0:
            cutoff = cutoff.unsqueeze(0)
        elif cutoff.dim() >= 2:
            raise ValueError("cutoff should be 0, 1 dimension tensor or float")

        if latvec.size(0) != 1 and cutoff.size(0) == 1:
            cutoff = cutoff.repeat_interleave(latvec.size(0))

        return latvec, cutoff

    def _positions_check(self):
        """Check positions type (fraction or not) and unit."""
        is_frac = self.geometry.frac_list
        dim = self.positions.dim()

        # transfer periodic positions to bohr
        position_pe = self.positions[self.mask_pe]
        _mask = is_frac[self.mask_pe]

        # whether fraction coordinates in the range [0, 1)
        if torch.any(position_pe[_mask] >= 1) or torch.any(
            position_pe[_mask] < 0
        ):
            position_pe[_mask] = torch.abs(position_pe[_mask]) - torch.floor(
                torch.abs(position_pe[_mask])
            )

        position_pe[_mask] = torch.matmul(
            position_pe[_mask], self.latvec[is_frac]
        )
        self.positions[self.mask_pe] = position_pe

    def get_cell_translations(self, **kwargs):
        """Get cell translation vectors."""
        pos_ext = kwargs.get("positive_extention", 1)
        neg_ext = kwargs.get("negative_extention", 1)

        _tmp = torch.floor(
            self.cutoff * torch.norm(self.invlatvec, dim=-1).T
        ).T
        ranges = torch.stack([-(neg_ext + _tmp), pos_ext + _tmp])

        # 1D/ 2D cell translation
        ranges[torch.stack([self.mask_zero, self.mask_zero])] = 0

        # Length of the first, second and third column in ranges
        leng = ranges[1, :].long() - ranges[0, :].long() + 1

        # Number of cells
        ncell = leng[..., 0] * leng[..., 1] * leng[..., 2]

        # Cell translation vectors in relative coordinates
        # Large values are padded at the end of short cell vectors to exceed cutoff distance
        cellvec = pack(
            [
                torch.stack(
                    [
                        torch.linspace(
                            iran[0, 0], iran[1, 0], ile[0]
                        ).repeat_interleave(ile[2] * ile[1]),
                        torch.linspace(iran[0, 1], iran[1, 1], ile[1])
                        .repeat(ile[0])
                        .repeat_interleave(ile[2]),
                        torch.linspace(iran[0, 2], iran[1, 2], ile[2]).repeat(
                            ile[0] * ile[1]
                        ),
                    ]
                )
                for ile, iran in zip(leng, ranges.transpose(1, 0))
            ],
            value=1e3,
        )
        rcellvec = pack(
            [
                torch.matmul(ilv.transpose(0, 1), icv.T.unsqueeze(-1)).squeeze(
                    -1
                )
                for ilv, icv in zip(self.latvec, cellvec)
            ],
            value=1e3,
        )

        return cellvec, rcellvec, ncell

    def _periodic_distance(self):
        """Get distances between central cell and neighbour cells."""
        mask_central_cell = (self.rcellvec != 0).sum(-1) == 0
        positions = self.rcellvec.unsqueeze(2) + self.positions.unsqueeze(1)
        size_system = self.atomic_numbers.ne(0).sum(-1)
        positions_vec = -positions.unsqueeze(-3) + self.positions.unsqueeze(
            1
        ).unsqueeze(-2)
        distance = pack(
            [
                torch.sqrt(
                    (
                        (
                            ipos[:, :inat].repeat(1, inat, 1)
                            - torch.repeat_interleave(icp[:inat], inat, 0)
                        )
                        ** 2
                    ).sum(-1)
                ).reshape(-1, inat, inat)
                for ipos, icp, inat in zip(
                    positions, self.positions, size_system
                )
            ],
            value=1e3,
        )

        return mask_central_cell, positions, positions_vec, distance

    @property
    def n_atoms_pe(self):
        """Periodic number of atoms, include all images."""
        mask = self.neighbour.any(-1).any(-1)
        return pack(
            torch.split(
                torch.repeat_interleave(self.geometry.n_atoms, mask.sum(-1)),
                tuple(mask.sum(-1)),
            )
        )

    def supercell(self, idx: Tensor):

        # convert single to batch
        idx = idx if idx.dim() == 2 else idx.unsqueeze(0)

        ranges = pack([torch.zeros(*idx.shape), idx - 1])

        cellvec = pack(
            [
                torch.stack(
                    [
                        torch.linspace(
                            iran[0, 0], iran[1, 0], ile[0]
                        ).repeat_interleave(ile[2] * ile[1]),
                        torch.linspace(iran[0, 1], iran[1, 1], ile[1])
                        .repeat(ile[0])
                        .repeat_interleave(ile[2]),
                        torch.linspace(iran[0, 2], iran[1, 2], ile[2]).repeat(
                            ile[0] * ile[1]
                        ),
                    ]
                )
                for ile, iran in zip(idx, ranges.transpose(1, 0))
            ],
            value=1e3,
        )
        rcellvec = pack(
            [
                torch.matmul(ilv.transpose(0, 1), icv.T.unsqueeze(-1)).squeeze(
                    -1
                )
                for ilv, icv in zip(self.latvec, cellvec)
            ],
            value=1e3,
        )

        positions = rcellvec.unsqueeze(2) + self.positions.unsqueeze(1)
        atomic_numbers = self.atomic_numbers.repeat(1, positions.shape[-3])
        positions = positions.flatten(-3, -2)
        cell = self.geometry.cell * idx

        return Geometry(atomic_numbers, positions, cell=cell)

    @property
    def atomic_numbers_pe(self):
        """Periodic number of atoms, include all images."""
        return pack(
            [
                number.repeat(nc, 1)
                for number, nc in zip(self.atomic_numbers, self.ncell)
            ]
        )

    def _neighbourlist(self):
        """Get distance matrix of neighbour list according to periodic boundary condition."""
        _mask = self.neighbour.any(-1).any(-1)
        neighbour_pos = pack(
            [
                self.positions_pe[ibatch][_mask[ibatch]]
                for ibatch in range(self.cutoff.size(0))
            ],
            value=1e3,
        )
        neighbour_vec = pack(
            [
                self.positions_vec[ibatch][_mask[ibatch]]
                for ibatch in range(self.cutoff.size(0))
            ],
            value=1e3,
        )
        neighbour_dis = pack(
            [
                self.periodic_distances[ibatch][_mask[ibatch]]
                for ibatch in range(self.cutoff.size(0))
            ],
            value=1e3,
        )

        return neighbour_pos, neighbour_vec, neighbour_dis

    def _inverse_lattice(self):
        """Get inverse lattice vectors."""
        # build a mask for zero vectors in 1D/ 2D lattice vectors
        mask_zero = self.latvec.eq(0).all(-1)
        _latvec = self.latvec + torch.diag_embed(
            mask_zero.type(self.latvec.dtype)
        )

        # inverse lattice vectors
        _invlat = torch.transpose(
            torch.linalg.solve(
                _latvec,
                torch.eye(_latvec.shape[-1]).repeat(_latvec.shape[0], 1, 1),
            ),
            -1,
            -2,
        )
        _invlat[mask_zero] = 0

        return _invlat, mask_zero

    def _reciprocal_lattice(self):
        """Get reciprocal lattice vectors"""
        return 2 * np.pi * self.invlatvec

    def _kpoints(self, **kwargs):
        """Calculate K-points."""
        _kpoints = kwargs.get("kpoints", None)
        _klines = kwargs.get("klines", None)

        if _kpoints is not None:
            assert _klines is None, "One of kpoints and klines should be None"
            assert isinstance(_kpoints, Tensor), (
                "kpoints should be" + f"torch.Tensor, but get {type(_kpoints)}"
            )
            _kpoints = (
                _kpoints if _kpoints.dim() == 2 else _kpoints.unsqueeze(0)
            )
            # all atomic_numbers transfer to batch
            assert len(_kpoints) == len(
                self.atomic_numbers
            ), f"length of kpoints do not equal to {len(self.atomic_numbers)}"
            assert _kpoints.shape[1] == 3, "column of _kpoints si not 3"

            return self._super_sampling(_kpoints)
        elif _klines is not None:
            assert isinstance(_klines, Tensor), (
                "klines should be" + f"torch.Tensor, but get {type(_klines)}"
            )
            _klines = _klines if _klines.dim() == 3 else _klines.unsqueeze(0)

            return self._klines(_klines)

        else:
            _kpoints = torch.ones(self._n_batch, 3, dtype=torch.int32)

            return self._super_sampling(_kpoints)

    def _super_sampling(self, _kpoints):
        """Super sampling."""
        _n_kpoints = _kpoints[..., 0] * _kpoints[..., 1] * _kpoints[..., 2]
        _kpoints_inv = 0.5 / _kpoints
        _kpoints_inv2 = 1.0 / _kpoints
        _nkxyz = _kpoints[..., 0] * _kpoints[..., 1] * _kpoints[..., 2]
        n_ind = tuple(_nkxyz)
        _nkx, _nkyz = _kpoints[..., 0], _kpoints[..., 1] * _kpoints[..., 2]
        _nky, _nkxz = _kpoints[..., 1], _kpoints[..., 0] * _kpoints[..., 2]
        _nkz, _nkxy = _kpoints[..., 2], _kpoints[..., 0] * _kpoints[..., 1]

        # create baseline of kpoints, if n_kpoint in x direction is N,
        # the value will be [0.5 / N] * n_kpoint_x * n_kpoint_y * n_kpoint_z
        _x_base = torch.repeat_interleave(_kpoints_inv[..., 0], _nkxyz)
        _y_base = torch.repeat_interleave(_kpoints_inv[..., 1], _nkxyz)
        _z_base = torch.repeat_interleave(_kpoints_inv[..., 2], _nkxyz)

        # create K-mesh increase in each direction range from 0~1
        _x_incr = torch.cat(
            [
                torch.repeat_interleave(torch.arange(ii) * iv, yz)
                for ii, yz, iv in zip(_nkx, _nkyz, _kpoints_inv2[..., 0])
            ]
        )
        _y_incr = torch.cat(
            [
                torch.repeat_interleave(torch.arange(iy) * iv, iz).repeat(ix)
                for ix, iy, iz, xz, iv in zip(
                    _nkx, _nky, _nkz, _nkxz, _kpoints_inv2[..., 1]
                )
            ]
        )
        _z_incr = torch.cat(
            [
                (torch.arange(iz) * iv).repeat(xy)
                for iz, xy, iv in zip(_nkz, _nkxy, _kpoints_inv2[..., 2])
            ]
        )

        all_kpoints = torch.stack(
            [
                pack(torch.split((_x_base + _x_incr).unsqueeze(1), n_ind)),
                pack(torch.split((_y_base + _y_incr).unsqueeze(1), n_ind)),
                pack(torch.split((_z_base + _z_incr).unsqueeze(1), n_ind)),
            ]
        )

        k_weights = pack(
            torch.split(torch.ones(_n_kpoints.sum()), tuple(_n_kpoints))
        )
        k_weights = k_weights / _n_kpoints.unsqueeze(-1)

        return all_kpoints.squeeze(-1).permute(1, 2, 0), _n_kpoints, k_weights

    def _klines(self, klines: Tensor):
        """K-lines for band structure calculations."""
        if self._n_batch == 1:
            if klines.dim() == 2:
                klines = klines.unsqueeze(0)
            elif klines.dim() > 3 or klines.dim() == 1:
                raise ValueError(
                    f"klines dims should be 2 or 3, get {klines.dim()}"
                )
        else:
            assert self._n_batch == len(
                klines
            ), f"klines size {len(klines)} is not consistent with batch size {self._n_batch}"
            assert (
                klines.dim() == 3
            ), f"klines dims should be 3, get {klines.dim()}"
        assert klines.shape[-1] == 7, (
            "Shape error, for each K-line path, "
            + "last dimension shold include:[kx1, ky1, kz1, kx2, ky2, kz2, N]"
            + f"but get {klines.shape[-1]} numbers in the last dimension"
        )

        # Faltten all K-Lines so that we can deal with K-Lines only once
        _n_kpoints = klines[..., -1].sum(-1).long()
        _klines = klines.flatten(0, 1)
        _mask = _klines[..., -1].ge(1)

        # Extend and get interval K-Points
        delta_k = _klines[..., 3:6] - _klines[..., :3]

        delta_k[_mask] = delta_k[_mask] / (
            _klines[..., -1][_mask].unsqueeze(-1) - 1
        )
        delta_k = torch.repeat_interleave(delta_k, _klines[..., -1].long(), 0)

        repeat_nums = torch.cat(
            [torch.arange(ik) for ik in _klines[..., -1].long()]
        ).unsqueeze(-1)
        klines_ext = (
            torch.repeat_interleave(
                _klines[..., :3], _klines[..., -1].long(), 0
            )
            + delta_k * repeat_nums
        )
        klines_ext = pack(klines_ext.split(tuple(_n_kpoints.tolist())))

        # Create averaged weight for each K-Lines
        k_weights = pack(
            torch.split(
                torch.repeat_interleave(1.0 / _n_kpoints, _n_kpoints),
                tuple(_n_kpoints),
            )
        )

        return klines_ext, _n_kpoints, k_weights

    def get_reciprocal_volume(self):
        """Get reciprocal lattice unit cell volume."""
        return abs(torch.det(2 * np.pi * (self.invlatvec.transpose(0, 1))))

    @property
    def neighbour(self):
        """Get neighbour list according to periodic boundary condition."""
        return torch.stack(
            [
                self.periodic_distances[ibatch].le(self.cutoff[ibatch])
                for ibatch in range(self.cutoff.size(0))
            ]
        )

    @property
    def distances(self) -> Tensor:
        """Distance matrix between atoms in the system."""
        return self.neighbour_dis.permute(0, 2, 3, 1)

    @property
    def distance_vectors(self) -> Tensor:
        """Distance vector matrix between atoms in the system."""
        return self.neighbour_vec

    @property
    def cellvec_neighbour(self):
        """Return cell vector which distances between all atoms in return cell
        and center cell are smaller than cutoff."""
        _mask = self.neighbour.any(-1).any(-1)
        _cellvec = self.cellvec.permute(0, -1, -2)
        _neighbour_vec = pack(
            [
                _cellvec[ibatch][_mask[ibatch]]
                for ibatch in range(self.cutoff.size(0))
            ]
        )

        return _neighbour_vec.permute(0, -1, -2)

    @property
    def phase(self):
        """Select kpoint for each interactions."""
        kpoint = 2.0 * np.pi * self.kpoints

        # shape: [n_batch, n_cell, 3]
        cell_vec = self.cellvec_neighbour

        return pack(
            [
                torch.exp(
                    (0.0 + 1.0j) * torch.einsum("ij, ijk-> ik", ik, cell_vec)
                )
                for ik in kpoint.permute(1, 0, -1)
            ]
        )

    def unique_atomic_numbers(self) -> Tensor:
        """Identifies and returns a tensor of unique atomic numbers.

        This method offers a means to identify the types of elements present
        in the system(s) represented by a `Geometry` object.

        Returns:
            unique_atomic_numbers: A tensor specifying the unique atomic
                numbers present.
        """
        return self.geometry.unique_atomic_numbers()


class Geometry:
    """Data structure for storing geometric information on molecular systems.

    The `Geometry` class stores any information that is needed to describe a
    chemical system; atomic numbers, positions, etc. This class also permits
    batch system representation. However, mixing of PBC & non-PBC systems is
    strictly forbidden.

    Arguments:
        atomic_numbers: Atomic numbers of the atoms.
        positions : Coordinates of the atoms.
        units: Unit in which ``positions`` were specified. For a list of
            available units see :mod:`.units`. [DEFAULT='bohr']

    Attributes:
        atomic_numbers: Atomic numbers of the atoms.
        positions : Coordinates of the atoms.
        n_atoms: Number of atoms in the system.

    Notes:
        When representing multiple systems, the `atomic_numbers` & `positions`
        tensors will be padded with zeros. Tensors generated from ase atoms
        objects or HDF5 database entities will not share memory with their
        associated numpy arrays, nor will they inherit their dtype.

    Warnings:
        At this time, periodic boundary conditions are not supported.

    Examples:
        Geometry instances may be created by directly passing in the atomic
        numbers & atom positions

        >>> from tbmalt import Geometry
        >>> H2 = Geometry(torch.tensor([1, 1]),
        >>>               torch.tensor([[0.00, 0.00, 0.00],
        >>>                             [0.00, 0.00, 0.79]]))
        >>> print(H2)
        Geometry(H2)

        Or from an ase.Atoms object

        >>> from ase.build import molecule
        >>> CH4_atoms = molecule('CH4')
        >>> print(CH4_atoms)
        Atoms(symbols='CH4', pbc=False)
        >>> CH4 = Geometry.from_ase_atoms(CH4_atoms)
        >>> print(CH4)
        Geometry(CH4)

        Multiple systems can be represented by a single ``Geometry`` instance.
        To do this, simply pass in lists or packed tensors where appropriate.

    """

    __slots__ = [
        "atomic_numbers",
        "positions",
        "n_atoms",
        "updated_dist_vec",
        "cell",
        "is_periodic",
        "periodic_list",
        "frac_list",
        "pbc",
        "_n_batch",
        "_mask_dist",
        "__dtype",
        "__device",
    ]

    def __init__(
        self,
        atomic_numbers: Union[Tensor, List[Tensor]],
        positions: Union[Tensor, List[Tensor]],
        cell: Union[Tensor, List[Tensor]] = None,
        frac: Union[float, List[float]] = None,
        units: Optional[str] = "bohr",
        **kwargs,
    ):

        # "pack" will only effect lists of tensors
        self.atomic_numbers: Tensor = pack(atomic_numbers)
        self.positions: Tensor = pack(positions)
        self.updated_dist_vec = kwargs.get("updated_dist_vec", None)

        # bool tensor is_periodic defines if there is solid
        if cell is None:
            self.is_periodic = False  # no system is solid
            self.cell = None
        else:
            cell = pack(cell)
            if cell.eq(0).all():
                self.is_periodic = False  # all cell is zeros
                self.cell = None
            else:
                _cell = Pbc(cell, frac, units)
                self.cell, self.periodic_list, self.frac_list, self.pbc = (
                    _cell.cell,
                    _cell.periodic_list,
                    _cell.frac_list,
                    _cell.pbc,
                )
                self.is_periodic = True if self.periodic_list.any() else False

        # Mask for clearing padding values in the distance matrix.
        if (temp_mask := self.atomic_numbers != 0).all():
            self._mask_dist: Union[Tensor, bool] = False
        else:
            self._mask_dist: Union[Tensor, bool] = ~(
                temp_mask.unsqueeze(-2) * temp_mask.unsqueeze(-1)
            )

        self.n_atoms: Tensor = self.atomic_numbers.count_nonzero(-1)

        # Number of batches if in batch mode (for internal use only)
        self._n_batch: Optional[int] = (
            None if self.atomic_numbers.dim() == 1 else len(atomic_numbers)
        )

        # Ensure the distances are in atomic units (bohr)
        if units != "bohr":
            self.positions: Tensor = self.positions * length_units[units]

        # These are static, private variables and must NEVER be modified!
        self.__device = self.positions.device
        self.__dtype = self.positions.dtype

        # Check for size discrepancies in `positions` & `atomic_numbers`
        if self.atomic_numbers.ndim == 2:
            check = len(atomic_numbers) == len(positions)
            assert check, "`atomic_numbers` & `positions` size mismatch found"

        # Ensure tensors are on the same device (only two present currently)
        if self.positions.device != self.positions.device:
            raise RuntimeError("All tensors must be on the same device!")

    @property
    def device(self) -> torch.device:
        """The device on which the `Geometry` object resides."""
        return self.__device

    @device.setter
    def device(self, *args):
        # Instruct users to use the ".to" method if wanting to change device.
        raise AttributeError(
            "Geometry object's dtype can only be modified "
            'via the ".to" method.'
        )

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by geometry object."""
        return self.__dtype

    @property
    def distances(self) -> Tensor:
        """Distance matrix between atoms in the system."""
        if self.updated_dist_vec is None:
            dist = torch.cdist(self.positions, self.positions, p=2)
            # Ensure padding area is zeroed out
            dist[self._mask_dist] = 0
            torch.diagonal(dist, dim1=-2, dim2=-1).zero_()
            return dist
        else:
            return torch.sqrt((self.updated_dist_vec**2).sum(-1))

    @property
    def distance_vectors(self) -> Tensor:
        """Distance vector matrix between atoms in the system."""
        if self.updated_dist_vec is None:
            dist_vec = self.positions.unsqueeze(-2) - self.positions.unsqueeze(
                -3
            )
            dist_vec[self._mask_dist] = 0
            return dist_vec
        else:
            return self.updated_dist_vec

    @property
    def chemical_symbols(self) -> list:
        """Chemical symbols of the atoms present."""
        return batch_chemical_symbols(self.atomic_numbers)

    def unique_atomic_numbers(self) -> Tensor:
        """Identifies and returns a tensor of unique atomic numbers.

        This method offers a means to identify the types of elements present
        in the system(s) represented by a `Geometry` object.

        Returns:
            unique_atomic_numbers: A tensor specifying the unique atomic
                numbers present.
        """
        return torch.unique(self.atomic_numbers[self.atomic_numbers.ne(0)])

    @classmethod
    def from_ase_atoms(
        cls,
        atoms: Union[Atoms, List[Atoms]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        units: str = "angstrom",
    ) -> "Geometry":
        """Instantiates a Geometry instance from an `ase.Atoms` object.

        Multiple atoms objects can be passed in to generate a batched Geometry
        instance which represents multiple systems.

        Arguments:
            atoms: Atoms object(s) to instantiate a Geometry instance from.
            device: Device on which to create any new tensors. [DEFAULT=None]
            dtype: dtype to be used for floating point tensors. [DEFAULT=None]
            units: Length unit used by `Atoms` object. [DEFAULT='angstrom']

        Returns:
            geometry: The resulting ``Geometry`` object.

        Warnings:
            Periodic boundary conditions are not currently supported.

        Raises:
            NotImplementedError: If the `ase.Atoms` object has periodic
                boundary conditions enabled along any axis.
        """
        # If not specified by the user; ensure that the default dtype is used,
        # rather than inheriting from numpy. Failing to do this will case some
        # *very* hard to diagnose errors.
        dtype = torch.get_default_dtype() if dtype is None else dtype

        if not isinstance(atoms, list):  # If a single system
            return cls(  # Create a Geometry instance and return it
                torch.tensor(atoms.get_atomic_numbers(), device=device),
                torch.tensor(atoms.positions, device=device, dtype=dtype),
                torch.tensor(atoms.cell, device=device, dtype=dtype),
                units=units,
            )

        else:  # If a batch of systems
            return cls(  # Create a batched Geometry instance and return it
                [
                    torch.tensor(
                        a.get_atomic_numbers(), dtype=torch.long, device=device
                    )
                    for a in atoms
                ],
                [
                    torch.tensor(a.positions, device=device, dtype=dtype)
                    for a in atoms
                ],
                [
                    torch.tensor(a.cell, device=device, dtype=dtype)
                    for a in atoms
                ],
                units=units,
            )

    def to_vasp(
        self,
        output: str,
        direct=False,
        selective_dynamics=False,
        zrange=(0, 10000),
    ):
        assert self.cell is not None, "VASP only support PBC system"

        positions = self.positions / length_units["angstrom"]
        cells = self.cell / length_units["angstrom"]
        cell_xyz = torch.sqrt((cells**2).sum(-2)).unsqueeze(-2)
        positions = positions / cell_xyz if direct else positions

        def write_single(number, position, cell):
            f = open(output, "w")
            f.write(self.__repr__() + "\n")
            f.write("1.0 \n")

            for i in range(3):
                for j in range(3):
                    f.write(" %15.10f" % cell[i, j])
                f.write("\n")

            for i in torch.unique(number):
                f.write(chemical_symbols[i] + " ")
            f.write("\n")

            for i in torch.unique(number):
                f.write(" %d" % (sum(number == i)) + " ")
            f.write("\n")

            if selective_dynamics:
                f.write("selective dynamics\n")

            if direct:
                f.write("Direct\n")
            else:
                f.write("Cartesian\n")
            print("selective_dynamics", selective_dynamics)

            if selective_dynamics:
                for i in torch.unique(number):
                    print(i)
                    pos = position[number == i]
                    for j in range(len(pos)):
                        for k in range(3):
                            f.write(" %.12f" % pos[j, k])

                        if pos[j, 2] > zrange[0] and pos[j, 2] < zrange[-1]:
                            select = " False False False"
                        else:
                            select = " True True True"
                        print("select", select, pos[j, 2])
                        f.write(select + "\n")
            else:
                for i in torch.unique(number):
                    pos = position[number == i]
                    for j in range(len(pos)):
                        for k in range(3):
                            f.write(" %.12f" % pos[j, k])
                        f.write("\n")

        # Write output file
        if self._n_batch is None:
            write_single(self.atomic_numbers, positions, cells)
        else:
            for number, position, cell in zip(
                self.atomic_numbers, positions, cells
            ):
                write_single(number, position, cell)

    def remove_atoms(self, low_limit, up_limit):
        mask = self.positions[..., 2] < up_limit
        mask = mask * self.positions[..., 2] > low_limit

        if self._n_batch is None:
            atomic_numbers = self.atomic_numbers[mask]
            positions = self.positions[mask]
        else:
            atomic_numbers = pack(
                [atom[imask] for atom, imask in zip(self.atomic_numbers, mask)]
            )
            positions = pack(
                [pos[imask] for pos, imask in zip(self.positions, mask)]
            )

        return Geometry(atomic_numbers, positions, cell=self.cell)

    def to(self, device: torch.device) -> "Geometry":
        """Returns a copy of the `Geometry` instance on the specified device

        This method creates and returns a new copy of the `Geometry` instance
        on the specified device "``device``".

        Arguments:
            device: Device to which all associated tensors should be moved.

        Returns:
            geometry: A copy of the `Geometry` instance placed on the
                specified device.

        Notes:
            If the `Geometry` instance is already on the desired device then
            `self` will be returned.
        """
        # Developers Notes: It is imperative that this function gets updated
        # whenever new attributes are added to the `Geometry` class. Otherwise
        # this will return an incomplete `Geometry` object.
        if self.atomic_numbers.device == device:
            return self
        else:
            return self.__class__(
                self.atomic_numbers.to(device=device),
                self.positions.to(device=device),
            )

    def __getitem__(self, selector) -> "Geometry":
        """Permits batched Geometry instances to be sliced as needed."""
        # Block this if the instance has only a single system
        if self.atomic_numbers.ndim != 2:
            raise IndexError(
                "Geometry slicing is only applicable to batches of systems."
            )

        if self.cell is None:
            return self.__class__(
                self.atomic_numbers[selector], self.positions[selector]
            )
        else:
            return self.__class__(
                self.atomic_numbers[selector],
                self.positions[selector],
                cell=self.cell[selector],
            )

    def __add__(self, other: "Geometry") -> "Geometry":
        """Combine two `Geometry` objects together."""
        if self.__class__ != other.__class__:
            raise TypeError(
                "Addition can only take place between two Geometry objects."
            )

        # Catch for situations where one or both systems are not batched.
        s_batch = self.atomic_numbers.ndim == 2
        o_batch = other.atomic_numbers.ndim == 2

        an_1 = torch.atleast_2d(self.atomic_numbers)
        an_2 = torch.atleast_2d(other.atomic_numbers)

        pos_1 = self.positions
        pos_2 = other.positions

        pos_1 = pos_1 if s_batch else pos_1.unsqueeze(0)
        pos_2 = pos_2 if o_batch else pos_2.unsqueeze(0)

        if self.cell is not None:
            cell_1 = self.cell if s_batch else self.cell.unsqueeze(0)
            cell_2 = other.cell if o_batch else other.cell.unsqueeze(0)

            return self.__class__(
                merge([an_1, an_2]),
                merge([pos_1, pos_2]),
                cell=merge([cell_1, cell_2]),
            )
        else:
            return self.__class__(merge([an_1, an_2]), merge([pos_1, pos_2]))

    def __eq__(self, other: "Geometry") -> bool:
        """Check if two `Geometry` objects are equivalent."""
        # Note that batches with identical systems but a different order will
        # return False, not True.

        if self.__class__ != other.__class__:
            raise TypeError(
                f'"{self.__class__}" ==  "{other.__class__}" '
                f"evaluation not implemented."
            )

        def shape_and_value(a, b):
            return a.shape == b.shape and torch.allclose(a, b)

        return all(
            [
                shape_and_value(self.atomic_numbers, other.atomic_numbers),
                shape_and_value(self.positions, other.positions),
            ]
        )

    def __repr__(self) -> str:
        """Creates a string representation of the Geometry object."""
        # Return Geometry(CH4) for a single system & Geometry(CH4, H2O, ...)
        # for multiple systems. Only the first & last two systems get shown if
        # there are more than four systems (this prevents endless spam).

        def get_formula(atomic_numbers: Tensor) -> str:
            """Helper function to get reduced formula."""
            # If n atoms > 30; then use the reduced formula
            if len(atomic_numbers) > 30:
                return "".join(
                    [
                        (
                            f"{chemical_symbols[z]}{n}"
                            if n != 1
                            else f"{chemical_symbols[z]}"
                        )
                        for z, n in zip(
                            *atomic_numbers.unique(return_counts=True)
                        )
                        if z != 0
                    ]
                )  # <- Ignore zeros (padding)

            # Otherwise list the elements in the order they were specified
            else:
                return "".join(
                    [
                        (
                            f"{chemical_symbols[int(z)]}{int(n)}"
                            if n != 1
                            else f"{chemical_symbols[z]}"
                        )
                        for z, n in zip(
                            *torch.unique_consecutive(
                                atomic_numbers, return_counts=True
                            )
                        )
                        if z != 0
                    ]
                )

        if self.atomic_numbers.dim() == 1:  # If a single system
            formula = get_formula(self.atomic_numbers)
        else:  # If multiple systems
            if self.atomic_numbers.shape[0] < 4:  # If n<4 systems; show all
                formulas = [get_formula(an) for an in self.atomic_numbers]
                formula = " ,".join(formulas)
            else:  # If n>4; show only the first and last two systems
                formulas = [
                    get_formula(an)
                    for an in self.atomic_numbers[[0, 1, -2, -1]]
                ]
                formula = "{}, {}, ..., {}, {}".format(*formulas)

        # Wrap the formula(s) in the class name and return
        return f"{self.__class__.__name__}({formula})"

    def __str__(self) -> str:
        """Creates a printable representation of the System."""
        # Just redirect to the `__repr__` method
        return repr(self)


####################
# Helper Functions #
####################
def batch_chemical_symbols(
    atomic_numbers: Union[Tensor, List[Tensor]],
) -> list:
    """Converts atomic numbers to their chemical symbols.

    This function allows for en-mass conversion of atomic numbers to chemical
    symbols.

    Arguments:
        atomic_numbers: Atomic numbers of the elements.

    Returns:
        symbols: The corresponding chemical symbols.

    Notes:
        Padding vales, i.e. zeros, will be ignored.

    """
    a_nums = atomic_numbers

    # Catch for list tensors (still faster doing it this way)
    if isinstance(a_nums, list) and isinstance(a_nums[0], Tensor):
        a_nums = pack(a_nums, value=0)

    # Convert from atomic numbers to chemical symbols via a itemgetter
    symbols = np.array(  # numpy must be used as torch cant handle strings
        itemgetter(*a_nums.flatten())(chemical_symbols)
    ).reshape(a_nums.shape)
    # Mask out element "X", aka padding values
    mask = symbols != "X"
    if symbols.ndim == 1:
        return symbols[mask].tolist()
    else:
        return [s[m].tolist() for s, m in zip(symbols, mask)]


def unique_atom_pairs(
    geometry: Optional[Geometry] = None,
    unique_atomic_numbers: Optional[Tensor] = None,
    elements: list = None,
) -> Tensor:
    """Returns a tensor specifying all unique atom pairs.

    This takes `Geometry` instance and identifies all atom pairs. This use
    useful for identifying all possible two body interactions possible within
    a given system.

    Arguments:
         geometry: `Geometry` instance representing the target system.

    Returns:
        unique_atom_pairs: A tensor specifying all unique atom pairs.
    """
    if geometry is not None:
        uan = geometry.unique_atomic_numbers()
    elif unique_atomic_numbers is not None:
        uan = unique_atomic_numbers
    elif elements is not None:
        uan = torch.tensor([data_numbers[element] for element in elements])
    else:
        raise ValueError("Both geometry and unique_atomic_numbers are None.")

    n_global = len(uan)
    return torch.stack(
        [uan.repeat(n_global), uan.repeat_interleave(n_global)]
    ).T


def to_atomic_numbers(species: list) -> Tensor:
    """Return atomic numbers from element species."""
    return torch.tensor([chemical_symbols.index(isp) for isp in species])


def to_element_species(atomic_numbers: Union[Tensor]) -> list:
    """Return element species from atomic numbers."""
    assert atomic_numbers.dim() in (
        1,
        2,
    ), f"get input dimension {atomic_numbers.dim()} not 1 or 2"
    if atomic_numbers.dim() == 1:
        return [chemical_symbols[int(ia)] for ia in atomic_numbers]
    else:
        return [
            [chemical_symbols[int(ia)] for ia in atomic_number]
            for atomic_number in atomic_numbers
        ]


class GeometryPbcOneCell(Geometry):
    """Transfer periodic boundary condition to molecule like system."""

    def __init__(self, geometry: Geometry, periodic):
        assert geometry.is_periodic, "This class only works when PBC is True"
        self.geometry = geometry
        self.periodic = periodic
        # self.n_atoms = (self.geometry.n_atoms * self.geometry.periodic.ncell).long()
        self.n_atoms = (self.geometry.n_atoms * self.cell_mask.sum(-1)).long()

        self.pe_ind0 = self.periodic.ncell.repeat_interleave(
            self.geometry.n_atoms
        )
        self.pe_ind = self.cell_mask.sum(-1).repeat_interleave(
            self.geometry.n_atoms
        )

        # method 1, smaller size
        _atomic_numbers = self.geometry.atomic_numbers[
            self.geometry.atomic_numbers.ne(0)
        ]
        _atomic_numbers = torch.repeat_interleave(_atomic_numbers, self.pe_ind)
        self.atomic_numbers = pack(_atomic_numbers.split(tuple(self.n_atoms)))

        # method 2
        # self.atomic_numbers = pack([number.repeat_interleave(mask) for number, mask in zip(
        #     self.geometry.atomic_numbers, self.cell_mask.sum(-1).long())])

        self.positions = self._to_onecell()

        # Mask for clearing padding values in the distance matrix.
        if (temp_mask := self.atomic_numbers != 0).all():
            self._mask_dist: Union[Tensor, bool] = False
        else:
            self._mask_dist: Union[Tensor, bool] = ~(
                temp_mask.unsqueeze(-2) * temp_mask.unsqueeze(-1)
            )

    @property
    def cell_mask(self):
        return self.periodic.neighbour.any(-1).any(-1)

    @property
    def n_cell(self):
        return (self.n_atoms / self.n_central_atoms).long()

    @property
    def n_central_atoms(self):
        return self.geometry.n_atoms

    @property
    def distances(self) -> Tensor:
        dist = torch.cdist(self.positions, self.positions, p=2)
        # Ensure padding area is zeroed out
        dist[self._mask_dist] = 0.0

        # cdist bug, sometimes distances diagonal is not zero
        _ind = torch.arange(dist.shape[-1])
        if not (dist[..., _ind, _ind].eq(0)).all():
            dist[..., _ind, _ind] = 0

        return dist

    @property
    def distance_vectors(self) -> Tensor:
        """Distance vector matrix between atoms in the system."""
        dist_vec = self.positions.unsqueeze(-2) - self.positions.unsqueeze(-3)
        dist_vec[self._mask_dist] = 0
        return dist_vec

    def _to_onecell(self):
        """Transfer periodic positions to molecule like positions."""
        # Permute positions to make sure atomic numbers and positions are consistent
        _pos = self.periodic.positions_pe[self.cell_mask]
        _pos = _pos.split(tuple(self.cell_mask.sum(-1).tolist()), 0)

        # method 1, smaller size
        return pack(
            torch.cat(
                [
                    ipos[:ind, :ia].flatten(0, 1)
                    for ipos, ind, ia in zip(
                        _pos, self.cell_mask.sum(-1), self.geometry.n_atoms
                    )
                ]
            ).split(tuple(self.n_atoms))
        )

        # method 2
        # _pos = self.geometry.periodic.positions_pe[self.cell_mask]
        # return pack(_pos.split(tuple(self.cell_mask.sum(-1).tolist()), 0)).flatten(1, 2)

    @property
    def cell_mat(self) -> Tensor:
        """Return indices of cell vector of corresponding atoms."""
        # When write PBC cell to non-PBC like system, atoms in system come
        # from different cells, it's important to label the cell indices.
        # TODO, try to replace pack
        # method 1, smaller size
        cellvec = self.periodic.cellvec.permute(0, -1, 1)[self.cell_mask]
        cellvec = pack(
            [
                icell.repeat_interleave(ind, dim=0)
                for icell, ind in zip(
                    cellvec.split(tuple(self.cell_mask.sum(-1))),
                    self.geometry.n_atoms,
                )
            ],
            value=self.pad_values,
        )

        # method 2
        # cellvec = self.geometry.periodic.cellvec.permute(0, -1, 1)[self.cell_mask]
        # cellvec = pack(cellvec.split(tuple(self.cell_mask.sum(-1))), value=self.pad_values)
        # cellvec = pack(cellvec.repeat_interleave(self.geometry.n_atoms, dim=0)
        #                .split(tuple(self.geometry.n_atoms)), value=self.pad_values).flatten(1, 2)

        return cellvec

    @property
    def cell2d_mat(self) -> Tensor:
        """Return indices of cell and the shape is similar to distances."""
        tmp = self.cell_mat.unsqueeze(-2) + self.cell_mat.unsqueeze(-3)
        tmp1 = tmp - self.cell_mat.unsqueeze(-3)
        tmp2 = tmp - self.cell_mat.unsqueeze(-2)

        return torch.cat([tmp1, tmp2], dim=-1)

    @property
    def central_cell_ind(self):
        return (self.cell2d_mat[..., :3] == 0).sum(-1) == 3

    @property
    def is_periodic(self):
        """This is a quasi molecule-like geometry."""
        return False

    @property
    def pad_values(self):
        return 1e6


_pbc = ["cluster", "1d", "2d", "3d", "mix"]


class Pbc:
    """Cell class to deal with periodic boundary conditions.

    Arguments:
        cell: Atomic numbers of the atoms.
        frac :
        units: Unit in which ``positions`` were specified. For a list of
            available units see :mod:`.units`. [DEFAULT='bohr']

    Attributes:
        cell: Atomic numbers of the atoms.
        frac : Coordinates of the atoms.
        n_atoms: Number of atoms in the system.


    """

    def __init__(
        self,
        cell: Union[Tensor, List[Tensor]],
        frac=None,
        units: Optional[str] = "bohr",
        **kwargs,
    ):
        """Check cell type and dimension, transfer to batch tensor."""

        if isinstance(cell, list):
            cell = pack(cell)
        elif isinstance(cell, Tensor):
            if cell.dim() == 2:
                cell = cell.unsqueeze(0)
            elif cell.dim() < 2 or cell.dim() > 3:
                raise ValueError("input cell dimension is not 2 or 3")

        if cell.size(dim=-2) != 3:
            raise ValueError(
                "input cell should be defined by three lattice vectors"
            )

        # non-periodic systems in cell will be zero
        self.periodic_list = torch.tensor([ic.ne(0).any() for ic in cell])

        # some systems in batch is fraction coordinate
        if frac is not None:
            self.frac_list = (
                torch.stack([ii.ne(0).any() for ii in frac])
                & self.periodic_list
            )
        else:
            self.frac_list = torch.zeros(cell.size(0), dtype=bool)

        # transfer positions from angstrom to bohr
        if units != "bohr":
            cell: Tensor = cell * length_units[units]

        # Sum of the dimensions of periodic boundary condition
        sum_dim = cell.ne(0).any(-1).sum(dim=-1)

        if not torch.all(torch.tensor([isd == sum_dim[0] for isd in sum_dim])):
            self.pbc = [_pbc[isd] for isd in sum_dim]
        else:
            self.pbc = _pbc[sum_dim[0]]

        self.cell = cell

    @property
    def get_cell_lengths(self):
        """Get the length of each lattice vector."""
        return torch.linalg.norm(self.cell, dim=-1)

    @property
    def get_cell_angles(self):
        """Get the angles alpha, beta and gamma of lattice vectors."""
        _cos = torch.nn.CosineSimilarity(dim=0)
        cosine = torch.stack(
            [
                torch.tensor(
                    [
                        _cos(self.cell[ibatch, 1], self.cell[ibatch, 2]),
                        _cos(self.cell[ibatch, 0], self.cell[ibatch, 2]),
                        _cos(self.cell[ibatch, 0], self.cell[ibatch, 1]),
                    ]
                )
                for ibatch in range(self.cell.size(0))
            ]
        )
        return torch.acos(cosine) * 180 / np.pi


def unique_atom_pairs(
    geometry: Optional[Geometry] = None,
    unique_atomic_numbers: Optional[Tensor] = None,
    elements: list = None,
) -> Tensor:
    """Returns a tensor specifying all unique atom pairs.

    This takes `Geometry` instance and identifies all atom pairs. This use
    useful for identifying all possible two body interactions possible within
    a given system.

    Arguments:
         geometry: `Geometry` instance representing the target system.

    Returns:
        unique_atom_pairs: A tensor specifying all unique atom pairs.
    """
    if geometry is not None:
        uan = geometry.unique_atomic_numbers()
    elif unique_atomic_numbers is not None:
        uan = unique_atomic_numbers
    elif elements is not None:
        uan = torch.tensor([data_numbers[element] for element in elements])
    else:
        raise ValueError("Both geometry and unique_atomic_numbers are None.")

    n_global = len(uan)
    return torch.stack(
        [uan.repeat(n_global), uan.repeat_interleave(n_global)]
    ).T


def batch_chemical_symbols(
    atomic_numbers: Union[Tensor, List[Tensor]],
) -> list:
    """Converts atomic numbers to their chemical symbols.

    This function allows for en-mass conversion of atomic numbers to chemical
    symbols.

    Arguments:
        atomic_numbers: Atomic numbers of the elements.

    Returns:
        symbols: The corresponding chemical symbols.

    Notes:
        Padding vales, i.e. zeros, will be ignored.

    """
    a_nums = atomic_numbers

    # Catch for list tensors (still faster doing it this way)
    if isinstance(a_nums, list) and isinstance(a_nums[0], Tensor):
        a_nums = pack(a_nums, value=0)

    # Convert from atomic numbers to chemical symbols via a itemgetter
    symbols = np.array(  # numpy must be used as torch cant handle strings
        itemgetter(*a_nums.flatten())(chemical_symbols)
    ).reshape(a_nums.shape)
    # Mask out element "X", aka padding values
    mask = symbols != "X"
    if symbols.ndim == 1:
        return symbols[mask].tolist()
    else:
        return [s[m].tolist() for s, m in zip(symbols, mask)]


if __name__ == "__main__":
    import ase.io as io

    geo = io.read("POSCAR")  # or your geometry file
    geometry = Geometry.from_ase_atoms([geo])
    print(geometry)
