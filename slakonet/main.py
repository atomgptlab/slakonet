"""Enhanced SimpleDftb calculator with DOS, band gap, and band structure analysis."""

import torch
import ase.io as io
from slakonet.atoms import Geometry, Periodic
from slakonet.basis import Basis
from slakonet.skfeed import (
    SkfFeed,
    SkfParamFeed,
    _get_hs_dict,
    _get_onsite_dict,
)
from jarvis.core.kpoints import Kpoints3D as Kpoints
from ase import Atoms as AseAtoms
from jarvis.core.atoms import Atoms as JAtoms
from jarvis.core.specie import atomic_numbers_to_symbols
from jarvis.core.atoms import ase_to_atoms
from slakonet.slaterkoster import fermi, hs_matrix
from jarvis.core.atoms import Atoms
from slakonet.utils import eighb, pack
import matplotlib.pyplot as plt
from jarvis.core.specie import atomic_numbers_to_symbols

from slakonet.fermi import fermi_search, fermi_dirac, fermi_smearing
import numpy as np
from slakonet.utils import eighb, create_feeds, generate_shell_dict_upto_Z65
from slakonet.fermi import fermi_smearing
from slakonet.interpolation import PolyInterpU
import torch
import torch
from slakonet.atoms import Geometry
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms as AseAtoms
import numpy as np
import torch
from jarvis.core.atoms import Atoms as JarvisAtoms
from jarvis.core.atoms import ase_to_atoms
import torch.nn as nn

try:
    from phonopy import Phonopy
    from phonopy.file_IO import write_FORCE_CONSTANTS, write_disp_yaml
except Exception:
    pass
# torch.set_default_dtype(torch.float32)
# torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float32)
H2E = 27.211


class SimpleDftb:
    """Enhanced DFTB calculator for periodic systems with analysis tools."""

    def __init__(
        self,
        geometry,
        model,
        max_Z=100,
        # max_Z=65,
        cutoff=10.0,
        kpoints=None,
        klines=None,
        repulsive=True,
        with_eigenvectors=False,
        device=None,
        kT=0.025,  # eV for Fermi smearing
        H2E=27.211,  # Hartree to eV
        compute_forces=True,
        include_dos_data=True,
        include_HS=True,
        use_float32=True,
        alpha=0.1,
        beta=0.1,
        updated_skfs=None,
        fermi_surface=False,
        # shell_dict=None,
        # h_feed=None,
        # s_feed=None,
        # nelectron=None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_float32 = use_float32
        self.geometry = geometry
        self.model = model
        self.max_Z = max_Z
        self.cutoff = cutoff
        self.repulsive = repulsive
        self.with_eigenvectors = with_eigenvectors
        self.kT = kT
        self.H2E = H2E
        self.compute_forces = compute_forces
        self.include_dos_data = include_dos_data
        self.include_HS = include_HS
        # Setup basis and feeds
        self.shell_dict = generate_shell_dict_upto_Z65()
        self.basis = Basis(self.geometry.atomic_numbers, self.shell_dict)
        # Use provided SKFs or get from model
        if updated_skfs is not None:
            self.updated_skfs = updated_skfs
        else:
            self.updated_skfs = self.model.get_updated_skfs()

        self.h_feed = create_feeds(self.updated_skfs, self.shell_dict, "H")
        self.s_feed = create_feeds(self.updated_skfs, self.shell_dict, "S")
        # ===== CRITICAL: Store original kpoints/klines for force calculations =====
        self._original_kpoints = kpoints
        self._original_klines = klines
        # Precompute electron lookup table (gradient-safe)
        self.electron_lookup = self._build_electron_lookup()
        self.nelectron = self._compute_nelectrons()
        # print("self.geometry", self.geometry)
        # print("self.geometry.cell", self.geometry.cell)
        # Setup periodic boundary conditions
        self.periodic = Periodic(
            self.geometry,
            self.geometry.cell,
            cutoff=self.cutoff,
            kpoints=kpoints,
            klines=klines,
        )
        self.kpoints = self.periodic.kpoints
        self.k_weights = self.periodic.k_weights.to(self.device)
        self.max_nk = self.periodic.n_kpoints.max()

        # Cache for computed properties
        self._results = None
        self.alpha = alpha
        self.beta = beta
        self.fermi_surface = fermi_surface

    def calculate_dos(
        self,
        energy_range=(-10, 5),
        num_points=5000,
        sigma=0.1,
        fermi_shift=True,
        unit="eV",
    ):
        """
        Calculate density of states with Gaussian broadening.

        Parameters:
        -----------
        energy_range : tuple
            Energy range (E_min, E_max) for DOS calculation
        num_points : int
            Number of energy grid points
        sigma : float
            Gaussian broadening parameter (in same units as energy_range)
        fermi_shift : bool
            If True, shift energies so Fermi energy is at zero
        unit : str
            'eV' or 'Ha' for energy units (currently only 'eV' supported)

        Returns:
        --------
        tuple : (energy_grid, dos) both as torch.Tensors
        """
        # Ensure we have results
        if self._results is None:
            self.calculate()

        # Get eigenvalues (already in eV from calculate())
        eigenvals = self._results["eigenvalues"]  # [batch, nk, nb] in eV

        # Apply Fermi shift if requested
        if fermi_shift:
            # eigenvalues are already Fermi-shifted in calculate()
            pass  # Already shifted
        else:
            # Add Fermi energy back
            fermi_energy = self._results["fermi_energy"]  # in eV
            eigenvals = eigenvals + fermi_energy

        # Create energy grid
        energy_grid = torch.linspace(
            energy_range[0], energy_range[1], num_points, device=self.device
        )
        dos = torch.zeros(num_points, device=self.device)

        # Convert sigma to tensor on same device
        sigma_tensor = torch.tensor(
            sigma, device=self.device, dtype=energy_grid.dtype
        )

        # Gaussian broadening function (vectorized)
        def gaussian(x_grid, mu_val, sig):
            pi_tensor = torch.tensor(
                torch.pi, device=self.device, dtype=x_grid.dtype
            )
            return torch.exp(-0.5 * ((x_grid - mu_val) / sig) ** 2) / (
                sig * torch.sqrt(2 * pi_tensor)
            )

        # Calculate DOS using vectorized approach
        nbatch, nkpoints, nbands = eigenvals.shape

        for ik in range(nkpoints):
            # Get k-point weight - handle different k_weights shapes
            if len(self.k_weights.shape) == 2:
                weight = self.k_weights[0, ik]  # Extract scalar from 2D tensor
            elif ik < len(self.k_weights):
                weight = self.k_weights[ik]
            else:
                weight = torch.tensor(1.0 / nkpoints, device=self.device)

            # Get all bands for this k-point
            kpoint_eigenvals = eigenvals[0, ik, :]  # Shape: (nbands,)

            # Process each band individually
            for ib in range(nbands):
                eigenval = kpoint_eigenvals[ib]  # Single eigenvalue

                # Add Gaussian contribution for this eigenvalue
                gaussian_contrib = gaussian(
                    energy_grid, eigenval, sigma_tensor
                )
                dos += weight * gaussian_contrib

        return energy_grid, dos

    def _build_electron_lookup(self):
        """Build gradient-preserving electron count lookup table."""
        from jarvis.core.specie import Specie

        electron_table = torch.zeros(self.max_Z + 1, device=self.device)

        for pair_key, skf in self.updated_skfs.items():
            symbol = pair_key.split("-")[0]
            Z = Specie(symbol).Z

            skf_dict = skf.to_dict()
            if "atomic_data" in skf_dict and skf_dict["atomic_data"]:
                occupations = skf_dict["atomic_data"]["occupations"]
                if isinstance(occupations, list):
                    occupations = torch.tensor(occupations, device=self.device)
                electron_table[Z] = occupations.sum()

        return electron_table

    def _compute_nelectrons(self):
        """Compute total electrons using differentiable indexing."""
        atomic_nums = self.geometry.atomic_numbers.flatten()
        mask = atomic_nums != 0
        total_electrons = self.electron_lookup[atomic_nums[mask]].sum()
        return total_electrons.unsqueeze(0)

    def _solve_eigenvalue_problem(self, H, S):
        """Solve H*c = E*S*c with appropriate precision."""
        n_kpoints = self.max_nk.item()
        eigenvalues_list = []
        eigenvecs_list = []
        occupations_list = []

        for ik in range(n_kpoints):
            h_k = H[..., ik]
            s_k = S[..., ik]

            # CRITICAL: Use float64 for eigenvalue decomposition
            # This is where precision matters most
            if self.use_float32:
                h_k = h_k.to(torch.complex128)  # Complex128 for stability
                s_k = s_k.to(torch.complex128)

            # Solve generalized eigenvalue problem
            eigenvals, eigenvecs = eighb(h_k, s_k, scheme="chol")

            # Convert back to float32 after solve (if needed)
            if self.use_float32:
                eigenvals = eigenvals.to(torch.float32)
                if eigenvecs is not None:
                    eigenvecs = eigenvecs.to(torch.complex64)

            """
            # ===== CRITICAL FIX: Normalize eigenvectors =====
            if eigenvecs is not None:
                # Compute norms: <c|S|c> for each eigenvector
                if eigenvecs.is_complex():
                    # norms[i] = sqrt(<c_i|S|c_i>)
                    norms = torch.sqrt(
                        torch.sum(eigenvecs.conj() * (s_k @ eigenvecs), dim=0).real
                    )
                else:
                    norms = torch.sqrt(
                        torch.sum(eigenvecs * (s_k @ eigenvecs), dim=0)
                    )

                # Normalize: c_normalized = c / sqrt(<c|S|c>)
                eigenvecs = eigenvecs / norms.unsqueeze(0)
            # ===== End normalization =====
            """
            # Fermi occupation
            occ, _ = fermi(eigenvals, self.nelectron.to(self.device))

            eigenvalues_list.append(eigenvals)
            eigenvecs_list.append(eigenvecs)
            occupations_list.append(occ)

        # Stack and convert to eV
        eigenvalues = torch.stack(eigenvalues_list, dim=1) * self.H2E
        eigenvectors = (
            torch.stack(eigenvecs_list, dim=1)
            if self.with_eigenvectors
            else None
        )
        occupations = torch.stack(occupations_list, dim=1)

        return eigenvalues, eigenvectors, occupations

    def _compute_repulsive_energy(self):
        """Compute pair repulsive potential energy."""
        from jarvis.core.specie import atomic_numbers_to_symbols

        # Build atomic number to symbol mapping
        zz = list(range(1, self.max_Z))
        z = atomic_numbers_to_symbols(zz)
        atomic_num_to_symbol = dict(zip(zz, z))

        # Get distance matrix
        atom_pairs = self.basis.atomic_number_matrix("atomic").to(self.device)
        dist_vecs = self.periodic.distance_vectors.to(self.device)
        dist_mat = torch.sqrt((dist_vecs**2).sum(-1) + 1e-10)

        # Initialize total repulsive energy
        total_rep_energy = torch.zeros(1, device=self.device)

        # Get unique atom pairs
        uan = self.periodic.unique_atomic_numbers()
        n_global = len(uan)
        uap = torch.stack(
            [uan.repeat(n_global), uan.repeat_interleave(n_global)]
        ).T.to(self.device)

        # Loop over unique atom pairs
        for iap in uap:
            element_i = atomic_num_to_symbol.get(iap[0].item())
            element_j = atomic_num_to_symbol.get(iap[1].item())

            if element_i is None or element_j is None:
                continue

            element_pair = "-".join(tuple(sorted([element_i, element_j])))

            if element_pair not in self.updated_skfs:
                continue

            skf = self.updated_skfs[element_pair]
            if not skf.r_spline:
                return 0
            # Create atom pair mask
            mask_i = atom_pairs[..., 0] == iap[0]
            mask_j = atom_pairs[..., 1] == iap[1]
            mask_pair = mask_i & mask_j

            # Only non-zero distances
            mask_nonzero = dist_mat.gt(0.2)
            # mask_nonzero = dist_mat.gt(1e-8)
            r_cutoff = skf.r_spline.cutoff  # *0.529
            mask_cutoff = dist_mat < (r_cutoff / 0.529)
            mask = mask_pair & mask_nonzero & mask_cutoff
            # mask = mask_pair & mask_nonzero

            if not mask.any():
                continue

            if mask.any():
                d_masked = dist_mat[mask] * 0.529
                print(f"\n{element_pair}:")
                print(f"  Min distance: {d_masked.min():.3f} Å")
                print(f"  Max distance: {d_masked.max():.3f} Å")
                print(f"  Number of pairs: {len(d_masked)}")
                print(f"  Cutoff: {r_cutoff:.3f} Å")

            # After computing pair_energy:

            # Before the computation:
            for pair, skf in self.updated_skfs.items():
                if skf.r_spline:
                    print(f"\n{pair} SKF:")
                    print(
                        f"  Grid range: {skf.r_spline.grid.min():.3f} to {skf.r_spline.grid.max():.3f}"
                    )
                    print(f"  Cutoff: {skf.r_spline.cutoff:.3f}")
                    print(f"  Exp coef: {skf.r_spline.exp_coef}")
                    print(f"  Tail coef: {skf.r_spline.tail_coef}")

            # Get grid and coefficients
            grid = skf.r_spline.grid.to(self.device)  # *0.529  # *2
            exp_coef = skf.r_spline.exp_coef.to(self.device)  # * 27.211
            spline_coef = skf.r_spline.spline_coef.to(self.device)  # * 27.211
            tail_coef = skf.r_spline.tail_coef.to(self.device)  # * 27.211

            # Distance-based region masks (mutually exclusive!)
            in_tail = (d_masked >= grid[0]) & (d_masked <= grid[1])
            in_spline = (d_masked > grid[1]) & (d_masked < grid[-1])
            in_exp = (d_masked >= grid[-1]) & (d_masked < r_cutoff)

            # Initialize energy for this pair type
            pair_energy = torch.zeros_like(d_masked)

            # 1. Tail region (closest distances)
            if in_tail.any():
                d_tail = d_masked[in_tail]
                ind = torch.searchsorted(grid[:2], d_tail) - 1
                ind = torch.clamp(ind, 0, 0)  # Only one interval in tail
                dr = d_tail - grid[ind]

                pair_energy[in_tail] = (
                    tail_coef[0]
                    + tail_coef[1] * dr
                    + tail_coef[2] * dr**2
                    + tail_coef[3] * dr**3
                    + tail_coef[4] * dr**4
                    + tail_coef[5] * dr**5
                )

            # 2. Spline region (middle distances)
            if in_spline.any():
                d_spline = d_masked[in_spline]
                ind = torch.searchsorted(grid, d_spline) - 1
                ind = torch.clamp(ind, 0, len(grid) - 2)

                r_pol = spline_coef[ind]
                dr = d_spline - grid[ind]

                pair_energy[in_spline] = (
                    r_pol[..., 0]
                    + r_pol[..., 1] * dr
                    + r_pol[..., 2] * dr**2
                    + r_pol[..., 3] * dr**3
                )

            # 3. Exponential region (far distances)
            if in_exp.any():
                d_exp = d_masked[in_exp]
                pair_energy[in_exp] = (
                    torch.exp(-exp_coef[0] * d_exp + exp_coef[1]) + exp_coef[2]
                )

            if pair_energy.abs().max() > 10:  # Flag if any single pair > 10 eV
                print(f"  WARNING: Large repulsive energy detected!")
                print(f"  Max pair energy: {pair_energy.max():.2f} eV")
                print(f"  Min pair energy: {pair_energy.min():.2f} eV")
            # d_masked = dist_mat[mask]*0.529
            # Accumulate (0.5 to avoid double counting)
            total_rep_energy += 0.5 * pair_energy.sum()

        return total_rep_energy

    def _evaluate_wavefunction_on_grid(self, grid_cart, psi_coef, positions):
        """
        Evaluate wavefunction on a real-space grid.

        This is a simplified version using Gaussian-type orbitals.
        For production, you'd want to use the actual DFTB basis functions.
        """
        nx, ny, nz, _ = grid_cart.shape
        psi_grid = np.zeros(
            (nx, ny, nz), dtype=np.complex128
        )  # Changed to complex!

        # Get basis information
        atomic_numbers = self.geometry.atomic_numbers[0].cpu().numpy()

        orbital_idx = 0
        for iatom, Z in enumerate(atomic_numbers):
            if Z == 0:  # Skip padding
                continue

            pos = positions[iatom]  # [3]

            # Get shells for this atom
            shells = self.shell_dict.get(int(Z), [])

            for shell in shells:
                n_orbitals = 2 * shell + 1  # Number of orbitals in this shell

                for m in range(n_orbitals):
                    if orbital_idx >= len(psi_coef):
                        break

                    coef = psi_coef[orbital_idx]  # This is complex

                    # Simple Gaussian approximation for orbital
                    # In reality, you'd use proper Slater-Koster basis
                    zeta = 2.0  # Orbital exponent (simplified)

                    # Distance from atom to grid points
                    dx = grid_cart[..., 0] - pos[0]
                    dy = grid_cart[..., 1] - pos[1]
                    dz = grid_cart[..., 2] - pos[2]
                    r = np.sqrt(dx**2 + dy**2 + dz**2)

                    # Gaussian approximation: φ(r) ∝ exp(-ζr²)
                    if shell == 0:  # s orbital
                        orbital = np.exp(-zeta * r**2)
                    elif shell == 1:  # p orbital
                        if m == 0:
                            orbital = dx * np.exp(-zeta * r**2)
                        elif m == 1:
                            orbital = dy * np.exp(-zeta * r**2)
                        else:
                            orbital = dz * np.exp(-zeta * r**2)
                    else:  # d orbitals (simplified)
                        orbital = r**2 * np.exp(-zeta * r**2)

                    psi_grid += coef * orbital
                    orbital_idx += 1

        return psi_grid

    def calculate_charge_density(
        self,
        grid_spacing=0.1,  # Angstrom
        padding=2.0,  # Angstrom padding around cell
    ):
        """
        Calculate electron charge density on a real-space grid.

        Returns:
            grid_points: (nx, ny, nz, 3) array of grid point coordinates
            charge_density: (nx, ny, nz) array of charge density values
        """
        # Ensure we have eigenvectors
        if self._results is None or "eigenvectors" not in self._results:
            self.with_eigenvectors = True
            self.calculate()

        eigenvectors = self._results[
            "eigenvectors"
        ]  # [batch, nk, nbands, norbitals]
        occupations = self._results["occupations"]  # [batch, nk, nbands]

        # Get cell information
        cell = self.geometry.cell.cpu().numpy()[0]  # [3, 3]
        positions = (
            self.geometry.positions.detach().cpu().numpy()[0]
        )  # [natoms, 3]

        # Create real-space grid
        # Determine grid dimensions
        cell_lengths = np.linalg.norm(cell, axis=1)
        nx = int(np.ceil(cell_lengths[0] / grid_spacing)) + int(
            2 * padding / grid_spacing
        )
        ny = int(np.ceil(cell_lengths[1] / grid_spacing)) + int(
            2 * padding / grid_spacing
        )
        nz = int(np.ceil(cell_lengths[2] / grid_spacing)) + int(
            2 * padding / grid_spacing
        )

        # Create grid in fractional coordinates
        x = np.linspace(
            -padding / cell_lengths[0], 1 + padding / cell_lengths[0], nx
        )
        y = np.linspace(
            -padding / cell_lengths[1], 1 + padding / cell_lengths[1], ny
        )
        z = np.linspace(
            -padding / cell_lengths[2], 1 + padding / cell_lengths[2], nz
        )

        grid_frac = np.meshgrid(x, y, z, indexing="ij")
        grid_frac = np.stack(grid_frac, axis=-1)  # [nx, ny, nz, 3]

        # Convert to Cartesian coordinates
        grid_cart = np.dot(grid_frac, cell)  # [nx, ny, nz, 3]

        # Initialize charge density (real-valued)
        charge_density = np.zeros((nx, ny, nz), dtype=np.float64)

        # Evaluate wavefunctions on grid
        nbatch, nkpoints, nbands, norbitals = eigenvectors.shape

        print(f"Calculating charge density on {nx}×{ny}×{nz} grid...")

        for ik in range(nkpoints):
            # Get k-point weight
            if len(self.k_weights.shape) == 2:
                weight = self.k_weights[0, ik].cpu().item()
            else:
                weight = self.k_weights[ik].cpu().item()

            for ib in range(nbands):
                occ = occupations[0, ik, ib].cpu().item()
                if occ < 1e-6:  # Skip unoccupied states
                    continue

                # Get wavefunction coefficients for this k-point and band
                psi_coef = (
                    eigenvectors[0, ik, ib, :].detach().cpu().numpy()
                )  # [norbitals], complex

                # Evaluate atomic orbitals on grid and sum with coefficients
                psi_grid = self._evaluate_wavefunction_on_grid(
                    grid_cart, psi_coef, positions
                )  # Complex array

                # Add contribution to charge density: ρ += weight * occ * |ψ|²
                # Take absolute value squared to get real density
                charge_density += weight * occ * np.abs(psi_grid) ** 2

            print(f"  k-point {ik+1}/{nkpoints} done")

        print("✅ Charge density calculation complete")

        return grid_cart, charge_density

    def _compute_atomic_charges(self, eigenvectors, occupations, S):
        """
        Compute atomic charges using direct populations with internal normalization.

        This method is self-contained and handles eigenvector normalization internally,
        so it works even if eigenvectors from the solver are not normalized.

        For DFTB's tight-binding basis, Löwdin transformation is unnecessary
        and can introduce errors due to k-point averaging. Direct populations
        from the properly normalized density matrix work well.

        Parameters
        ----------
        eigenvectors : torch.Tensor [batch, n_k, n_bands, n_orbitals]
            Wavefunction coefficients (may be unnormalized)
        occupations : torch.Tensor [batch, n_k, n_bands]
            Occupation numbers (0 to 1)
        S : torch.Tensor [batch, n_orbitals, n_orbitals, n_k]
            Overlap matrix at each k-point

        Returns
        -------
        atomic_charges : torch.Tensor [batch, n_atoms]
            Atomic charges (positive = electron deficiency)
        """
        nbatch, nkpoints, nbands, norbitals = eigenvectors.shape
        device = eigenvectors.device
        dtype = eigenvectors.dtype
        atomic_nums = self.geometry.atomic_numbers[0]

        # Build density matrix with per-k-point normalization
        density_matrix = torch.zeros(
            norbitals, norbitals, dtype=dtype, device=device
        )

        for ik in range(nkpoints):
            # Get k-point weight
            if len(self.k_weights.shape) == 2:
                weight = self.k_weights[0, ik]
            else:
                weight = (
                    self.k_weights[ik]
                    if ik < len(self.k_weights)
                    else torch.tensor(1.0 / nkpoints, device=device)
                )

            # ===== CRITICAL: Get overlap for THIS k-point =====
            s_k = S[0, :, :, ik]
            if s_k.dtype != dtype:
                s_k = s_k.to(dtype)

            for ib in range(nbands):
                occ = occupations[0, ik, ib]
                if occ < 1e-8:
                    continue

                psi = eigenvectors[0, ik, ib, :]  # [norbitals]

                # ===== NORMALIZE eigenvector with THIS k-point's S =====
                if psi.is_complex():
                    norm_squared = (psi.conj() @ s_k @ psi).real
                else:
                    norm_squared = psi @ s_k @ psi

                # Check for numerical issues
                if norm_squared < 1e-10:
                    continue  # Skip degenerate/zero eigenvector

                norm = torch.sqrt(norm_squared)
                psi_normalized = psi / norm
                # ===== End normalization =====

                # Build density matrix with normalized eigenvector
                if psi_normalized.is_complex():
                    density_matrix += (
                        weight
                        * occ
                        * torch.outer(psi_normalized, psi_normalized.conj())
                    )
                else:
                    density_matrix += (
                        weight
                        * occ
                        * torch.outer(psi_normalized, psi_normalized)
                    )

        # Direct populations (no Löwdin transformation needed)
        if density_matrix.is_complex():
            orbital_populations = torch.real(torch.diag(density_matrix))
        else:
            orbital_populations = torch.diag(density_matrix)

        # Map orbitals to atoms
        orbital_to_atom = []
        for iatom, Z in enumerate(atomic_nums):
            Z_int = Z.item()
            if Z_int == 0:
                continue
            shells = self.shell_dict.get(Z_int, [])
            num_orbitals = sum(2 * shell + 1 for shell in shells)
            orbital_to_atom.extend([iatom] * num_orbitals)

        # Sum populations by atom
        atomic_populations = torch.zeros(len(atomic_nums), device=device)
        for orb_idx, atom_idx in enumerate(orbital_to_atom):
            if orb_idx < len(orbital_populations):
                atomic_populations[atom_idx] += orbital_populations[orb_idx]

        # Compute charges: Q = Z - N
        atomic_charges = torch.zeros(len(atomic_nums), device=device)
        for iatom, Z in enumerate(atomic_nums):
            Z_int = Z.item()
            if Z_int == 0:
                continue
            nuclear_charge = self.electron_lookup[Z_int]
            atomic_charges[iatom] = nuclear_charge - atomic_populations[iatom]

        return atomic_charges.unsqueeze(0)

    def plot_charge_density_slice(
        self, plane="xy", slice_coord=0.5, grid_spacing=0.2
    ):
        """
        Plot a 2D slice of the charge density.

        Args:
            plane: 'xy', 'xz', or 'yz'
            slice_coord: Fractional coordinate along the perpendicular axis (0 to 1)
            grid_spacing: Grid spacing in Angstrom
        """
        import matplotlib.pyplot as plt

        grid_cart, charge_density = self.calculate_charge_density(
            grid_spacing=grid_spacing
        )

        nx, ny, nz = charge_density.shape

        if plane == "xy":
            iz = int(slice_coord * nz)
            data = charge_density[:, :, iz]
            extent = [0, grid_cart.shape[1], 0, grid_cart.shape[0]]
            xlabel, ylabel = "Y (Å)", "X (Å)"
        elif plane == "xz":
            iy = int(slice_coord * ny)
            data = charge_density[:, iy, :]
            extent = [0, grid_cart.shape[2], 0, grid_cart.shape[0]]
            xlabel, ylabel = "Z (Å)", "X (Å)"
        else:  # yz
            ix = int(slice_coord * nx)
            data = charge_density[ix, :, :]
            extent = [0, grid_cart.shape[2], 0, grid_cart.shape[1]]
            xlabel, ylabel = "Z (Å)", "Y (Å)"

        plt.figure(figsize=(10, 8))
        plt.imshow(data, origin="lower", extent=extent, cmap="viridis")
        plt.colorbar(label="Charge Density (e/Å³)")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(
            f"Charge Density - {plane.upper()} plane at {slice_coord:.2f}"
        )

        # Overlay atomic positions
        positions = self.geometry.positions.detach().cpu().numpy()[0]
        if plane == "xy":
            plt.scatter(
                positions[:, 1],
                positions[:, 0],
                c="red",
                marker="o",
                s=100,
                edgecolors="white",
            )
        elif plane == "xz":
            plt.scatter(
                positions[:, 2],
                positions[:, 0],
                c="red",
                marker="o",
                s=100,
                edgecolors="white",
            )
        else:
            plt.scatter(
                positions[:, 2],
                positions[:, 1],
                c="red",
                marker="o",
                s=100,
                edgecolors="white",
            )

        plt.tight_layout()
        plt.savefig(f"charge_density_{plane}.png", dpi=300)
        plt.show()

    def calculate_planar_averaged_potential(
        self,
        axis="z",  # Direction perpendicular to interface
        num_points=200,
    ):
        """
        Calculate planar-averaged electrostatic potential along an axis.

        This is needed for band offset calculations at interfaces.

        Args:
            axis: 'x', 'y', or 'z' - direction to average along
            num_points: Number of points along the axis

        Returns:
            positions: Array of positions along axis (Angstrom)
            potential: Planar-averaged potential (eV)
        """
        # Ensure calculation is done
        if self._results is None:
            self.calculate()

        # Get charge density on grid
        grid_spacing = 0.2  # Angstrom
        grid_cart, charge_density = self.calculate_charge_density(
            grid_spacing=grid_spacing
        )

        # Determine axis
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis_idx = axis_map[axis.lower()]

        # Get cell information
        cell = self.geometry.cell.cpu().numpy()[0]
        cell_length = np.linalg.norm(cell[axis_idx])

        # Average over perpendicular directions
        other_axes = [0, 1, 2]
        other_axes.remove(axis_idx)
        other_axes = tuple(other_axes)

        # Planar average
        planar_avg = np.mean(charge_density, axis=other_axes)

        # Create position array
        positions = np.linspace(0, cell_length, len(planar_avg))

        # Convert charge density to potential using Poisson equation
        # For a simple approximation, integrate twice
        # ∇²V = -ρ/ε₀

        # This is simplified - for production, solve Poisson equation properly
        potential = self._charge_to_potential(planar_avg, positions)

        return positions, potential

    def _charge_to_potential(self, charge_density, positions):
        """
        Convert charge density to electrostatic potential.
        Simplified Poisson solver for 1D.
        """
        # Physical constants
        epsilon_0 = 8.854187817e-12  # F/m
        e_charge = 1.602176634e-19  # C

        # Convert units and solve Poisson equation
        # This is a simplified version - integrate charge density twice

        # First integration: electric field
        dx = positions[1] - positions[0] if len(positions) > 1 else 1.0
        electric_field = np.cumsum(charge_density) * dx

        # Second integration: potential
        potential = -np.cumsum(electric_field) * dx

        # Shift so minimum is at zero
        potential -= np.min(potential)

        return potential

    def calculate_macroscopic_average(
        self,
        positions,
        potential,
        period_length=None,
    ):
        """
        Calculate macroscopic average of potential (removes atomic oscillations).

        This implements the method from the InterMat code.

        Args:
            positions: Position array
            potential: Potential array
            period_length: Period for averaging (auto-detect if None)

        Returns:
            positions_avg: Positions for averaged potential
            potential_avg: Macroscopically averaged potential
        """
        from scipy.interpolate import CubicSpline

        # Create spline interpolation
        S = CubicSpline(positions, potential)

        # Auto-detect period if not provided
        if period_length is None:
            from scipy.signal import find_peaks

            # Find peaks to estimate periodicity
            peaks, _ = find_peaks(potential, prominence=0.1)

            if len(peaks) > 1:
                # Estimate period from peak spacing
                peak_spacing = np.diff(positions[peaks])
                period_length = np.median(peak_spacing)
            else:
                # Default fallback
                period_length = (positions[-1] - positions[0]) / 10

        print(f"Using period length: {period_length:.3f} Å")

        # Do macroscopic averaging
        positions_avg = []
        potential_avg = []

        for pos in positions:
            if pos - period_length / 2 < positions[0]:
                continue
            if pos + period_length / 2 > positions[-1]:
                continue

            # Integrate over period
            from scipy.integrate import quad

            integral = quad(
                S, pos - period_length / 2, pos + period_length / 2
            )[0]
            avg_val = integral / period_length

            positions_avg.append(pos)
            potential_avg.append(avg_val)

        return np.array(positions_avg), np.array(potential_avg)

    def calculate_interface_band_offset(
        self,
        film_region,  # (start, end) positions for film in Angstrom
        substrate_region,  # (start, end) positions for substrate in Angstrom
        film_bandgap,  # Film band gap (eV)
        substrate_bandgap,  # Substrate band gap (eV)
        axis="z",
        plot=True,
    ):
        """
        Calculate valence and conduction band offsets at an interface.

        Args:
            film_region: Tuple (start, end) defining film region
            substrate_region: Tuple (start, end) defining substrate region
            film_bandgap: Band gap of film material (eV)
            substrate_bandgap: Band gap of substrate material (eV)
            axis: Direction perpendicular to interface
            plot: Whether to create visualization

        Returns:
            dict with 'VBO' (valence band offset) and 'CBO' (conduction band offset)
        """
        # Calculate planar-averaged potential
        positions, potential = self.calculate_planar_averaged_potential(
            axis=axis
        )

        # Calculate macroscopic average
        pos_avg, pot_avg = self.calculate_macroscopic_average(
            positions, potential
        )

        # Get average potential in film region
        film_mask = (pos_avg >= film_region[0]) & (pos_avg <= film_region[1])
        film_potential = np.mean(pot_avg[film_mask])

        # Fit linear trend in film region for polar correction
        from numpy import ones, vstack
        from numpy.linalg import lstsq

        film_pos = pos_avg[film_mask]
        film_pot = pot_avg[film_mask]
        A = vstack([film_pos, ones(len(film_pos))]).T
        m_film, c_film = lstsq(A, film_pot, rcond=None)[0]

        # Get average potential in substrate region
        subs_mask = (pos_avg >= substrate_region[0]) & (
            pos_avg <= substrate_region[1]
        )
        substrate_potential = np.mean(pot_avg[subs_mask])

        # Fit linear trend in substrate region
        subs_pos = pos_avg[subs_mask]
        subs_pot = pot_avg[subs_mask]
        A = vstack([subs_pos, ones(len(subs_pos))]).T
        m_subs, c_subs = lstsq(A, subs_pot, rcond=None)[0]

        # Calculate potential difference
        # Check if polar (significant slope in either region)
        is_polar = abs(m_film) > 0.01 or abs(m_subs) > 0.01

        if is_polar:
            # For polar interfaces, use midpoint
            mid_point = (positions[-1] - positions[0]) / 2
            film_pot_mid = m_film * mid_point + c_film
            subs_pot_mid = m_subs * mid_point + c_subs
            delta_V = subs_pot_mid - film_pot_mid
            print(f"⚠️  Polar interface detected")
        else:
            delta_V = substrate_potential - film_potential

        print(f"Potential difference (ΔV): {delta_V:.3f} eV")

        # Calculate band offset
        # ΔE_v (VBM difference) = ΔV (from band gaps of bulk materials)
        # For now, assume we know the VBM positions from bulk calculations
        # In practice, you'd calculate this from bulk SlakoNet calculations

        # Get VBM and CBM from current calculation
        fermi = self._results["fermi_energy"].cpu().item()
        bandgap = self._results["bandgap"].cpu().item()

        # Valence Band Offset (simplified)
        # VBO = ΔV + (E_v^film - E_v^substrate)
        # where E_v are VBM energies from bulk calculations

        VBO = delta_V  # Simplified - add bulk corrections

        # Conduction Band Offset
        CBO = VBO + (substrate_bandgap - film_bandgap)

        results = {
            "VBO": VBO,
            "CBO": CBO,
            "delta_V": delta_V,
            "is_polar": is_polar,
            "film_potential": film_potential,
            "substrate_potential": substrate_potential,
        }

        if plot:
            self._plot_band_offset(
                positions,
                potential,
                pos_avg,
                pot_avg,
                film_region,
                substrate_region,
                results,
            )

        return results

    def get_fermi_surface(
        self,
        band_indices=None,
        nk=50,
        fermi_energy=None,
        energy_tolerance=0.05,
        plot=True,
        save_path=None,
    ):
        """
        Calculate and visualize Fermi surface for specified bands

        Args:
            band_indices: List of band indices to plot (None = auto-detect bands crossing E_F)
            nk: Number of k-points per direction for mesh
            fermi_energy: Fermi level in eV (None = use self._results["fermi_energy"])
            energy_tolerance: Energy window around E_F to consider (eV)
            plot: Whether to generate 3D visualization
            save_path: Path to save the figure

        Returns:
            Dictionary containing Fermi surface data and metrics
        """
        import numpy as np
        from scipy.interpolate import griddata

        # Step 1: Get Fermi level from results
        # if fermi_energy is None:
        if self._results is None:
            self.calculate()
        print(f"Using Fermi level from results: eV")
        fermi_energy = self._results["fermi_energy"]
        print(f"Using Fermi level from results:  eV", fermi_energy)

        # Step 2: Generate dense k-point mesh in first Brillouin zone
        kx = np.linspace(-0.5, 0.5, nk)
        ky = np.linspace(-0.5, 0.5, nk)
        kz = np.linspace(-0.5, 0.5, nk)
        kpoints = np.array(np.meshgrid(kx, ky, kz, indexing="ij"))
        kpoints_flat = kpoints.reshape(3, -1).T
        print(" kpoints", kpoints)
        print("kpoints_flat", kpoints_flat)
        # Step 3: Calculate band energies at all k-points
        print(f"Calculating bands at {len(kpoints_flat)} k-points...")
        all_bands = []

        for i, kpt in enumerate(kpoints_flat):
            if i % 10000 == 0:
                print(f"Progress: {i}/{len(kpoints_flat)}")

            eigenvals = self.solve_kpoint(kpt)
            all_bands.append(eigenvals)

        all_bands = np.array(all_bands)  # Shape: (n_kpts, n_bands)

        # Step 4: Identify bands crossing Fermi level
        if band_indices is None:
            band_indices = []
            for band_idx in range(all_bands.shape[1]):
                band_energies = all_bands[:, band_idx]
                if band_energies.min() < fermi_energy < band_energies.max():
                    band_indices.append(band_idx)
            print(
                f"Found {len(band_indices)} bands crossing Fermi level: {band_indices}"
            )

        if len(band_indices) == 0:
            print(
                "Warning: No bands cross the Fermi level. Material may be an insulator."
            )
            return None

        # Step 5: Extract Fermi surface data for each band
        fermi_surfaces = {}

        for band_idx in band_indices:
            band_energies = all_bands[:, band_idx].reshape(nk, nk, nk)

            # Classify as hole or electron pocket
            center_energy = band_energies[nk // 2, nk // 2, nk // 2]
            fs_type = "hole" if center_energy < fermi_energy else "electron"

            fermi_surfaces[band_idx] = {
                "energies": band_energies,
                "type": fs_type,
                "kpoints": kpoints,
            }

        # Step 6: Calculate Fermi surface metrics
        metrics = self._calculate_fermi_metrics(
            fermi_surfaces, fermi_energy, kpoints
        )

        # Step 7: Visualization
        if plot:
            self._plot_fermi_surfaces(
                fermi_surfaces, fermi_energy, energy_tolerance, save_path
            )

        return {
            "fermi_surfaces": fermi_surfaces,
            "fermi_energy": fermi_energy,
            "band_indices": band_indices,
            "metrics": metrics,
            "grid_shape": (nk, nk, nk),
        }

    def _calculate_fermi_metrics(self, fermi_surfaces, fermi_energy, kpoints):
        """Calculate properties from Fermi surface"""
        import numpy as np

        metrics = {}

        for band_idx, fs_data in fermi_surfaces.items():
            energies = fs_data["energies"]

            # Estimate Fermi surface area (number of points near E_F)
            near_fermi = np.abs(energies - fermi_energy) < 0.05
            surface_points = np.sum(near_fermi)

            # Calculate approximate effective mass from band curvature
            # Take central finite difference
            nk = energies.shape[0]
            center = nk // 2

            if center > 1 and center < nk - 1:
                # Second derivative in kx direction
                d2E_dkx2 = (
                    energies[center + 1, center, center]
                    + energies[center - 1, center, center]
                    - 2 * energies[center, center, center]
                )

                # Effective mass: m* = ℏ²/(d²E/dk²)
                hbar_eV_s = 6.582119569e-16  # eV·s
                dk = 1.0 / nk  # in reciprocal lattice units

                if abs(d2E_dkx2) > 1e-6:
                    m_star = hbar_eV_s**2 / (d2E_dkx2 / dk**2)
                    m_star_ratio = m_star / 9.109e-31  # ratio to electron mass
                else:
                    m_star_ratio = None
            else:
                m_star_ratio = None

            metrics[f"band_{band_idx}"] = {
                "type": fs_data["type"],
                "surface_points": int(surface_points),
                "effective_mass_ratio": m_star_ratio,
            }

        return metrics

    def _plot_fermi_surfaces(
        self, fermi_surfaces, fermi_energy, energy_tolerance, save_path
    ):
        """Generate 3D visualization of Fermi surfaces"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from skimage import measure
        except ImportError:
            print("Warning: matplotlib and scikit-image required for plotting")
            return

        n_bands = len(fermi_surfaces)
        fig = plt.figure(figsize=(6 * n_bands, 6))

        # Color schemes
        hole_colors = [
            "#FFD700",
            "#FFA500",
            "#FF8C00",
        ]  # Gold/Orange for holes
        electron_colors = [
            "#000000",
            "#FFA500",
            "#FFD700",
        ]  # Black/Orange for electrons

        for idx, (band_idx, fs_data) in enumerate(fermi_surfaces.items()):
            ax = fig.add_subplot(1, n_bands, idx + 1, projection="3d")

            energies = fs_data["energies"]
            kpoints = fs_data["kpoints"]
            fs_type = fs_data["type"]

            # Create isosurface at Fermi energy using marching cubes
            try:
                verts, faces, normals, values = measure.marching_cubes(
                    energies,
                    level=fermi_energy,
                    spacing=(1.0 / energies.shape[0],) * 3,
                )

                # Transform to k-space coordinates (-0.5 to 0.5)
                verts = verts - 0.5

                # Plot the surface
                colors = hole_colors if fs_type == "hole" else electron_colors

                ax.plot_trisurf(
                    verts[:, 0],
                    verts[:, 1],
                    verts[:, 2],
                    triangles=faces,
                    cmap="YlOrRd" if fs_type == "hole" else "binary",
                    alpha=0.8,
                    edgecolor="none",
                    shade=True,
                )

                # Draw Brillouin zone boundary (cube)
                self._draw_brillouin_zone(ax)

                ax.set_xlabel("$k_x$ (2π/a)", fontsize=12)
                ax.set_ylabel("$k_y$ (2π/a)", fontsize=12)
                ax.set_zlabel("$k_z$ (2π/a)", fontsize=12)
                ax.set_title(
                    f"Band number: {band_idx+1}\nFermi surface type: {fs_type}",
                    fontsize=14,
                    color="blue",
                )

                # Set equal aspect ratio
                ax.set_box_aspect([1, 1, 1])
                ax.set_xlim([-0.5, 0.5])
                ax.set_ylim([-0.5, 0.5])
                ax.set_zlim([-0.5, 0.5])

            except Exception as e:
                print(
                    f"Could not generate isosurface for band {band_idx}: {e}"
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved Fermi surface plot to {save_path}")

        plt.show()

    def _draw_brillouin_zone(self, ax, color="gray", alpha=0.3):
        """Draw first Brillouin zone boundary (cubic)"""
        import numpy as np

        # Define cube vertices
        r = 0.5
        vertices = [
            [-r, -r, -r],
            [r, -r, -r],
            [r, r, -r],
            [-r, r, -r],  # bottom
            [-r, -r, r],
            [r, -r, r],
            [r, r, r],
            [-r, r, r],  # top
        ]

        # Define edges
        edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # bottom face
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # top face
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # vertical edges
        ]

        for edge in edges:
            points = np.array([vertices[edge[0]], vertices[edge[1]]])
            ax.plot3D(*points.T, color=color, alpha=alpha, linewidth=1)

    def _plot_band_offset(
        self,
        positions,
        potential,
        pos_avg,
        pot_avg,
        film_region,
        substrate_region,
        results,
    ):
        """Create band offset visualization."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))

        # Plot raw potential
        plt.plot(positions, potential, "k-", alpha=0.3, label="Raw potential")

        # Plot macroscopic average
        plt.plot(
            pos_avg, pot_avg, "b-", linewidth=2, label="Macroscopic average"
        )

        # Mark film region
        plt.axvspan(
            film_region[0],
            film_region[1],
            alpha=0.2,
            color="red",
            label="Film",
        )
        film_pot = results["film_potential"]
        plt.axhline(y=film_pot, color="red", linestyle="--", alpha=0.5)

        # Mark substrate region
        plt.axvspan(
            substrate_region[0],
            substrate_region[1],
            alpha=0.2,
            color="blue",
            label="Substrate",
        )
        subs_pot = results["substrate_potential"]
        plt.axhline(y=subs_pot, color="blue", linestyle="--", alpha=0.5)

        # Add offset annotation
        mid_pos = (positions[-1] - positions[0]) / 2
        plt.annotate(
            f"ΔV = {results['delta_V']:.3f} eV\nVBO = {results['VBO']:.3f} eV\nCBO = {results['CBO']:.3f} eV",
            xy=(mid_pos, (film_pot + subs_pot) / 2),
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.xlabel("Position (Å)", fontsize=12)
        plt.ylabel("Potential (eV)", fontsize=12)
        plt.title("Interface Band Offset", fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("band_offset.png", dpi=300)
        print("✅ Band offset plot saved to band_offset.png")
        plt.close()

    def write_locpot(self, filename="LOCPOT", grid_spacing=0.1):
        """
        Write charge density in VASP LOCPOT format.
        """
        grid_cart, charge_density = self.calculate_charge_density(
            grid_spacing=grid_spacing
        )

        nx, ny, nz = charge_density.shape
        cell = self.geometry.cell.detach().cpu().numpy()[0]
        positions = self.geometry.positions.detach().cpu().numpy()[0]
        atomic_numbers = self.geometry.atomic_numbers[0].detach().cpu().numpy()

        # Convert to Bohr for VASP format
        cell_angstrom = cell * 0.529177  # Bohr to Angstrom

        with open(filename, "w") as f:
            # Header
            f.write("Charge density from SlakoNet\n")
            f.write("   1.00000000000000\n")

            # Lattice vectors
            for i in range(3):
                f.write(
                    f"  {cell_angstrom[i,0]:21.16f} {cell_angstrom[i,1]:21.16f} {cell_angstrom[i,2]:21.16f}\n"
                )

            # Atom types and counts
            from jarvis.core.specie import atomic_numbers_to_symbols

            unique_Z = np.unique(atomic_numbers[atomic_numbers > 0])
            symbols = atomic_numbers_to_symbols(unique_Z.tolist())

            f.write("  " + "  ".join(symbols) + "\n")

            counts = [np.sum(atomic_numbers == z) for z in unique_Z]
            f.write("  " + "  ".join(map(str, counts)) + "\n")

            f.write("Direct\n")

            # Atomic positions (fractional)
            for z in unique_Z:
                mask = atomic_numbers == z
                atom_pos = positions[mask]
                # Convert to fractional coordinates
                frac_pos = np.linalg.solve(cell.T, atom_pos.T).T
                for pos in frac_pos:
                    f.write(
                        f"  {pos[0]:19.16f} {pos[1]:19.16f} {pos[2]:19.16f}\n"
                    )

            f.write("\n")
            f.write(f"  {nx} {ny} {nz}\n")

            # Write charge density
            # VASP format: fast index z, then y, then x
            density_flat = charge_density.transpose(2, 1, 0).flatten()

            for i, val in enumerate(density_flat):
                f.write(f" {val:17.11E}")
                if (i + 1) % 5 == 0:
                    f.write("\n")

            if len(density_flat) % 5 != 0:
                f.write("\n")

        print(f"✅ Charge density written to {filename}")

    def calculate(self):
        """Main calculation method."""
        compute_forces = self.compute_forces
        include_dos_data = self.include_dos_data
        include_HS = self.include_HS
        # Enable gradient tracking for positions if forces needed
        if compute_forces:
            needs_recreation = (
                not self.geometry.positions.requires_grad
                or not self.geometry.cell.requires_grad
            )
            if needs_recreation:
                # CRITICAL: Set requires_grad BEFORE creating Periodic object
                # if not self.geometry.positions.requires_grad:
                self.geometry.positions.requires_grad_(True)
                self.geometry.cell.requires_grad_(True)
                self.periodic = Periodic(
                    self.geometry,
                    self.geometry.cell,
                    cutoff=self.cutoff,
                    kpoints=self._original_kpoints,  # Use stored original
                    klines=self._original_klines,  # Use stored original
                )
                self.k_weights = self.periodic.k_weights.to(self.device)
                self.max_nk = self.periodic.n_kpoints.max()
        # Build Hamiltonian and overlap matrices
        H = hs_matrix(self.periodic, self.basis, self.h_feed).to(self.device)
        S = hs_matrix(self.periodic, self.basis, self.s_feed).to(self.device)

        # Solve eigenvalue problem
        eigenvalues, eigenvectors, occupations = (
            self._solve_eigenvalue_problem(H, S)
        )

        # Compute Fermi energy
        fermi_energy = fermi_search(
            eigenvalues=eigenvalues,
            n_electrons=self.nelectron,
            k_weights=self.k_weights,
            kT=self.kT,
        )

        # Electronic energy
        electronic_energy = torch.sum(
            occupations * eigenvalues * self.k_weights.unsqueeze(-1)
        )

        if self.repulsive:
            potential_energy = self._compute_repulsive_energy()  # * self.H2E
            total_energy = self.alpha * electronic_energy + potential_energy
            print("potential_energy", potential_energy)
            print("electronic_energy", electronic_energy)
        else:
            potential_energy = torch.tensor(0.0, device=self.device)
            total_energy = electronic_energy

        # Shift eigenvalues relative to Fermi level
        eigenvalues_shifted = eigenvalues - fermi_energy

        # Compute bandgap
        Ef_expanded = fermi_energy.view(-1, 1, 1)
        occ = eigenvalues <= Ef_expanded
        unocc = eigenvalues > Ef_expanded

        vbm = torch.where(
            occ,
            eigenvalues,
            torch.tensor(
                float("-inf"), dtype=eigenvalues.dtype, device=self.device
            ),
        )
        cbm = torch.where(
            unocc,
            eigenvalues,
            torch.tensor(
                float("inf"), dtype=eigenvalues.dtype, device=self.device
            ),
        )

        vbm_val = vbm.max(dim=-1)[0].max(dim=-1)[0]
        cbm_val = cbm.min(dim=-1)[0].min(dim=-1)[0]
        bandgap = (cbm_val - vbm_val).clamp(min=0.0)

        # Compute forces
        forces = None
        stress = None
        if compute_forces:
            try:
                grad_outputs = torch.autograd.grad(
                    total_energy,
                    self.geometry.positions,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=False,
                )

                forces = (
                    -self.beta * grad_outputs[0]
                )  # Forces are negative gradient
                # print("forcessssssss", forces)
                dE_dh = torch.autograd.grad(
                    total_energy,
                    self.geometry.cell,
                    retain_graph=True,
                    create_graph=True,
                )[0]
                cell = self.geometry.cell[0]
                volume = torch.abs(torch.det(cell))
                # stress = 160.21766208* dE_dh[0] @ cell.T / volume
                # ===== VIRIAL TERM =====
                positions = self.geometry.positions[0]
                mask = self.geometry.atomic_numbers[0] > 0
                stress_virial = torch.einsum(
                    "ia,ib->ab", forces[0][mask], positions[mask]
                )

                # ===== TOTAL STRESS =====
                stress_tensor = (stress_virial - dE_dh[0] @ cell.T) / volume

                # Voigt + GPa
                stress = (
                    torch.tensor(
                        [
                            stress_tensor[0, 0],
                            stress_tensor[1, 1],
                            stress_tensor[2, 2],
                            stress_tensor[1, 2],
                            stress_tensor[0, 2],
                            stress_tensor[0, 1],
                        ],
                        device=self.device,
                    )
                    * 160.21766208
                )

            except RuntimeError as e:
                print(f"❌ Error computing forces: {e}")
                print(
                    "⚠️  Forces set to zero - positions may not be in computation graph"
                )
                forces = torch.zeros_like(self.geometry.positions)
        # print('forces',forces)
        self._results = {
            "energy": total_energy,
            "eigenvalues": eigenvalues_shifted,
            "fermi_energy": fermi_energy,
            "electronic_energy": electronic_energy,
            "potential_energy": potential_energy,
            "forces": forces,
            "stress": stress,
            "bandgap": bandgap,
            "occupations": occupations,
            "geometry": self.geometry,
            "basis": self.basis,
        }
        # Calculate DOS data if needed
        # include_dos_data=False
        if include_dos_data:
            energy_grid, dos = self.calculate_dos(
                energy_range=(-10, 10),
                num_points=5000,
                sigma=0.1,
                fermi_shift=True,
            )

            fermi_idx = torch.argmin(torch.abs(energy_grid))
            dos_at_fermi = dos[fermi_idx]

            fermi_region_mask = torch.abs(energy_grid) < 2.0
            fermi_region_dos = dos[fermi_region_mask]
            fermi_region_energies = energy_grid[fermi_region_mask]

            min_dos_idx_local = torch.argmin(fermi_region_dos)
            gap_center_energy = fermi_region_energies[min_dos_idx_local]

            dx = energy_grid[1] - energy_grid[0]
            total_states = torch.trapz(dos, energy_grid)

            dos_data = {
                "dos_at_fermi": dos_at_fermi.item(),
                "dos_gap_center_eV": gap_center_energy.item(),
                "dos_total_states": total_states.item(),
                "dos_at_fermi_tensor": dos_at_fermi,
                "dos_gap_center_tensor": gap_center_energy,
                "dos_total_states_tensor": total_states,
                "dos_energy_grid_tensor": energy_grid,
                "dos_values_tensor": dos,
            }
        else:
            dos_data = {}
        self._results.update(dos_data)

        # Store results
        if self.include_HS:
            self._results["hamiltonian"] = H
            self._results["overlap"] = S
        # print("self.with_eigenvectors", self.with_eigenvectors)
        if self.fermi_surface:
            print("get_fermi_surface", "HERE")
            self.get_fermi_surface()

        if self.with_eigenvectors:
            self._results["eigenvectors"] = eigenvectors
            """
            atomic_charges_mulliken = self._compute_mulliken_charges_debug(
            #atomic_charges_mulliken = self._compute_mulliken_charges(
                eigenvectors, occupations, S
            )
            """
            atomic_charges = self._compute_atomic_charges(
                eigenvectors, occupations, S
            )
            charge_data = {"charges": atomic_charges}
            self._results.update(charge_data)
        return self._results

    # Properties for easy access
    @property
    def energy(self):
        if self._results is None:
            self.calculate()
        return self._results["energy"]

    @property
    def eigenvalues(self):
        if self._results is None:
            self.calculate()
        return self._results["eigenvalues"]

    @property
    def fermi_energy(self):
        if self._results is None:
            self.calculate()
        return self._results["fermi_energy"]

    @property
    def electronic_energy(self):
        if self._results is None:
            self.calculate()
        return self._results["electronic_energy"]

    @property
    def potential_energy(self):
        if self._results is None:
            self.calculate()
        return self._results["potential_energy"]

    @property
    def forces(self):
        # TODO: Check running calculate twice??
        if self._results is None:
            self.calculate()
        return self._results["forces"]

    @property
    def bandgap(self):
        if self._results is None:
            self.calculate()
        return self._results["bandgap"]

    @property
    def occupations(self):
        if self._results is None:
            self.calculate()
        return self._results["occupations"]

    @property
    def hamiltonian(self):
        if self._results is None:
            self.calculate()
        return self._results["hamiltonian"]

    @property
    def overlap(self):
        if self._results is None:
            self.calculate()
        return self._results["overlap"]


def run_calc(
    ase_atoms=None,
    model_path=None,
    model=None,
    kpoints_array=[1, 1, 1],
    device="cuda",
    compute_forces=True,
    alpha=0.1,
    beta=0.1,
    elements_needed=None,
    updated_skfs=None,  # NEW
    with_eigenvectors=False,
):
    if model is None:
        if model_path is None:
            raise ValueError("Either model or model_path must be provided")
        from slakonet.optim import MultiElementSkfParameterOptimizer

        model = MultiElementSkfParameterOptimizer.load_ultra_compact_lazy(
            model_path, elements_needed=elements_needed
        )
        model = model.to(device).float()
        model.eval()

    model.eval()

    geometry = Geometry.from_ase_atoms([ase_atoms])
    geometry.positions.requires_grad_(True)
    kpoints = torch.tensor(kpoints_array)

    s = SimpleDftb(
        geometry,
        kpoints=kpoints,
        model=model,
        compute_forces=compute_forces,
        alpha=alpha,
        beta=beta,
        updated_skfs=updated_skfs,  # Pass filtered SKFs
    )
    res = s.calculate()
    return res


class FilteredModelView(nn.Module):
    """Lightweight wrapper that filters which SKF pairs are used"""

    def __init__(self, full_model, elements_needed):
        super().__init__()
        self.full_model = full_model
        self.elements_needed = set(elements_needed)

        # Determine which pairs to use
        self.active_pairs = set()
        for pair_key in full_model.skf_optimizers.keys():
            elem1, elem2 = pair_key.split("-")
            if elem1 in self.elements_needed and elem2 in self.elements_needed:
                self.active_pairs.add(pair_key)

        print(
            f"   Using {len(self.active_pairs)} pairs: {sorted(self.active_pairs)}"
        )

    def get_updated_skfs(self):
        """Return only the SKFs for active elements"""
        updated_skfs = {}
        for pair_key in self.active_pairs:
            if pair_key in self.full_model.skf_optimizers:
                updated_skfs[pair_key] = self.full_model.skf_optimizers[
                    pair_key
                ].get_updated_skf()
        return updated_skfs

    def to(self, device):
        # The full model is already on device, just return self
        return self

    def float(self):
        return self

    def eval(self):
        return self

    def __getattr__(self, name):
        # Delegate other attributes to full_model
        if name in ["full_model", "elements_needed", "active_pairs"]:
            return object.__getattribute__(self, name)
        return getattr(self.full_model, name)


class SlakoNetCalculator(Calculator):
    """ASE Calculator interface for SlakoNet with dynamic element filtering"""

    implemented_properties = [
        "energy",
        "forces",
        "stress",
        "bandgap",
        "eigenvalues",
        "fermi_energy",
    ]

    # Class-level model cache for sharing across instances
    _model_cache = {}

    def __init__(
        self,
        model=None,
        model_path=None,
        kpoints_array=[1, 1, 1],
        device="cuda",
        alpha=0.1,
        beta=0.1,
        compute_forces=True,
        elements_needed=None,
        use_cached_model=True,
        with_eigenvectors=False,
        **kwargs,
    ):
        """
        Initialize SlakoNet calculator

        Args:
            model: Loaded SlakoNet model (MultiElementSkfParameterOptimizer)
            model_path: Path to model file (if model not provided)
            use_cached_model: If True, reuse loaded model across calculator instances
            elements_needed: Set of elements to filter (optional, can auto-detect)
            **kwargs: Additional arguments passed to Calculator
        """
        Calculator.__init__(self, **kwargs)

        # Load or reuse model
        if model is None and model_path is not None:
            if use_cached_model and model_path in self._model_cache:
                print(f"♻️  Reusing cached model")
                model = self._model_cache[model_path]
            else:
                from slakonet.optim import MultiElementSkfParameterOptimizer

                print(f"🔄 Loading full model from {model_path}...")

                # Load FULL model with caching
                model = MultiElementSkfParameterOptimizer.load_with_cache(
                    model_path, cache_dir=".model_cache"
                )

                # Move to device and set dtype once
                model = model.to(device).float()
                model.eval()

                # Cache for reuse
                if use_cached_model:
                    self._model_cache[model_path] = model
                    print(f"✅ Model cached for reuse")

        elif model is None:
            raise ValueError("Either model or model_path must be provided")

        # Store the full model
        self.full_model = model

        # Store settings
        self.model_path = model_path
        self.kpoints_array = kpoints_array
        self.device = device
        self.compute_forces = compute_forces
        self.alpha = alpha
        self.beta = beta
        self.with_eigenvectors = with_eigenvectors

        # Initialize with elements if provided
        self.elements_needed = None
        self.active_pairs = None

        if elements_needed:
            self.set_elements(elements_needed)

    def set_elements(self, elements_needed):
        """
        Set which elements to use for calculations.
        This filters the model to only use relevant element pairs.

        Args:
            elements_needed: Set or list of element symbols (e.g., {'Si', 'Ge'})
        """
        if elements_needed is None:
            self.elements_needed = None
            self.active_pairs = None
            return

        elements_needed = set(elements_needed)

        # Only update if elements changed
        if self.elements_needed == elements_needed:
            return

        self.elements_needed = elements_needed

        # Determine which pairs are needed
        self.active_pairs = set()
        for pair_key in self.full_model.skf_optimizers.keys():
            elem1, elem2 = pair_key.split("-")
            if elem1 in elements_needed and elem2 in elements_needed:
                self.active_pairs.add(pair_key)

        # Silent mode for high-throughput
        # print(f"🎯 Using {len(self.active_pairs)} pairs for {sorted(elements_needed)}")

    def _get_filtered_skfs(self):
        """Get SKFs filtered by active element pairs"""
        if self.active_pairs is None:
            # No filtering, return all
            return self.full_model.get_updated_skfs()

        # Return only active pairs
        updated_skfs = {}
        for pair_key in self.active_pairs:
            if pair_key in self.full_model.skf_optimizers:
                updated_skfs[pair_key] = self.full_model.skf_optimizers[
                    pair_key
                ].get_updated_skf()
        return updated_skfs

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces", "bandgap", "eigenvalues", "results"],
        system_changes=all_changes,
    ):
        """
        Calculate properties using SlakoNet

        Args:
            atoms: ASE Atoms object
            properties: List of properties to calculate
            system_changes: Changes since last calculation
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        # Auto-detect elements from structure if not set
        if self.elements_needed is None:
            elements_in_structure = set(atoms.get_chemical_symbols())
            self.set_elements(elements_in_structure)
        # Get filtered SKFs
        filtered_skfs = self._get_filtered_skfs()

        # Run calculation with filtered model
        result = self._run_calc_with_filtered_skfs(
            atoms=atoms,
            filtered_skfs=filtered_skfs,
        )

        # Extract results and convert to numpy arrays on CPU
        self.results["energy"] = result["energy"].detach().cpu().numpy().item()
        self.results["result"] = result
        if "forces" in properties and result.get("forces") is not None:
            forces = result["forces"].detach().cpu().numpy()
            self.results["forces"] = forces.reshape(-1, 3)
        if "stress" in properties and result.get("stress") is not None:
            stress = result["stress"].detach().cpu().numpy()
            self.results["stress"] = stress.reshape(-1, 3)
            # self.results["stress"] = 160.21766208*stress.reshape(-1, 3)

        if "fermi_energy" in result:
            self.results["fermi_energy"] = (
                result["fermi_energy"].detach().cpu().numpy().item()
            )
        if "bandgap" in result:
            self.results["bandgap"] = (
                result["bandgap"].detach().cpu().numpy().item()
            )

        if "eigenvalues" in result:
            self.results["eigenvalues"] = (
                result["eigenvalues"].detach().cpu().numpy()
            )

    def get_bandgap(self):
        """Convenience method to get bandgap"""
        if "bandgap" not in self.results:
            raise RuntimeError("Calculation not performed yet")
        return self.results["bandgap"]

    def get_fermi_energy(self):
        """Convenience method to get Fermi energy"""
        if "fermi_energy" not in self.results:
            raise RuntimeError("Calculation not performed yet")
        return self.results["fermi_energy"]

    def _run_calc_with_filtered_skfs(self, atoms, filtered_skfs):
        """Run calculation using only filtered SKF pairs"""
        from slakonet.atoms import Geometry
        from slakonet.main import SimpleDftb

        # Create a temporary filtered model wrapper
        class FilteredModel:
            def __init__(self, filtered_skfs):
                self.filtered_skfs = filtered_skfs

            def get_updated_skfs(self):
                return self.filtered_skfs

            def to(self, device):
                return self

            def float(self):
                return self

            def eval(self):
                return self

        # Create geometry
        geometry = Geometry.from_ase_atoms([atoms])
        geometry.positions.requires_grad_(True)
        kpoints = torch.tensor(self.kpoints_array)

        # Use filtered model
        filtered_model = FilteredModel(filtered_skfs)

        # Run SimpleDftb with filtered model
        calc = SimpleDftb(
            geometry,
            kpoints=kpoints,
            model=filtered_model,
            compute_forces=self.compute_forces,
            alpha=self.alpha,
            beta=self.beta,
            device=self.device,
            with_eigenvectors=self.with_eigenvectors,
        )

        result = calc.calculate()
        return result


class SlakoNetCalculatorX(Calculator):
    """ASE Calculator interface for SlakoNet with dynamic element filtering"""

    implemented_properties = ["energy", "forces", "stress"]

    # Class-level model cache for sharing across instances
    _model_cache = {}

    def __init__(
        self,
        model=None,
        model_path=None,
        kpoints_array=[1, 1, 1],
        device="cuda",
        alpha=0.1,
        beta=0.1,
        compute_forces=True,
        elements_needed=None,
        use_cached_model=True,  # NEW: Enable model reuse
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)

        # Load or reuse model
        if model is None and model_path is not None:
            if use_cached_model and model_path in self._model_cache:
                print(f"♻️  Reusing cached model for {model_path}")
                model = self._model_cache[model_path]
            else:
                from slakonet.optim import MultiElementSkfParameterOptimizer

                print(f"🔄 Loading full model from {model_path}...")

                # Load FULL model with caching
                model = MultiElementSkfParameterOptimizer.load_with_cache(
                    model_path, cache_dir=".model_cache"
                )

                # Cache for reuse
                if use_cached_model:
                    self._model_cache[model_path] = model

        elif model is None:
            raise ValueError("Either model or model_path must be provided")

        # Prepare model once
        self.full_model = model.to(device).float()
        self.full_model.eval()

        # Store settings
        self.model_path = model_path
        self.elements_needed = elements_needed
        self.kpoints_array = kpoints_array
        self.device = device
        self.compute_forces = compute_forces
        self.alpha = alpha
        self.beta = beta

        # Create filtered view if elements specified
        if elements_needed:
            print(f"🎯 Filtering model for elements: {elements_needed}")
            self.active_model = self._create_filtered_model(elements_needed)
        else:
            self.active_model = self.full_model

    def _create_filtered_model(self, elements_needed):
        """Create a lightweight filtered view of the model"""
        # This creates a wrapper that only uses specified element pairs
        # without copying the entire model
        return FilteredModelView(self.full_model, elements_needed)

    def set_elements(self, elements_needed):
        """Dynamically change which elements to use"""
        if elements_needed:
            print(f"🔄 Switching to elements: {elements_needed}")
            self.active_model = self._create_filtered_model(elements_needed)
            self.elements_needed = elements_needed
        else:
            self.active_model = self.full_model
            self.elements_needed = None

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces", "bandgap", "eigenvalues", "results"],
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        # Auto-detect elements if not specified
        if self.elements_needed is None:
            elements_in_structure = set(atoms.get_chemical_symbols())
            self.set_elements(elements_in_structure)

        # Use the active (possibly filtered) model
        result = run_calc(
            ase_atoms=atoms,
            model=self.active_model,
            model_path=None,
            device=self.device,
            kpoints_array=self.kpoints_array,
            compute_forces=self.compute_forces,
            alpha=self.alpha,
            beta=self.beta,
            elements_needed=None,
        )

        # Extract results
        self.results["energy"] = result["energy"].detach().cpu().numpy().item()
        self.results["result"] = result

        if "forces" in properties:
            forces = result["forces"].detach().cpu().numpy()
            self.results["forces"] = forces.reshape(-1, 3)

        if "fermi_energy" in result:
            self.results["fermi_energy"] = (
                result["fermi_energy"].detach().cpu().numpy().item()
            )

        if "bandgap" in result:
            self.results["bandgap"] = (
                result["bandgap"].detach().cpu().numpy().item()
            )

        if "eigenvalues" in result:
            self.results["eigenvalues"] = (
                result["eigenvalues"].detach().cpu().numpy()
            )


class SlakoNetCalculatorX(Calculator):
    """ASE Calculator interface for SlakoNet"""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model=None,
        model_path=None,
        kpoints_array=[1, 1, 1],
        device="cuda",
        alpha=0.1,
        beta=0.1,
        compute_forces=True,
        elements_needed=None,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)

        # Load model if needed
        if model is None and model_path is not None:
            from slakonet.optim import MultiElementSkfParameterOptimizer

            print(f"🔄 Loading model from {model_path}...")
            model = MultiElementSkfParameterOptimizer.load_ultra_compact_lazy(
                model_path, elements_needed=elements_needed
            )
        elif model is None:
            raise ValueError("Either model or model_path must be provided")

        # ✅ PREPARE MODEL ONCE AT INITIALIZATION
        self.model = model.to(device).float()
        self.model.eval()

        # Store other settings
        self.model_path = model_path
        self.elements_needed = elements_needed
        self.kpoints_array = kpoints_array
        self.device = device
        self.compute_forces = compute_forces
        self.alpha = alpha
        self.beta = beta

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces", "bandgap", "eigenvalues", "results"],
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        # ✅ Use pre-prepared model (no loading/moving/converting here)
        result = run_calc(
            ase_atoms=atoms,
            model=self.model,  # Pass the already-prepared model
            model_path=None,  # Don't reload
            device=self.device,
            kpoints_array=self.kpoints_array,
            compute_forces=self.compute_forces,
            alpha=self.alpha,
            beta=self.beta,
            elements_needed=None,  # Already filtered at init
        )

        # Extract results
        self.results["energy"] = result["energy"].detach().cpu().numpy().item()
        self.results["result"] = result

        if "forces" in properties:
            forces = result["forces"].detach().cpu().numpy()
            self.results["forces"] = forces.reshape(-1, 3)

        if "fermi_energy" in result:
            self.results["fermi_energy"] = (
                result["fermi_energy"].detach().cpu().numpy().item()
            )

        if "bandgap" in result:
            self.results["bandgap"] = (
                result["bandgap"].detach().cpu().numpy().item()
            )

        if "eigenvalues" in result:
            self.results["eigenvalues"] = (
                result["eigenvalues"].detach().cpu().numpy()
            )


# Example usage
if __name__ == "__main__":

    from ase.build import bulk
    from ase.optimize import BFGS
    from slakonet.optim import MultiElementSkfParameterOptimizer

    model_path = "Si_only.pt"
    model = MultiElementSkfParameterOptimizer.load_ultra_compact(model_path)
    n = 9
    atoms = Atoms.from_poscar("tests/POSCAR_Si.txt").make_supercell_matrix(
        [n, n, n]
    )
    ase_atoms = atoms.ase_converter()
    x = run_calc(ase_atoms=ase_atoms, model_path=model_path)
    print("tst x", x)
    import sys

    sys.exit()

    ase_atoms.calc = SimpleDftbCalculator(model, kpoints=[2, 2, 2])

    # Get energy and forces
    energy = ase_atoms.get_potential_energy()
    forces = ase_atoms.get_forces()

    print(f"Energy: {energy:.4f} eV")
    print(f"Forces:\n{forces}")

    ############################################################

    from slakonet.interpolation import PolyInterpU
    import sys

    # Example 1: Basic usage
    from ase.build import bulk
    from ase.optimize import BFGS

    model_path = "Si_only.pt"
    # model_path = "Al_only.pt"
    atoms = Atoms.from_poscar(
        "tests/POSCAR_Si.txt"
    )  # .make_supercell_matrix([2, 2, 2])
    ase_atoms = atoms.ase_converter()
    x = run_calc(ase_atoms=ase_atoms, model_path=model_path)
    print(x)
    import sys

    sys.exit()

    model = MultiElementSkfParameterOptimizer.load_ultra_compact(model_path)
    atoms = Atoms.from_poscar("tests/POSCAR_Si.txt").make_supercell_matrix(
        [2, 2, 2]
    )
    # atoms = Atoms.from_poscar("tests/POSCAR").make_supercell_matrix([2, 2, 2])
    print(atoms)
    ase_atoms = atoms.ase_converter()
    geometry = Geometry.from_ase_atoms([ase_atoms])
    geometry.positions.requires_grad_(True)
    # Setup k-points and k-lines
    kpoints2 = torch.tensor([5, 5, 5])  # For DOS
    klines = torch.tensor(
        [
            [0.0, 0.0, 0.0, -0.5, 0.5, 0.0, 10],
            [-0.5, 0.5, 0.0, -0.5, 0.5, -0.07654977, 10],
            [-0.5, 0.5, -0.07654977, -0.28827489, 0.28827489, -0.28827489, 10],
            [-0.28827489, 0.28827489, -0.28827489, 0.0, 0.0, 0.0, 10],
            [0.0, 0.0, 0.0, 0.5, 0.5, -0.5, 10],
            [0.5, 0.5, -0.5, 0.28827489, 0.71172511, -0.71172511, 10],
            [0.28827489, 0.71172511, -0.71172511, 0.0, 0.5, -0.5, 10],
            [0.0, 0.5, -0.5, -0.25, 0.75, -0.25, 10],
            [-0.25, 0.75, -0.25, 0.07654977, 0.92345023, -0.5, 10],
            [0.07654977, 0.92345023, -0.5, 0.5, 0.5, -0.5, 10],
            [0.5, 0.5, -0.5, -0.5, 0.5, 0.0, 10],
            [-0.5, 0.5, 0.0, -0.25, 0.75, -0.25, 10],
        ]
    )
    device = "cuda"
    model = model.to(device)
    model.eval()

    model = model.float()
    s = SimpleDftb(geometry, kpoints=kpoints2, model=model)
    # s = SimpleDftb(geometry,klines=klines,model=model)
    print("ele", s.nelectron)
    res = s.calculate()
    print("res", res)
    calc = SimpleDftbCalculator(
        model=model,
        device="cuda",
    )
    ase_atoms.calc = calc  # set_calculator(calc)
    # print("Energy:", atoms.get_potential_energy())
    print("Forces:", ase_atoms.get_forces())

    import sys

    sys.exit()
    klines = torch.tensor(
        [
            [0.0, 0.0, 0.0, -0.5, 0.5, 0.0, 10],
            [-0.5, 0.5, 0.0, -0.5, 0.5, -0.07654977, 10],
            [-0.5, 0.5, -0.07654977, -0.28827489, 0.28827489, -0.28827489, 10],
            [-0.28827489, 0.28827489, -0.28827489, 0.0, 0.0, 0.0, 10],
            [0.0, 0.0, 0.0, 0.5, 0.5, -0.5, 10],
            [0.5, 0.5, -0.5, 0.28827489, 0.71172511, -0.71172511, 10],
            [0.28827489, 0.71172511, -0.71172511, 0.0, 0.5, -0.5, 10],
            [0.0, 0.5, -0.5, -0.25, 0.75, -0.25, 10],
            [-0.25, 0.75, -0.25, 0.07654977, 0.92345023, -0.5, 10],
            [0.07654977, 0.92345023, -0.5, 0.5, 0.5, -0.5, 10],
            [0.5, 0.5, -0.5, -0.5, 0.5, 0.0, 10],
            [-0.5, 0.5, 0.0, -0.25, 0.75, -0.25, 10],
        ]
    )

    atoms = Atoms.from_poscar(
        "tests/POSCAR_Si.txt"
    ).ase_converter()  # .make_supercell_matrix([2, 2, 2])
    # atoms = Atoms.from_poscar("tests/POSCAR").make_supercell_matrix([2, 2, 2])
    print(atoms)
    # geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
    # geometry.positions.requires_grad_(True)
    # Load model
    calc = get_calculator("Si_only.pt", klines=klines)
    # calc = get_calculator("Si_only.pt", kpoints=[4, 4, 4])

    # Create structure
    atoms = bulk("Si", "diamond", a=5.00)
    atoms.calc = calc

    # Get energy and forces
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print(f"Energy: {energy:.4f} eV")
    print(f"Forces:\n{forces}")

    # Get DFTB-specific properties
    print(f"Fermi energy: {calc.get_fermi_energy():.4f} eV")
    print(f"Band gap: {calc.get_bandgap():.4f} eV")

    # Example 2: Structure optimization
    atoms = bulk("Si", "diamond", a=5.0)  # Wrong lattice constant
    atoms.calc = calc

    opt = BFGS(atoms)
    opt.run(fmax=0.01)  # Optimize until forces < 0.01 eV/Å

    print(f"Optimized lattice: {atoms.cell.cellpar()[0]:.3f} Å")
    sys.exit()

    # Example 3: From POSCAR
    from ase.io import read

    atoms = read("POSCAR")
    atoms.calc = get_calculator(
        "slakonet_v1.pt",
        kpoints=[6, 6, 6],
        cutoff=20.0,
        repulsive=True,
    )

    energy = atoms.get_potential_energy()
    bandgap = atoms.calc.get_bandgap()

    # Example 4: Band structure calculation
    from jarvis.core.kpoints import Kpoints3D
    from jarvis.core.atoms import Atoms as JarvisAtoms

    jarvis_atoms = ase_atoms(atoms)
    kpoints = Kpoints3D().kpath(jarvis_atoms, line_density=20)
    klines = kpts_to_klines(kpoints.kpts, default_points=2)

    calc_bands = SimpleDftbCalculator(
        model=model, klines=klines, device="cuda"
    )

    atoms.calc = calc_bands
    atoms.get_potential_energy()

    eigenvalues = calc_bands.get_eigenvalues()
    # Plot bands...

    # Example 5: Molecular dynamics
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    from ase import units

    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)

    def print_energy(a=atoms):
        epot = a.get_potential_energy()
        ekin = a.get_kinetic_energy()
        print(f"E_pot = {epot:.4f} eV, E_kin = {ekin:.4f} eV")

    dyn.attach(print_energy, interval=10)
    dyn.run(100)

    # Example 6: NEB (Nudged Elastic Band)
    from ase.neb import NEB
    from ase.optimize import FIRE

    # Initial and final states
    initial = atoms.copy()
    final = atoms.copy()
    final.positions[0] += [0.5, 0, 0]  # Move one atom

    # Create NEB
    images = [initial]
    for i in range(5):
        images.append(initial.copy())
    images.append(final)

    neb = NEB(images)
    neb.interpolate()

    # Set calculator for all images
    for img in images[1:-1]:
        img.calc = calc

    # Optimize
    optimizer = FIRE(neb)
    optimizer.run(fmax=0.05)
    sys.exit()

    # Setup parameters
    shell_dict = generate_shell_dict_upto_Z65()
    path_to_skf = "tests/Si-Si.skf"
    path_to_skf = "tests/Al-Al.skf"
    from slakonet.skf import Skf

    interpolator = PolyInterpU
    sk = Skf.from_skf(path_to_skf)
    # print(sk)
    dd = sk.to_dict()
    # print(sk.to_dict())
    skf = Skf.from_dict(dd)

    integral_type = "H"
    hs_dict, onsite_hs_dict = {}, {}
    hs_dict = _get_hs_dict(
        hs_dict, interpolator, skf, integral_type  # , **kwargs
    )
    elements = path_to_skf.split("/")[1].split(".skf")[0].split("-")
    if elements[0] == elements[1]:
        print("same element")
        onsite_hs_dict = _get_onsite_dict(
            onsite_hs_dict, skf, shell_dict, integral_type
        )

    h_feed = SkfFeed(hs_dict, onsite_hs_dict, shell_dict)

    integral_type = "S"
    hs_dict, onsite_hs_dict = {}, {}
    hs_dict = _get_hs_dict(
        hs_dict, interpolator, skf, integral_type  # , **kwargs
    )
    elements = path_to_skf.split("/")[1].split(".skf")[0].split("-")
    if elements[0] == elements[1]:
        print("same element")
        onsite_hs_dict = _get_onsite_dict(
            onsite_hs_dict, skf, shell_dict, integral_type
        )

    s_feed = SkfFeed(hs_dict, onsite_hs_dict, shell_dict)
    # nelectron = torch.tensor([8])  # skparams.qzero.sum(-1)
    if "atomic_data" in dd:  # and skf_dict["atomic_data"]:
        occupations = dd["atomic_data"]["occupations"]
        nelectron = torch.tensor(
            [2 * sum(occupations)]
        )  # Factor of 2 for spin
    # nelectron = torch.tensor([8])  # skparams.qzero.sum(-1)
    print("nelectron", nelectron)
    # Create calculator for band structure
    calc_bands = SimpleDftb(
        geometry,
        shell_dict=shell_dict,
        klines=klines,
        h_feed=h_feed,
        s_feed=s_feed,
        nelectron=nelectron,
    )
    calc_bands.get_repulsive_energy()
    # Run calculation
    print("Computing band structure...")
    eigenvalues_bands = calc_bands()
    # print("forces",calc_bands.get_forces())
    # print("forces", calc_bands._compute_forces_finite_diff())
    # x, y = calc_bands.calculate_phonon_modes()
    print("\nPlotting band structure...")
    import sys

    sys.exit()
    fig_bands, ax_bands = calc_bands.plot_band_structure(
        fermi_shift=True, save_path="bands_enhanced.png"
    )
    plt.show()
