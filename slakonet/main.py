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
from slakonet.slaterkoster import fermi, hs_matrix
from jarvis.core.atoms import Atoms
from slakonet.utils import eighb, pack
import matplotlib.pyplot as plt

from slakonet.fermi import fermi_search, fermi_dirac, fermi_smearing
import numpy as np
from slakonet.utils import eighb
from slakonet.fermi import fermi_smearing

try:
    from phonopy import Phonopy
    from phonopy.file_IO import write_FORCE_CONSTANTS, write_disp_yaml
except Exception:
    pass
torch.set_default_dtype(torch.float32)
# torch.set_default_dtype(torch.float64)
# torch.set_default_dtype(torch.float32)
H2E = 27.211


def generate_shell_dict_upto_Z65():
    """Generate shell_dict for atomic numbers 1-65."""
    shell_dict = {}
    for Z in range(1, 100):
        if Z <= 2:  # H, He
            shell_dict[Z] = [0]
        elif Z <= 10:  # Li to Ne
            shell_dict[Z] = [0, 1]
        elif Z <= 20:  # Na to Ca
            shell_dict[Z] = [0, 1]
        elif Z <= 30:  # Sc to Zn
            shell_dict[Z] = [0, 1, 2]
        elif Z <= 36:  # Ga to Kr
            shell_dict[Z] = [0, 1]
        elif Z <= 48:  # transition metals
            shell_dict[Z] = [0, 1, 2]
        elif Z <= 54:  # In to Xe
            shell_dict[Z] = [0, 1]
        elif Z <= 57:  # Cs, Ba, La
            shell_dict[Z] = [0, 1, 2]
        else:  # lanthanides
            shell_dict[Z] = [0, 1, 2, 3]
    return shell_dict


class SimpleDftb:
    """Enhanced DFTB calculator for periodic systems with analysis tools."""

    def __init__(
        self,
        geometry,
        shell_dict,
        h_feed=None,
        s_feed=None,
        nelectron=None,
        kpoints=None,
        klines=None,
        repulsive=False,
        device=None,
        with_eigenvectors=False,
        pairwise_cutoff_length=7,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.geometry = geometry
        self.shell_dict = shell_dict
        self.h_feed = h_feed
        self.s_feed = s_feed
        self.dtype = torch.complex64
        # self.dtype = torch.complex128
        self.repulsive = repulsive
        self.device = device
        self.with_eigenvectors = with_eigenvectors
        # self.device="cuda"
        # self.device = torch.device("cpu")
        # print("self.device", self.device)
        self.nelectron = nelectron.to(self.device)
        # Initialize basis
        self.basis = Basis(self.geometry.atomic_numbers, self.shell_dict)
        self.atom_orbitals = self.basis.orbs_per_atom
        self.pairwise_cutoff_length = pairwise_cutoff_length

        print(
            f"SimpleDftb class using pairwise cutoff length of {self.pairwise_cutoff_length} Angstroms"
        )

        # Initialize periodic structure with k-points
        if kpoints is not None and klines is not None:
            self.periodic = Periodic(
                self.geometry,
                self.geometry.cell,
                cutoff=self.pairwise_cutoff_length,
                kpoints=kpoints,
                klines=klines,
            )
        elif kpoints is not None:
            self.periodic = Periodic(
                self.geometry, self.geometry.cell, cutoff=self.pairwise_cutoff_length, kpoints=kpoints
            )
        elif klines is not None:
            self.periodic = Periodic(
                self.geometry, self.geometry.cell, cutoff=self.pairwise_cutoff_length, klines=klines
            )
        else:
            self.periodic = Periodic(
                self.geometry, self.geometry.cell, cutoff=self.pairwise_cutoff_length
            )

        self.kpoints = self.periodic.kpoints
        self.k_weights = self.periodic.k_weights.to(self.device)
        self.max_nk = torch.max(self.periodic.n_kpoints)

        # Cache for computed properties
        self._fermi_energy = None
        self._forces = None
        self._band_gap = None
        self._occupations = None

    def compute_hs_matrices(self):
        """Compute Hamiltonian and overlap matrices."""
        # print("Computing H and S matrices...")
        self.ham = hs_matrix(self.periodic, self.basis, self.h_feed)
        self.overlap = hs_matrix(self.periodic, self.basis, self.s_feed)
        self.ham = self.ham.to(self.device)
        self.overlap = self.overlap.to(self.device)

    def solve_kpoint(self, ik):
        """Solve eigenvalue problem for k-point ik."""
        # Get matrices for this k-point
        h_k = self.ham[..., ik]
        s_k = self.overlap[..., ik]

        # Solve generalized eigenvalue problem
        # eigenvals, eigenvecs = eighb(h_k, s_k, scheme="chol")
        eigenvals, eigenvecs = eighb(h_k, s_k, scheme="chol")
        # eigenvals, eigenvecs = eighb(h_k, s_k,broadening_method=None,scheme="lowd")
        # try:
        #  eigenvals, eigenvecs = eighb(h_k, s_k,scheme="chol")
        # except:
        #  eigenvals, eigenvecs = eighb(h_k, s_k,broadening_method=None)

        # print("h_k device",h_k.device)
        # print("s_k device",s_k.device)
        # print("eigenvals",eigenvals.device)
        # print("eigenvecs",eigenvecs.device)

        # Calculate occupations
        occ, _ = fermi(eigenvals, self.nelectron.to(self.device))

        # Calculate density matrix
        c_occ = torch.sqrt(occ).unsqueeze(1).expand_as(eigenvecs) * eigenvecs
        density = torch.conj(c_occ) @ c_occ.transpose(1, 2)

        return eigenvals, eigenvecs, occ, density

    def __call__(self):
        """Main calculation routine."""
        self.compute_hs_matrices()

        eigenvalues = []
        densities = []
        occupations = []
        eigenvectors = []
        successful_k_indices = []  # Track which k-points succeeded
        # Loop over k-points
        for ik in range(self.max_nk):
            try:
                eigenvals, eigenvecs, occ, density = self.solve_kpoint(ik)
                eigenvalues.append(eigenvals)
                densities.append(density)
                occupations.append(occ)
                if self.with_eigenvectors:
                    eigenvectors.append(eigenvecs)
                successful_k_indices.append(ik)
            except Exception as exp:
                print("ik failed for", ik, exp)
                pass
        # Store results (keep on GPU)

        self.eigenvalue = pack(eigenvalues).permute(1, 0, 2)
        self.density = pack(densities).permute(1, 2, 3, 0)
        self._occupations = pack(occupations).permute(1, 0, 2)
        if self.with_eigenvectors:
            self.eigenvectors = pack(eigenvectors).permute(1, 2, 3, 0)

        # CRITICAL FIX: Filter k_weights to only include successful k-points
        if len(successful_k_indices) < self.max_nk:
            print(
                f"Warning: Only {len(successful_k_indices)}/{self.max_nk} k-points succeeded"
            )
            # Filter and renormalize k_weights
            successful_k_indices_tensor = torch.tensor(
                successful_k_indices, device=self.device
            )
            self.k_weights = self.k_weights[:, successful_k_indices_tensor]
            # Renormalize to sum to 1
            self.k_weights = self.k_weights / self.k_weights.sum(
                dim=1, keepdim=True
            )
            print(f"Filtered k_weights shape: {self.k_weights.shape}")

        # Clear cache
        self._fermi_energy = None
        self._band_gap = None

        return self.eigenvalue

    def _compute_forces_finite_diff(
        self, delta=1e-2, kpoints=torch.tensor([5, 5, 5])
    ):
        """
        Fallback force calculation using finite differences.
        """
        # This is not completely tested
        print("Computing forces using finite differences...")

        original_positions = self.geometry.positions.clone()
        forces = torch.zeros_like(original_positions)

        # Temporarily disable gradients for finite difference calculation
        self.geometry.positions = self.geometry.positions.detach()

        kpoints2 = kpoints  # torch.tensor([5, 5, 5])  # For DOS

        def get_energy_at_positions(positions):
            """Get energy for given positions."""

            # cell = torch.tensor(
            #    [
            #        [6.3573, -0.0000, 3.6704],
            #        [2.1191, 5.9937, 3.6704],
            #        [-0.0000, -0.0000, 7.3408],
            #    ]
            # )
            # geometry = Geometry(torch.tensor([[14, 14]]), positions, cell)
            geometry = Geometry(
                self.geometry.atomic_numbers, positions, self.geometry.cell
            )
            # print("positions",positions)
            calc = SimpleDftb(
                geometry,
                shell_dict=self.shell_dict,
                kpoints=kpoints2,
                # klines=klines,
                h_feed=self.h_feed,
                s_feed=self.s_feed,
                nelectron=self.nelectron,
                pairwise_cutoff_length=self.pairwise_cutoff_length
            )

            # Compute properties
            eigenvalues = calc()
            # Clear cache
            # self._fermi_energy = None
            # self._band_gap = None
            # self._occupations = None

            # Recalculate
            # self()
            return torch.sum(
                eigenvalues
            )  # self._calculate_electronic_energy()

        # Loop over atoms and coordinates
        n_atoms = original_positions.shape[1]
        n_coords = original_positions.shape[2]

        for i in range(n_atoms):
            for j in range(n_coords):
                # Forward step
                pos_forward = original_positions.clone()
                pos_forward[0, i, j] += delta
                energy_forward = get_energy_at_positions(pos_forward)

                # Backward step
                pos_backward = original_positions.clone()
                pos_backward[0, i, j] -= delta
                energy_backward = get_energy_at_positions(pos_backward)
                print(
                    "energy_forward,energy_backward",
                    energy_forward,
                    energy_backward,
                )
                # Central difference
                forces[0, i, j] = -(energy_forward - energy_backward) / (
                    2 * delta
                )

        # Restore original positions with gradients
        self.geometry.positions = original_positions.requires_grad_(True)

        return forces

    def calculate_phonon_modes(self, line_density=5, write_fc=True):
        """Calculate phonon modes and frequencies using Phonopy."""
        print("Setting up phonon calculation...")

        elements = (self.geometry.atomic_numbers.detach().numpy().tolist())[0]
        lattice_mat = self.geometry.cell.detach().numpy()[0]
        coords = self.geometry.positions.detach().numpy()[0]
        # print('elements',elements)
        # print('lattice_mat',lattice_mat)
        # print('coords',coords)

        atoms = JAtoms(
            lattice_mat=lattice_mat,
            elements=atomic_numbers_to_symbols(elements),
            coords=coords,
            cartesian=True,
        )
        kpoints = Kpoints().kpath(atoms, line_density=line_density)
        dim = [1, 1, 1]
        distance = 0.05
        # Convert to phonopy format
        bulk = atoms.phonopy_converter()
        self.phonon = Phonopy(
            bulk, [[dim[0], 0, 0], [0, dim[1], 0], [0, 0, dim[2]]]
        )

        # Generate displacements
        self.phonon.generate_displacements(distance=distance)

        print(
            f"Number of displaced supercells: {len(self.phonon.supercells_with_displacements)}"
        )

        # Get supercells with displacements
        supercells = self.phonon.supercells_with_displacements

        # Calculate forces for each displaced supercell
        set_of_forces = []

        for i, scell in enumerate(supercells):
            print(
                f"Calculating forces for displacement {i+1}/{len(supercells)}"
            )

            # Convert to ASE atoms
            ase_atoms = AseAtoms(
                symbols=scell.symbols,
                scaled_positions=scell.scaled_positions,
                cell=scell.cell,
                pbc=True,
            )

            geometry = Geometry.from_ase_atoms([ase_atoms])
            calc_bands = SimpleDftb(
                geometry,
                shell_dict=self.shell_dict,
                kpoints=torch.tensor([5, 5, 5]),
                # klines=klines,
                h_feed=self.h_feed,
                s_feed=self.s_feed,
                nelectron=self.nelectron,
                pairwise_cutoff_length=self.pairwise_cutoff_length,
            )

            # Run calculation
            print("Computing band structure...")
            eigenvalues_bands = calc_bands()
            # print("forces",calc_bands.get_forces())
            forces = (
                calc_bands._compute_forces_finite_diff().detach().numpy()[0]
            )
            # ase_atoms.calc = self.calculator

            # Calculate forces
            # forces = np.array(ase_atoms.get_forces())

            # Remove drift force
            drift_force = forces.sum(axis=0)
            for force in forces:
                force -= drift_force / forces.shape[0]

            set_of_forces.append(forces)

        # Produce force constants
        print("Producing force constants...")
        self.phonon.produce_force_constants(forces=set_of_forces)

        if write_fc:
            write_FORCE_CONSTANTS(
                self.phonon.force_constants, filename="FORCE_CONSTANTS"
            )

        # Write displacement file
        write_disp_yaml(
            self.phonon.displacements,
            self.phonon.supercell,
            filename="phonopy_disp.yaml",
        )

        # Calculate phonon DOS
        self.phonon.run_mesh(
            [40, 40, 40], is_gamma_center=True, is_mesh_symmetry=False
        )
        self.phonon.run_total_dos()
        tdos = self.phonon._total_dos
        freqs, ds = tdos.get_dos()
        freqs = np.array(freqs)
        freq_conversion_factor = 33.3566830
        freqs = freqs * freq_conversion_factor
        plt.plot(freqs, ds)
        plt.savefig("phdos.png")
        plt.close()
        # Get frequencies and modes at high-symmetry k-points
        # self._extract_phonon_data(kpoints)
        # print("self.phonon_frequencies",self.phonon_frequencies)
        return freqs, ds
        # return self.phonon_frequencies, self.phonon_modes

    def get_forces(self):
        """
        Calculate forces using automatic differentiation with improved numerical stability.
        This version properly uses the existing eighb function from slakonet.utils.
        """
        if self._forces is None:
            print("Computing forces with improved stability...")

            # Clear all cached values to ensure fresh calculation
            self._fermi_energy = None
            self._band_gap = None
            self._occupations = None

            # Ensure positions require gradients
            if not self.geometry.positions.requires_grad:
                print("Enabling gradients for positions...")
                self.geometry.positions = (
                    self.geometry.positions.detach().requires_grad_(True)
                )

            # Import the eighb function that your code uses

            # Stable electronic energy calculation
            def compute_stable_electronic_energy():
                """Compute electronic energy with better numerical stability."""

                # Recompute matrices
                self.compute_hs_matrices()

                # Check for NaN in matrices
                if (
                    torch.isnan(self.ham).any()
                    or torch.isnan(self.overlap).any()
                ):
                    print("Warning: NaN detected in H/S matrices!")
                    return torch.tensor(
                        0.0, requires_grad=True, device=self.device
                    )

                eigenvalues = []
                occupations = []
                valid_kpoints = 0

                for ik in range(self.max_nk):
                    try:
                        # Get matrices for this k-point
                        h_k = self.ham[..., ik]
                        s_k = self.overlap[..., ik]

                        # Add small regularization to overlap matrix for stability
                        reg_factor = 1e-8
                        eye_matrix = torch.eye(
                            s_k.shape[-1], device=s_k.device, dtype=s_k.dtype
                        )
                        s_k = s_k + reg_factor * eye_matrix

                        # Solve generalized eigenvalue problem using your existing function
                        eigenvals, eigenvecs = eighb(h_k, s_k)

                        # Check for NaN or inf in eigenvalues
                        if (
                            torch.isnan(eigenvals).any()
                            or torch.isinf(eigenvals).any()
                        ):
                            print(
                                f"Invalid eigenvalues at k-point {ik}, skipping..."
                            )
                            continue

                        # Use your existing fermi function for occupation calculation
                        occ, _ = fermi(eigenvals, self.nelectron)

                        # Check for NaN in occupations
                        if torch.isnan(occ).any():
                            print(
                                f"Invalid occupations at k-point {ik}, skipping..."
                            )
                            continue

                        eigenvalues.append(eigenvals)
                        occupations.append(occ)
                        valid_kpoints += 1

                    except Exception as e:
                        print(f"Error at k-point {ik}: {e}")
                        continue

                if valid_kpoints == 0:
                    print("No valid k-points computed!")
                    return torch.tensor(
                        0.0, requires_grad=True, device=self.device
                    )

                # Stack results and store them (matching your original structure)
                eigenvalues = torch.stack(eigenvalues).permute(1, 0, 2)
                occupations = torch.stack(occupations).permute(1, 0, 2)

                # Store for later use
                self.eigenvalue = eigenvalues
                self._occupations = occupations

                # Calculate electronic energy using your existing method
                # But ensure we only use valid k-points
                if valid_kpoints < self.max_nk:
                    # Adjust k_weights for valid k-points only
                    k_weights_valid = self.k_weights[:valid_kpoints]
                else:
                    k_weights_valid = self.k_weights

                # Use Fermi energy and smearing like in your original code
                fermi_energy = self.get_fermi_energy()
                kT_hartree = 0.025 / 27.211  # Convert eV to Hartree

                # Calculate electronic energy
                electronic_energy = torch.sum(
                    occupations * eigenvalues * k_weights_valid.unsqueeze(-1)
                )

                return electronic_energy.real

            # Compute energy
            try:
                electronic_energy = compute_stable_electronic_energy()
                print(f"Electronic energy: {electronic_energy.item():.8f} Ha")
                print(
                    f"Electronic energy requires grad: {electronic_energy.requires_grad}"
                )

                # Verify energy is finite and has gradients
                if not electronic_energy.requires_grad:
                    print(
                        "ERROR: Electronic energy does not require gradients!"
                    )
                    self._forces = torch.zeros_like(self.geometry.positions)
                    return self._forces

                if torch.isnan(electronic_energy) or torch.isinf(
                    electronic_energy
                ):
                    print(
                        f"ERROR: Electronic energy is {electronic_energy.item()}"
                    )
                    self._forces = torch.zeros_like(self.geometry.positions)
                    return self._forces

                # Calculate gradients
                try:
                    grad_outputs = torch.autograd.grad(
                        electronic_energy,
                        self.geometry.positions,
                        create_graph=False,  # Don't create computational graph for higher-order derivatives
                        retain_graph=False,  # Don't retain graph after computation
                        allow_unused=False,
                    )

                    forces_raw = grad_outputs[0]

                    # Check for NaN in forces
                    if torch.isnan(forces_raw).any():
                        print(
                            "NaN detected in raw forces, trying finite differences..."
                        )
                        self._forces = self._compute_forces_finite_diff()
                    else:
                        # Forces are negative gradient
                        self._forces = -forces_raw
                        max_force = torch.max(torch.abs(self._forces)).item()
                        print(
                            f"Forces computed successfully! Max component: {max_force:.6f} Ha/Bohr"
                        )

                        # Sanity check: forces shouldn't be too large
                        if max_force > 10.0:  # Arbitrary threshold
                            print(
                                "Warning: Very large forces detected, might be numerical instability"
                            )

                except RuntimeError as e:
                    print(f"Gradient calculation failed: {e}")
                    self._forces = self._compute_forces_finite_diff()

            except Exception as e:
                print(f"Energy calculation failed: {e}")
                self._forces = torch.zeros_like(self.geometry.positions)

        return self._forces

    def get_forcess(self):
        # original_cell = self.geometry.cell.clone()

        # original_positions = self.geometry.positions.clone()
        # original_positions.requires_grad_(True)
        if self._forces is None:
            self._forces, _ = torch.autograd.grad(
                self._calculate_electronic_energy(),
                self.geometry.positions,
                create_graph=True,
            )
        return self._forces

    def get_fermi_energy(self, kT=0.025):
        fermi_energy = fermi_search(
            # fermi_energy = fermi_search(
            eigenvalues=self.eigenvalue,
            n_electrons=self.nelectron,
            k_weights=self.k_weights,
            kT=kT,
            # k_weights=self.k_weights,
        )
        # print("fermi_energy main", fermi_energy, fermi_energy.device)
        return fermi_energy

    def get_eigenvalues(self, fermi_shift=True, unit="eV"):
        """
        Get eigenvalues with optional Fermi shift.

        Parameters:
        -----------
        fermi_shift : bool
            If True, shift eigenvalues so Fermi energy is at zero
        unit : str
            'eV' or 'Ha' for output units

        Returns:
        --------
        torch.Tensor : Eigenvalues with shape (nbatch, nkpoints, nbands)
        """
        eigenvals = self.eigenvalue.clone()

        if fermi_shift:
            fermi_energy = self.get_fermi_energy()
            eigenvals = eigenvals - fermi_energy

        if unit == "eV":
            eigenvals = eigenvals * H2E

        return eigenvals

    def calculate_band_gapX(self, kT=0.025):
        """Calculate band gap from eigenvalues and occupations.

        Parameters
        ----------
        kT : float
            Electronic temperature in eV for Fermi energy calculation

        Returns
        -------
        dict
            Dictionary containing:
            - 'gap' : Band gap in eV
            - 'vbm' : Valence band maximum in eV
            - 'cbm' : Conduction band minimum in eV
            - 'direct' : Boolean indicating if gap is direct
            - 'vbm_kpoint' : k-point index of VBM
            - 'cbm_kpoint' : k-point index of CBM
        """
        if self._band_gap is None:
            fermi_energy = self.get_fermi_energy(kT)
            eigenvals_eV = self.eigenvalue * H2E
            fermi_eV = fermi_energy * H2E

            # Masks
            occupied_mask = eigenvals_eV < fermi_eV
            unoccupied_mask = eigenvals_eV >= fermi_eV
            bands_at_fermi = torch.abs(eigenvals_eV - fermi_eV) < 1e-3

            # Debug (optional)
            # print("min(E)", eigenvals_eV.min().item(),
            #       "max(E)", eigenvals_eV.max().item(),
            #       "Ef", fermi_eV.item())
            # print("n_occ", occupied_mask.sum().item(),
            #       "n_unocc", unoccupied_mask.sum().item())

            # Case 1: metallic bands (already in your code)
            if torch.any(bands_at_fermi):
                print("System is metallic - bands cross Fermi level")
                self._band_gap = {
                    "gap": torch.tensor(0.0, device=eigenvals_eV.device),
                    "vbm": fermi_eV,
                    "cbm": fermi_eV,
                    "direct": False,
                    "vbm_kpoint": 0,
                    "cbm_kpoint": 0,
                }

            # Case 2: no occupied OR no unoccupied states (ill-defined gap)
            elif (not torch.any(occupied_mask)) or (
                not torch.any(unoccupied_mask)
            ):
                print(
                    "Warning: no occupied or no unoccupied states relative to Fermi; "
                    "treating system as metallic / gapless."
                )
                self._band_gap = {
                    "gap": torch.tensor(0.0, device=eigenvals_eV.device),
                    "vbm": fermi_eV,
                    "cbm": fermi_eV,
                    "direct": False,
                    "vbm_kpoint": 0,
                    "cbm_kpoint": 0,
                }

            else:
                # Case 3: proper insulator/semiconductor
                vbm = torch.max(eigenvals_eV[occupied_mask])
                cbm = torch.min(eigenvals_eV[unoccupied_mask])

                # Find k-point indices
                vbm_indices = torch.where(eigenvals_eV == vbm)
                cbm_indices = torch.where(eigenvals_eV == cbm)

                vbm_kpoint = (
                    vbm_indices[1][0] if len(vbm_indices[1]) > 0 else 0
                )
                cbm_kpoint = (
                    cbm_indices[1][0] if len(cbm_indices[1]) > 0 else 0
                )

                # Check if direct gap
                direct = vbm_kpoint == cbm_kpoint

                self._band_gap = {
                    "gap": cbm - vbm,
                    "vbm": vbm,
                    "cbm": cbm,
                    "direct": bool(direct),
                    "vbm_kpoint": int(vbm_kpoint.item()),
                    "cbm_kpoint": int(cbm_kpoint.item()),
                }

        return self._band_gap

    def calculate_band_gap(self, kT=0.025):
        """
        Calculate band gap from eigenvalues and occupations.

        Parameters:
        -----------
        kT : float
            Electronic temperature in eV for Fermi energy calculation

        Returns:
        --------
        dict : Dictionary containing:
            - 'gap' : Band gap in eV
            - 'vbm' : Valence band maximum in eV
            - 'cbm' : Conduction band minimum in eV
            - 'direct' : Boolean indicating if gap is direct
            - 'vbm_kpoint' : k-point index of VBM
            - 'cbm_kpoint' : k-point index of CBM
        """
        if self._band_gap is None:
            fermi_energy = self.get_fermi_energy(kT)
            eigenvals_eV = self.eigenvalue * H2E
            fermi_eV = fermi_energy * H2E

            # Find occupied and unoccupied states
            # Occupied: eigenvalue < fermi_energy
            # Unoccupied: eigenvalue > fermi_energy
            occupied_mask = eigenvals_eV < fermi_eV
            unoccupied_mask = eigenvals_eV >= fermi_eV
            bands_at_fermi = torch.abs(eigenvals_eV - fermi_eV) < 1e-3
            if torch.any(bands_at_fermi):

                print("System is metallic - bands cross Fermi level")
                # Metal or problematic case
                self._band_gap = {
                    "gap": torch.tensor(0.0),
                    "vbm": fermi_eV,
                    "cbm": fermi_eV,
                    "direct": False,
                    "vbm_kpoint": 0,
                    "cbm_kpoint": 0,
                }
            else:
                # Find VBM and CBM
                # print("occupied_mask",occupied_mask)
                # print("eigenvals_eV[occupied_mask]",eigenvals_eV[occupied_mask])
                vbm = torch.max(eigenvals_eV[occupied_mask])
                cbm = torch.min(eigenvals_eV[unoccupied_mask])

                # Find k-point indices
                vbm_indices = torch.where(eigenvals_eV == vbm)
                cbm_indices = torch.where(eigenvals_eV == cbm)

                vbm_kpoint = (
                    vbm_indices[1][0] if len(vbm_indices[1]) > 0 else 0
                )
                cbm_kpoint = (
                    cbm_indices[1][0] if len(cbm_indices[1]) > 0 else 0
                )

                # Check if direct gap
                direct = (vbm_kpoint == cbm_kpoint).item()

                self._band_gap = {
                    "gap": cbm - vbm,
                    "vbm": vbm,
                    "cbm": cbm,
                    "direct": direct,
                    "vbm_kpoint": vbm_kpoint.item(),
                    "cbm_kpoint": cbm_kpoint.item(),
                }

        return self._band_gap

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
            'eV' or 'Ha' for energy units

        Returns:
        --------
        tuple : (energy_grid, dos) both as torch.Tensors
        """
        # Get eigenvalues in requested units
        eigenvals = self.get_eigenvalues(fermi_shift=fermi_shift, unit=unit)

        # Debug: print shapes
        # print(f"Eigenvals shape: {eigenvals.shape}")
        # print(f"K-weights shape: {self.k_weights.shape}")

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

        # Flatten eigenvalues for easier processing
        eigenvals_flat = eigenvals.flatten()  # All eigenvalues in one tensor
        # print(f"Flattened eigenvals shape: {eigenvals_flat.shape}")

        # Calculate DOS using vectorized approach
        nbatch, nkpoints, nbands = eigenvals.shape

        for ik in range(nkpoints):
            # Get k-point weight - handle 2D k_weights tensor properly
            if len(self.k_weights.shape) == 2:
                weight = self.k_weights[0, ik]  # Extract scalar from 2D tensor
            elif ik < len(self.k_weights):
                weight = self.k_weights[ik]
            else:
                weight = torch.tensor(1.0 / nkpoints, device=self.device)

            # print(
            #    f"K-point {ik} weight: {weight.item():.6f}, weight shape: {weight.shape}"
            # )

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

    def calculate_projected_dos(
        self,
        atom_indices=None,
        orbital_indices=None,
        energy_range=(-10, 5),
        num_points=1000,
        sigma=0.1,
        fermi_shift=True,
        unit="eV",
    ):
        """
        Calculate projected density of states (PDOS).

        Parameters:
        -----------
        atom_indices : list, optional
            List of atom indices to project onto (0-indexed)
        orbital_indices : list, optional
            List of orbital indices to project onto
        energy_range : tuple
            Energy range for PDOS calculation
        num_points : int
            Number of energy grid points
        sigma : float
            Gaussian broadening parameter
        fermi_shift : bool
            If True, shift energies so Fermi energy is at zero
        unit : str
            'eV' or 'Ha' for energy units

        Returns:
        --------
        tuple : (energy_grid, pdos) both as torch.Tensors
        """
        # This would require eigenvectors and overlap matrices
        # Placeholder implementation - would need access to eigenvectors
        # from solve_kpoint method
        print("Warning: PDOS calculation requires storing eigenvectors")
        return self.calculate_dos(
            energy_range, num_points, sigma, fermi_shift, unit
        )

    def plot_band_structure(
        self,
        fermi_shift=True,
        unit="eV",
        figsize=(10, 6),
        save_path=None,
        show_fermi=True,
    ):
        """
        Plot band structure.

        Parameters:
        -----------
        fermi_shift : bool
            If True, shift bands so Fermi energy is at zero
        unit : str
            'eV' or 'Ha' for energy units
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the plot
        show_fermi : bool
            Whether to show Fermi energy line

        Returns:
        --------
        tuple : (fig, ax) matplotlib objects
        """
        eigenvals = self.get_eigenvalues(fermi_shift=fermi_shift, unit=unit)

        # Convert to numpy for plotting
        bands = (
            eigenvals[0].detach().cpu().numpy()
        )  # Shape: (nkpoints, nbands)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot each band
        nkpoints, nbands = bands.shape
        kpoint_indices = range(nkpoints)

        for ib in range(nbands):
            ax.plot(kpoint_indices, bands[:, ib], "b-", linewidth=1)

        # Show Fermi energy
        if show_fermi:
            if fermi_shift:
                ax.axhline(
                    y=0,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="E_F = 0.000 eV",
                )
            else:
                fermi_energy = self.get_fermi_energy()
                fermi_val = (
                    fermi_energy * H2E if unit == "eV" else fermi_energy
                )
                ax.axhline(
                    y=fermi_val.item(),
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"E_F = {fermi_val.item():.3f} {unit}",
                )
            ax.legend()

        # Formatting
        xlabel = "k-point"
        ylabel = (
            f"Energy - E_F ({unit})" if fermi_shift else f"Energy ({unit})"
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Electronic Band Structure")
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Band structure saved to {save_path}")

        return fig, ax

    def plot_dos(
        self,
        energy_range=(-10, 5),
        num_points=1000,
        sigma=0.1,
        fermi_shift=True,
        unit="eV",
        figsize=(8, 6),
        show_fermi=True,
        save_path=None,
    ):
        """
        Plot density of states.

        Parameters:
        -----------
        energy_range : tuple
            Energy range for DOS plot
        num_points : int
            Number of energy points
        sigma : float
            Gaussian broadening
        fermi_shift : bool
            If True, shift energies so Fermi energy is at zero
        unit : str
            'eV' or 'Ha' for energy units
        figsize : tuple
            Figure size (width, height)
        show_fermi : bool
            Whether to show Fermi energy line
        save_path : str, optional
            Path to save the plot

        Returns:
        --------
        tuple : (fig, ax) matplotlib objects
        """
        # Calculate DOS (returns tensors on GPU)
        energy_grid, dos = self.calculate_dos(
            energy_range, num_points, sigma, fermi_shift, unit
        )

        # Convert to numpy for plotting
        energy_np = energy_grid.detach().cpu().numpy()
        dos_np = dos.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(energy_np, dos_np, "b-", linewidth=2)
        ax.fill_between(energy_np, dos_np, alpha=0.3)

        # Show Fermi energy
        if show_fermi:
            if fermi_shift:
                ax.axvline(
                    0,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="E_F = 0.000 eV",
                )
            else:
                fermi_energy = self.get_fermi_energy()
                fermi_val = (
                    fermi_energy * H2E if unit == "eV" else fermi_energy
                )
                ax.axvline(
                    fermi_val.item(),
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"E_F = {fermi_val.item():.3f} {unit}",
                )
            ax.legend()

        # Formatting
        xlabel = (
            f"Energy - E_F ({unit})" if fermi_shift else f"Energy ({unit})"
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Density of States (states/{unit})")
        title = "Electronic Density of States"
        if fermi_shift:
            title += " (Fermi-shifted)"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(energy_range)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"DOS plot saved to {save_path}")

        return fig, ax

    def calculate_bulk_modulus(
        self, strain_range=0.02, num_strains=5, method="birch_murnaghan"
    ):
        """
        Calculate bulk modulus using finite strain method.

        Parameters:
        -----------
        strain_range : float
            Maximum strain to apply (±strain_range)
        num_strains : int
            Number of strain points to calculate
        method : str
            'birch_murnaghan' or 'polynomial' fitting method

        Returns:
        --------
        dict : Dictionary containing bulk modulus results
        """
        print("Calculating bulk modulus...")

        # Store original geometry
        original_cell = self.geometry.cell.clone()
        original_positions = self.geometry.positions.clone()

        # Create strain points
        strains = torch.linspace(-strain_range, strain_range, num_strains)
        volumes = []
        energies = []

        for i, strain in enumerate(strains):
            print(f"Strain point {i+1}/{num_strains}: strain = {strain:.4f}")

            # Apply isotropic strain to cell
            strain_factor = 1.0 + strain
            strained_cell = original_cell * strain_factor

            # Scale positions proportionally with cell
            strained_positions = original_positions * strain_factor

            # Update geometry with strained cell and positions
            self.geometry.cell = strained_cell
            self.geometry.positions = strained_positions

            # IMPORTANT: Enable gradients for positions if force calculation is needed
            if not self.geometry.positions.requires_grad:
                self.geometry.positions.requires_grad_(True)

            # Recreate periodic structure for new geometry
            try:
                # Store original k-point configuration
                has_kpoints = (
                    hasattr(self, "kpoints") and self.kpoints is not None
                )
                has_klines = (
                    hasattr(self, "klines") and self.klines is not None
                )

                if has_kpoints and has_klines:
                    original_kpoints = self.kpoints.clone()
                    original_klines = self.klines.clone()
                    self.periodic = Periodic(
                        self.geometry,
                        self.geometry.cell,
                        cutoff=self.pairwise_cutoff_length,
                        kpoints=original_kpoints,
                        klines=original_klines,
                    )
                elif has_kpoints:
                    original_kpoints = self.kpoints.clone()
                    self.periodic = Periodic(
                        self.geometry,
                        self.geometry.cell,
                        cutoff=self.pairwise_cutoff_length,
                        kpoints=original_kpoints,
                    )
                elif has_klines:
                    original_klines = self.klines.clone()
                    self.periodic = Periodic(
                        self.geometry,
                        self.geometry.cell,
                        cutoff=self.pairwise_cutoff_length,
                        klines=original_klines,
                    )
                else:
                    self.periodic = Periodic(
                        self.geometry, self.geometry.cell, cutoff=self.pairwise_cutoff_length
                    )

                # Update k_weights and max_nk
                self.k_weights = self.periodic.k_weights
                self.max_nk = torch.max(self.periodic.n_kpoints)

            except Exception as e:
                print(f"Warning: Could not recreate periodic structure: {e}")

            # Clear cached properties
            self._fermi_energy = None
            self._forces = None
            self._band_gap = None
            self._occupations = None

            # Recalculate with strained geometry
            eigenvalues = self()

            # Calculate electronic energy
            electronic_energy = self._calculate_electronic_energy()

            # Store results
            volume = torch.det(strained_cell).abs()
            volumes.append(volume)
            energies.append(electronic_energy)

            print(f"  Volume: {volume.item():.6f} Bohr³")
            print(f"  Electronic energy: {electronic_energy.item():.8f} Ha")

            # Test force calculation (optional - comment out if not needed)
            try:
                forces = self._compute_forces_finite_diff()
                # forces = self.get_forces()
                print(
                    f"  Max force component: {torch.max(torch.abs(forces)).item():.6f} Ha/Bohr"
                )
            except Exception as e:
                print(f"  Force calculation failed: {e}")

        # Restore original geometry
        self.geometry.cell = original_cell
        self.geometry.positions = original_positions

        # Restore gradient requirement if it was originally set
        if not original_positions.requires_grad:
            self.geometry.positions.requires_grad_(False)

        # Convert to tensors
        volumes = torch.stack(volumes)
        energies = torch.stack(energies)

        # Fit equation of state
        if method == "birch_murnaghan":
            bulk_modulus, eq_volume, eq_energy = self._fit_birch_murnaghan(
                volumes, energies
            )
        else:  # polynomial
            bulk_modulus, eq_volume, eq_energy = self._fit_polynomial_eos(
                volumes, energies
            )

        return {
            "bulk_modulus": bulk_modulus,  # GPa
            "equilibrium_volume": eq_volume,
            "equilibrium_energy": eq_energy,
            "strains": strains,
            "volumes": volumes,
            "energies": energies,
        }

    def calculate_bulk_modulus_old(
        self, strain_range=0.02, num_strains=5, method="birch_murnaghan"
    ):
        """
        Calculate bulk modulus using finite strain method.

        Parameters:
        -----------
        strain_range : float
            Maximum strain to apply (±strain_range)
        num_strains : int
            Number of strain points to calculate
        method : str
            'birch_murnaghan' or 'polynomial' fitting method

        Returns:
        --------
        dict : Dictionary containing:
            - 'bulk_modulus' : Bulk modulus in GPa
            - 'equilibrium_volume' : Equilibrium volume
            - 'equilibrium_energy' : Equilibrium energy
            - 'strains' : Applied strains
            - 'volumes' : Volumes for each strain
            - 'energies' : Energies for each strain
        """
        print("Calculating bulk modulus...")

        # Store original geometry
        original_cell = self.geometry.cell.clone()
        original_positions = self.geometry.positions.clone()

        # Create strain points
        strains = torch.linspace(-strain_range, strain_range, num_strains)
        volumes = []
        energies = []

        for i, strain in enumerate(strains):
            print(f"Strain point {i+1}/{num_strains}: strain = {strain:.4f}")

            # Apply isotropic strain to cell
            strain_factor = 1.0 + strain
            strained_cell = original_cell * strain_factor

            # Update geometry with strained cell
            self.geometry.cell = strained_cell

            # Recalculate with strained geometry
            eigenvalues = self()

            # Calculate electronic energy
            electronic_energy = self._calculate_electronic_energy()

            # Store results
            volume = torch.det(strained_cell).abs()
            volumes.append(volume)
            energies.append(electronic_energy)
            print("electronic_energy", electronic_energy)
            print("forces", self._compute_forces_finite_diff())
        # Restore original geometry
        self.geometry.cell = original_cell
        self.geometry.positions = original_positions

        # Convert to tensors
        volumes = torch.stack(volumes)
        energies = torch.stack(energies)

        # Fit equation of state
        if method == "birch_murnaghan":
            bulk_modulus, eq_volume, eq_energy = self._fit_birch_murnaghan(
                volumes, energies
            )
        else:  # polynomial
            bulk_modulus, eq_volume, eq_energy = self._fit_polynomial_eos(
                volumes, energies
            )

        return {
            "bulk_modulus": bulk_modulus,  # GPa
            "equilibrium_volume": eq_volume,
            "equilibrium_energy": eq_energy,
            "strains": strains,
            "volumes": volumes,
            "energies": energies,
        }

    def calculate_ev_curve(
        self,
        strain_range=0.2,
        num_points=15,
        method="birch_murnaghan",
        plot=True,
        save_path="EV_curve.png",
        figsize=(8, 6),
        cutoff=16.0,
    ):
        """
        Calculate and plot Energy-Volume (EV) curve for equation of state.

        Parameters:
        -----------
        strain_range : float
            Maximum strain range (±strain_range)
        num_points : int
            Number of strain points to calculate
        method : str
            Fitting method ('birch_murnaghan' or 'polynomial')
        plot : bool
            Whether to create and show plot
        save_path : str
            Path to save the plot
        figsize : tuple
            Figure size for the plot

        Returns:
        --------
        dict : Dictionary containing EV curve data and fitted parameters
        """
        print(f"Calculating EV curve with {num_points} points...")
        # Note: this is not tested yet
        # Store original geometry and periodic structure info
        original_cell = self.geometry.cell.clone()
        original_positions = self.geometry.positions.clone()
        original_periodic = self.periodic  # Keep reference to original

        # Store original k-point configuration
        has_kpoints = hasattr(self, "kpoints") and self.kpoints is not None
        has_klines = hasattr(self, "klines") and self.klines is not None
        kpoints = torch.tensor([5, 5, 5])
        if has_kpoints:
            original_kpoints = self.kpoints.clone()
        if has_klines:
            original_klines = self.klines.clone()

        # Create strain points (more points for smoother curve)
        strains = torch.linspace(-strain_range, strain_range, num_points)
        volumes = []
        energies = []
        total_energies = []  # Store total energies including repulsive

        def get_energy_at_positions(cell, positions):
            """Get energy for given positions."""

            # cell = torch.tensor(
            #    [
            #        [6.3573, -0.0000, 3.6704],
            #        [2.1191, 5.9937, 3.6704],
            #        [-0.0000, -0.0000, 7.3408],
            #    ]
            # )
            # geometry = Geometry(torch.tensor([[14, 14]]), positions, cell)
            geometry = Geometry(self.geometry.atomic_numbers, positions, cell)
            # print("positions",positions)
            calc = SimpleDftb(
                geometry,
                shell_dict=self.shell_dict,
                kpoints=kpoints,
                # klines=klines,
                h_feed=self.h_feed,
                s_feed=self.s_feed,
                nelectron=self.nelectron,
                pairwise_cutoff_length=self.pairwise_cutoff_length,
            )

            # Compute properties
            eigenvalues = calc()
            # Clear cache
            # self._fermi_energy = None
            # self._band_gap = None
            # self._occupations = None

            # Recalculate
            # self()
            # return torch.sum(
            #    eigenvalues
            # )
            return self._calculate_electronic_energy()

        # Need repulsion term
        for i, strain in enumerate(strains):
            print(f"EV point {i+1}/{num_points}: strain = {strain:.4f}")

            # Apply isotropic strain to cell
            strain_factor = 1.0 + strain
            strained_cell = original_cell * strain_factor

            # Scale positions proportionally with cell (important!)
            strained_positions = original_positions * strain_factor
            electronic_energy = get_energy_at_positions(
                strained_cell, strained_positions
            )
            volume = torch.det(strained_cell).abs()
            # print("orig cell",self.geometry.cell)
            # print("strained_cell",strained_cell)
            print(
                "RUNNING EV,i,strain,energy,volume",
                i,
                strain,
                electronic_energy,
                volume,
            )
            # Update geometry with strained cell and positions
            self.geometry.cell = strained_cell
            self.geometry.positions = strained_positions

            volumes.append(volume)
            energies.append(
                electronic_energy
            )  # Electronic only for comparison
            # total_energies.append(total_energy)  # Total energy for EOS fitting

        print("energies", torch.tensor(energies).unsqueeze(0))
        print("volumes", torch.tensor(volumes).unsqueeze(0))
        return energies, volumes

        # Fit equation of state
        if method == "birch_murnaghan":
            bulk_modulus, eq_volume, eq_energy = self._fit_birch_murnaghan(
                volumes, energies
            )
        else:  # polynomial
            bulk_modulus, eq_volume, eq_energy = self._fit_polynomial_eos(
                volumes, energies
            )

        return (volumes, energies)
        """
        # Create fitted curve for plotting
        vol_fit = torch.linspace(volumes.min(), volumes.max(), 100)

        if method == "polynomial":
            # Generate fitted polynomial curve
            V_np = volumes.detach().cpu().numpy().flatten()
            E_np = energies_for_fitting.detach().cpu().numpy().flatten()

            import numpy as np

            try:
                coeffs = np.polyfit(V_np, E_np, 2)
                c, b, a = coeffs

                vol_fit_np = vol_fit.detach().cpu().numpy()
                energy_fit_np = a + b * vol_fit_np + c * vol_fit_np**2
                energy_fit = torch.from_numpy(energy_fit_np).to(volumes.device)
            except:
                # Fallback to simple interpolation
                energy_fit = torch.interp(
                    vol_fit, volumes, energies_for_fitting
                )
        else:
            # Simple interpolation for Birch-Murnaghan (placeholder)
            energy_fit = torch.interp(vol_fit, volumes, energies_for_fitting)
        print("volumes", volumes)
        print("energies", energies)
        return (volumes, energies)
        """

    def _calculate_electronic_energy(self):
        """Calculate electronic energy from current eigenvalues."""
        # Ensure we have current eigenvalues
        if not hasattr(self, "eigenvalue") or self.eigenvalue is None:
            # Recalculate if needed
            self()
        # print("self.get_fermi_energy()",self.get_fermi_energy())
        fermi_energy = self.get_fermi_energy()

        # Calculate occupations using Fermi-Dirac distribution
        kT_hartree = 0.025 / H2E  # Convert eV to Hartree
        occupations = fermi_smearing(self.eigenvalue, fermi_energy, kT_hartree)

        # Calculate electronic energy
        electronic_energy = torch.sum(
            occupations * self.eigenvalue * self.k_weights.unsqueeze(-1)
        )

        return electronic_energy.real

    def _fit_birch_murnaghan(self, volumes, energies):
        """Fit Birch-Murnaghan equation of state."""
        # This is a simplified implementation
        # For a more robust fit, you might want to use scipy.optimize

        # Find minimum energy point
        min_idx = torch.argmin(energies)
        eq_volume = volumes[min_idx]
        eq_energy = energies[min_idx]

        # Simple finite difference approximation for bulk modulus
        # B = -V * d²E/dV² at equilibrium
        if len(volumes) >= 3 and min_idx > 0 and min_idx < len(volumes) - 1:
            # Get neighboring points for finite difference
            dV_forward = volumes[min_idx + 1] - volumes[min_idx]
            dV_backward = volumes[min_idx] - volumes[min_idx - 1]
            dE_forward = energies[min_idx + 1] - energies[min_idx]
            dE_backward = energies[min_idx] - energies[min_idx - 1]

            # Second derivative using finite differences
            d2E_dV2 = (
                2
                * (dE_forward / dV_forward - dE_backward / dV_backward)
                / (dV_forward + dV_backward)
            )

            # Bulk modulus: B = -V * d²E/dV²
            bulk_modulus = -eq_volume * d2E_dV2

            # Convert from Hartree/Bohr³ to GPa
            bulk_modulus = bulk_modulus * 29421.02648  # Conversion factor

            # Take absolute value and ensure it's positive
            bulk_modulus = torch.abs(bulk_modulus)
        else:
            bulk_modulus = torch.tensor(100.0)  # Default reasonable value

        return bulk_modulus, eq_volume, eq_energy

    def _fit_polynomial_eos(self, volumes, energies):
        """Fit polynomial equation of state."""
        # Fit quadratic polynomial E(V) = a + b*V + c*V²
        # Bulk modulus B = V * d²E/dV² = 2*c*V

        # Convert to numpy and ensure 1D arrays
        V_np = volumes.detach().cpu().numpy().flatten()
        E_np = energies.detach().cpu().numpy().flatten()

        print(f"Debug: V_np shape: {V_np.shape}, E_np shape: {E_np.shape}")

        # Ensure we have enough points
        if len(V_np) < 3:
            # Return default values
            eq_volume = volumes[torch.argmin(energies)]
            eq_energy = energies[torch.argmin(energies)]
            bulk_modulus = torch.tensor(100.0)  # Default value
            return bulk_modulus, eq_volume, eq_energy

        # Fit quadratic polynomial using numpy
        import numpy as np

        try:
            coeffs = np.polyfit(V_np, E_np, 2)
            c, b, a = coeffs  # polyfit returns highest degree first

            # Convert back to tensors
            a = torch.tensor(a, device=volumes.device, dtype=volumes.dtype)
            b = torch.tensor(b, device=volumes.device, dtype=volumes.dtype)
            c = torch.tensor(c, device=volumes.device, dtype=volumes.dtype)

            # Find equilibrium volume: dE/dV = b + 2*c*V = 0
            eq_volume = -b / (2 * c)
            eq_energy = a + b * eq_volume + c * eq_volume**2

            # Bulk modulus: B = V * d²E/dV² = 2*c*V
            bulk_modulus = 2 * c * eq_volume
            # Convert from Hartree/Bohr³ to GPa
            bulk_modulus = torch.abs(bulk_modulus) * 29421.02648

        except Exception as e:
            print(f"Polynomial fitting failed: {e}")
            # Fallback to simple minimum finding
            min_idx = torch.argmin(energies)
            eq_volume = volumes[min_idx]
            eq_energy = energies[min_idx]
            bulk_modulus = torch.tensor(100.0, device=volumes.device)

        return bulk_modulus, eq_volume, eq_energy

    def calculate_band_structure_properties(self):
        """
        Calculate detailed band structure properties.

        Returns:
        --------
        dict : Dictionary containing band structure analysis
        """
        eigenvals = self.get_eigenvalues(fermi_shift=True, unit="eV")
        fermi_energy = self.get_fermi_energy() * H2E  # Convert to eV

        # Basic properties
        nbatch, nkpoints, nbands = eigenvals.shape

        # Find valence and conduction bands
        occupied_mask = eigenvals < 0  # Below Fermi level (shifted to zero)
        unoccupied_mask = eigenvals >= 0  # Above Fermi level

        # Valence band analysis
        valence_bands = eigenvals[occupied_mask]
        if len(valence_bands) > 0:
            vbm = torch.max(valence_bands)
            vbm_indices = torch.where(eigenvals == vbm)
            valence_band_width = torch.max(valence_bands) - torch.min(
                valence_bands
            )
        else:
            vbm = torch.tensor(float("-inf"))
            vbm_indices = (
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([0]),
            )
            valence_band_width = torch.tensor(0.0)

        # Conduction band analysis
        conduction_bands = eigenvals[unoccupied_mask]
        if len(conduction_bands) > 0:
            cbm = torch.min(conduction_bands)
            cbm_indices = torch.where(eigenvals == cbm)
            conduction_band_width = torch.max(conduction_bands) - torch.min(
                conduction_bands
            )
        else:
            cbm = torch.tensor(float("inf"))
            cbm_indices = (
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([0]),
            )
            conduction_band_width = torch.tensor(0.0)

        # Band gap analysis
        band_gap = (
            cbm - vbm
            if vbm != float("-inf") and cbm != float("inf")
            else torch.tensor(0.0)
        )

        # Direct vs indirect gap analysis
        direct_gaps = []
        for ik in range(nkpoints):
            kpoint_bands = eigenvals[0, ik, :]
            kpoint_valence = kpoint_bands[kpoint_bands < 0]
            kpoint_conduction = kpoint_bands[kpoint_bands > 0]

            if len(kpoint_valence) > 0 and len(kpoint_conduction) > 0:
                kpoint_vbm = torch.max(kpoint_valence)
                kpoint_cbm = torch.min(kpoint_conduction)
                direct_gaps.append(kpoint_cbm - kpoint_vbm)

        if direct_gaps:
            direct_gaps = torch.stack(direct_gaps)
            minimum_direct_gap = torch.min(direct_gaps)
            direct_gap_kpoint = torch.argmin(direct_gaps)
        else:
            minimum_direct_gap = torch.tensor(float("inf"))
            direct_gap_kpoint = torch.tensor(0)

        # Effective mass calculation (simplified)
        effective_masses = self._calculate_effective_masses(eigenvals)

        return {
            "band_gap": band_gap.item(),
            "vbm": vbm.item() if vbm != float("-inf") else None,
            "cbm": cbm.item() if cbm != float("inf") else None,
            "valence_band_width": valence_band_width.item(),
            "conduction_band_width": conduction_band_width.item(),
            "minimum_direct_gap": (
                minimum_direct_gap.item()
                if minimum_direct_gap != float("inf")
                else None
            ),
            "direct_gap_kpoint": direct_gap_kpoint.item(),
            "is_direct_semiconductor": (
                abs(band_gap - minimum_direct_gap) < 0.01
                if minimum_direct_gap != float("inf")
                else False
            ),
            "effective_masses": effective_masses,
            "nkpoints": nkpoints,
            "nbands": nbands,
            "fermi_energy_original": fermi_energy,
        }

    def _calculate_effective_masses(self, eigenvals):
        """
        Calculate effective masses using finite differences.
        This is a simplified implementation.
        """
        # This requires k-point spacing information and second derivatives
        # For now, return placeholder values
        return {
            "electron_mass": None,  # Would need actual calculation
            "hole_mass": None,  # Would need actual calculation
            "note": "Effective mass calculation requires k-point derivatives",
        }

    def get_properties_dict(
        self, kT=0.025, include_bulk_modulus=False, include_dos_data=False
    ):
        """
        Get comprehensive dictionary of calculated electronic and mechanical properties.

        Parameters:
        -----------
        kT : float
            Electronic temperature in eV
        include_bulk_modulus : bool
            Whether to calculate and include bulk modulus (computationally expensive)
        include_dos_data : bool
            Whether to calculate and include DOS data

        Returns:
        --------
        dict : Dictionary containing various properties
        """
        # fermi_energy = 0 #self.get_fermi_energy(kT)
        # band_gap_info = 0 #self.calculate_band_gap(kT)
        # print('DOUBLE')
        try:
            fermi_energy = self.get_fermi_energy(kT)
            band_gap_info = self.calculate_band_gap(kT)
        except:
            fermi_energy = torch.tensor(0)
            band_gap_info = {}
            band_gap_info["vbm"] = torch.tensor(0)  # 0
            band_gap_info["cbm"] = torch.tensor(0)  # 0
            band_gap_info["gap"] = torch.tensor(0)  # 0
            band_gap_info["direct"] = True
            band_gap_info["vbm_kpoint"] = 0
            band_gap_info["cbm_kpoint"] = 0
            print("Check for errors 1")
        # Try to get band structure properties, but handle if it fails
        try:
            band_structure_props = self.calculate_band_structure_properties()
            valence_width = band_structure_props["valence_band_width"]
            conduction_width = band_structure_props["conduction_band_width"]
            min_direct_gap = band_structure_props["minimum_direct_gap"]
            is_direct_semi = band_structure_props["is_direct_semiconductor"]
        except:
            print("Check for errors 2")
            valence_width = 0.0
            conduction_width = 0.0
            min_direct_gap = None
            is_direct_semi = False

        properties = {
            # Basic electronic properties
            "fermi_energy_eV": (fermi_energy * H2E).item(),
            "fermi_energy_Ha": fermi_energy.item(),
            "band_gap_eV": band_gap_info["gap"],
            # "band_gap_eV": band_gap_info["gap"].item(),
            "vbm_eV": band_gap_info["vbm"].item(),
            "cbm_eV": band_gap_info["cbm"].item(),
            "is_direct_gap": band_gap_info["direct"],
            "vbm_kpoint": band_gap_info["vbm_kpoint"],
            "cbm_kpoint": band_gap_info["cbm_kpoint"],
            # Band structure properties
            "valence_band_width_eV": valence_width,
            "conduction_band_width_eV": conduction_width,
            "minimum_direct_gap_eV": min_direct_gap,
            "is_direct_semiconductor": is_direct_semi,
            # System properties
            "nkpoints": self.max_nk.item(),
            "nbands": self.eigenvalue.shape[-1],
            "nelectrons": int(self.nelectron.item()),
        }

        # Add bulk modulus data if requested
        if include_bulk_modulus:
            print("Calculating bulk modulus for properties dict...")
            try:
                bulk_info = self.calculate_bulk_modulus(
                    strain_range=0.01, num_strains=5
                )
                properties.update(
                    {
                        "bulk_modulus_GPa": bulk_info["bulk_modulus"].item(),
                        "equilibrium_volume_Bohr3": bulk_info[
                            "equilibrium_volume"
                        ].item(),
                        "equilibrium_energy_Ha": bulk_info[
                            "equilibrium_energy"
                        ].item(),
                    }
                )
            except Exception as e:
                print(f"Warning: Bulk modulus calculation failed: {e}")
                properties.update(
                    {
                        "bulk_modulus_GPa": None,
                        "equilibrium_volume_Bohr3": None,
                        "equilibrium_energy_Ha": None,
                    }
                )

        # Add DOS data if requested
        if include_dos_data:
            # print("Calculating DOS data for properties dict...")
            try:
                energy_grid, dos = self.calculate_dos(
                    energy_range=(-10, 10),
                    num_points=5000,
                    sigma=0.1,
                    fermi_shift=True,
                )

                # Keep everything as tensors for ML compatibility
                # Find DOS at Fermi level (energy closest to 0 when fermi_shifted)
                fermi_idx = torch.argmin(torch.abs(energy_grid))
                dos_at_fermi = dos[fermi_idx]

                # Find band gap from DOS (where DOS is minimum near Fermi level)
                # Create mask for region around Fermi level (±2 eV)
                fermi_region_mask = torch.abs(energy_grid) < 2.0
                fermi_region_dos = dos[fermi_region_mask]
                fermi_region_energies = energy_grid[fermi_region_mask]

                # Find minimum DOS in Fermi region
                min_dos_idx_local = torch.argmin(fermi_region_dos)
                gap_center_energy = fermi_region_energies[min_dos_idx_local]

                # Calculate total states using trapezoidal integration
                # torch.trapz equivalent
                dx = energy_grid[1] - energy_grid[0]  # Uniform grid spacing
                total_states = torch.trapz(dos, energy_grid)

                properties.update(
                    {
                        # Store as Python scalars for JSON serialization, but computed with torch
                        "dos_at_fermi": dos_at_fermi.item(),
                        "dos_gap_center_eV": gap_center_energy.item(),
                        "dos_total_states": total_states.item(),
                        # For ML training, you might want to keep these as tensors:
                        "dos_at_fermi_tensor": dos_at_fermi,  # Keep tensor for gradients
                        "dos_gap_center_tensor": gap_center_energy,  # Keep tensor
                        "dos_total_states_tensor": total_states,  # Keep tensor
                        # Optionally store full arrays as tensors (memory intensive)
                        "dos_energy_grid_tensor": energy_grid,  # Full tensor
                        "dos_values_tensor": dos,  # Full tensor
                        # Convert to lists for JSON compatibility
                        #'dos_energy_grid_eV': energy_grid.detach().cpu().numpy().tolist(),
                        #'dos_values': dos.detach().cpu().numpy().tolist(),
                    }
                )
            except Exception as e:
                print(f"Warning: DOS calculation failed: {e}")
                properties.update(
                    {
                        "dos_at_fermi": None,
                        "dos_gap_center_eV": None,
                        "dos_total_states": None,
                        "dos_at_fermi_tensor": None,
                        "dos_gap_center_tensor": None,
                        "dos_total_states_tensor": None,
                        "dos_energy_grid_tensor": None,
                        "dos_values_tensor": None,
                        "dos_energy_grid_eV": None,
                        "dos_values": None,
                    }
                )

        return properties
        """
        Get dictionary of calculated electronic properties.
        
        Parameters:
        -----------
        kT : float
            Electronic temperature in eV
            
        Returns:
        --------
        dict : Dictionary containing various electronic properties
        """
        fermi_energy = self.get_fermi_energy(kT)
        band_gap_info = self.calculate_band_gap(kT)

        properties = {
            "fermi_energy_eV": (fermi_energy * H2E).item(),
            "fermi_energy_Ha": fermi_energy.item(),
            "band_gap_eV": band_gap_info["gap"],
            # "band_gap_eV": band_gap_info["gap"].item(),
            "vbm_eV": band_gap_info["vbm"].item(),
            "cbm_eV": band_gap_info["cbm"].item(),
            "is_direct_gap": band_gap_info["direct"],
            "vbm_kpoint": band_gap_info["vbm_kpoint"],
            "cbm_kpoint": band_gap_info["cbm_kpoint"],
            "nkpoints": self.max_nk.item(),
            "nbands": self.eigenvalue.shape[-1],
            "nelectrons": int(self.nelectron.item()),
        }

        return properties


# Example usage
if __name__ == "__main__":
    # Read geometry
    atoms = Atoms.from_poscar("tests/POSCAR-SiC.vasp")
    atoms = Atoms.from_poscar("tests/POSCAR")
    geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
    cell = torch.tensor(
        [
            [6.3573, -0.0000, 3.6704],
            [2.1191, 5.9937, 3.6704],
            [-0.0000, -0.0000, 7.3408],
        ]
    )

    # pos=pos.requires_grad_(True)
    geometry = Geometry(
        torch.tensor([[14, 14]]),
        torch.tensor([[[7.4169, 5.2445, 12.8464], [1.0596, 0.7492, 1.8352]]]),
        cell,
    )
    geometry.positions.requires_grad_(True)
    # Setup k-points and k-lines
    kpoints2 = torch.tensor([5, 5, 5])  # For DOS
    atoms = Atoms.from_poscar("tests/POSCAR-Al")
    geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
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

    # Setup parameters
    shell_dict = generate_shell_dict_upto_Z65()
    path_to_skf = "tests/Si-Si.skf"
    path_to_skf = "tests/Al-Al.skf"
    from slakonet.skf import Skf
    from slakonet.interpolation import PolyInterpU, BSpline

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
    """
    print("hs_dict",hs_dict)
    import sys
    sys.exit()

    # Initialize feeds
    h_feed = SkfFeed.from_dir(
        path_to_skf,
        shell_dict,
        skf_type="skf",
        geometry=geometry,
        integral_type="H",
    )
    s_feed = SkfFeed.from_dir(
        path_to_skf,
        shell_dict,
        skf_type="skf",
        geometry=geometry,
        integral_type="S",
    )
    skparams = SkfParamFeed.from_dir(
        path_to_skf, geometry, skf_type="skf", repulsive=True
    )
    """

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
        pairwise_cutoff_length=5,
    )

    # Run calculation
    print("Computing band structure...")
    eigenvalues_bands = calc_bands()
    # print("forces",calc_bands.get_forces())
    # print("forces", calc_bands._compute_forces_finite_diff())
    # x, y = calc_bands.calculate_phonon_modes()
    print("\nPlotting band structure...")
    fig_bands, ax_bands = calc_bands.plot_band_structure(
        fermi_shift=True, save_path="bands_enhanced.png"
    )
    plt.show()
    import sys

    sys.exit()
    # Create calculator for DOS (with k-point grid)
    calc_dos = SimpleDftb(
        geometry,
        shell_dict=shell_dict,
        kpoints=kpoints2,
        h_feed=h_feed,
        s_feed=s_feed,
        nelectron=nelectron,
        pairwise_cutoff_length=5,
    )

    print("Computing DOS...")
    eigenvalues_dos = calc_dos()
    properties = calc_dos.get_properties_dict(
        include_bulk_modulus=True,  # Include bulk modulus calculation
        include_dos_data=False,  # Include DOS data
    )
    # print("properties", properties)
    import sys

    sys.exit()

    # Test that methods exist
    print("Testing method availability...")
    methods_to_test = [
        "get_fermi_energy",
        "calculate_band_gap",
        "calculate_band_structure_properties",
        "get_properties_dict",
    ]
    for method_name in methods_to_test:
        if hasattr(calc_dos, method_name):
            print(f"✓ {method_name}: Available")
        else:
            print(f"✗ {method_name}: Missing")

    # Get comprehensive properties (including DOS and bulk modulus)
    print("\n=== Comprehensive Electronic Properties ===")
    properties = calc_dos.get_properties_dict(
        include_bulk_modulus=True,  # Include bulk modulus calculation
        include_dos_data=True,  # Include DOS data
    )

    # Print key properties (not the full DOS arrays)
    key_properties = {
        k: v for k, v in properties.items() if not isinstance(v, list)
    }  # Skip DOS arrays for printing

    for key, value in key_properties.items():
        print(f"{key}: {value}")

    # Calculate band gap
    band_gap_info = calc_dos.calculate_band_gap()
    print(f"\nBand gap: {band_gap_info['gap']:.3f} eV")
    print(f"Direct gap: {band_gap_info['direct']}")

    # Plot band structure
    print("\nPlotting band structure...")
    fig_bands, ax_bands = calc_bands.plot_band_structure(
        fermi_shift=True, save_path="bands_enhanced.png"
    )
    plt.show()

    # Plot DOS
    print("Plotting DOS...")
    fig_dos, ax_dos = calc_dos.plot_dos(
        energy_range=(-8, 5),
        sigma=0.1,
        fermi_shift=True,
        save_path="dos_enhanced.png",
    )
    plt.show()

    # Calculate DOS tensor (stays on GPU for ML)
    energy_grid, dos_tensor = calc_dos.calculate_dos(
        energy_range=(-10, 8), fermi_shift=True
    )
    print(f"\nDOS tensor shape: {dos_tensor.shape}")
    print(f"DOS tensor device: {dos_tensor.device}")
    print(f"Energy grid device: {energy_grid.device}")

    # Calculate bulk modulus (computationally intensive)
    print("\n=== Bulk Modulus Calculation ===")
    bulk_modulus_info = calc_dos.calculate_bulk_modulus(
        strain_range=0.01, num_strains=5
    )
    print(f"Bulk modulus: {bulk_modulus_info['bulk_modulus'].item():.1f} GPa")
    print(
        f"Equilibrium volume: {bulk_modulus_info['equilibrium_volume'].item():.3f} Bohr³"
    )
    print(
        f"Equilibrium energy: {bulk_modulus_info['equilibrium_energy'].item():.6f} Ha"
    )

    # Calculate and plot EV curve
    print("\n=== Energy-Volume Curve ===")
    try:
        if hasattr(calc_dos, "calculate_ev_curve"):
            ev_results = calc_dos.calculate_ev_curve(
                strain_range=0.15,  # ±15% strain
                num_points=8,  # Fewer points for faster calculation
                method="polynomial",
                plot=True,
                save_path="EV_curve.png",
            )

            # Check if file was created
            import os

            if os.path.exists("EV_curve.png"):
                print("✓ EV_curve.png successfully created")
            else:
                print("✗ EV_curve.png was not created")

            # If tuple returned (with plot), extract the dict
            if isinstance(ev_results, tuple):
                fig, ax, ev_data = ev_results
                print(
                    f"EV curve bulk modulus: {ev_data['bulk_modulus'].item():.1f} GPa"
                )
            else:
                print(
                    f"EV curve bulk modulus: {ev_results['bulk_modulus'].item():.1f} GPa"
                )
        else:
            print("✗ calculate_ev_curve method not available")

    except Exception as e:
        print(f"✗ EV curve calculation failed: {e}")
        import traceback

        traceback.print_exc()

    # Get detailed band structure properties
    print("\n=== Band Structure Analysis ===")
    band_props = calc_dos.calculate_band_structure_properties()
    print(f"Valence band width: {band_props['valence_band_width']:.3f} eV")
    print(
        f"Conduction band width: {band_props['conduction_band_width']:.3f} eV"
    )
    if band_props["minimum_direct_gap"] is not None:
        print(f"Minimum direct gap: {band_props['minimum_direct_gap']:.3f} eV")
    else:
        print("Minimum direct gap: Not available")
    print(f"Is direct semiconductor: {band_props['is_direct_semiconductor']}")

    print("\nCalculation completed successfully!")
