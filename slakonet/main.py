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
# torch.set_default_dtype(torch.float32)
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
        model=None,
        ham=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.geometry = geometry
        self.shell_dict = shell_dict
        self.h_feed = h_feed
        self.s_feed = s_feed
        # self.dtype = torch.complex64
        self.dtype = torch.complex128
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
        self.model = model
        self.ham = ham
        if self.h_feed is None:
            updated_skfs = self.get_updated_skfs()
        if kpoints is not None and klines is not None:
            self.periodic = Periodic(
                self.geometry.to(device),
                self.geometry.cell.to(device),
                cutoff=20.0,
                kpoints=kpoints,
                klines=klines,
            )
        elif kpoints is not None:
            self.periodic = Periodic(
                self.geometry, self.geometry.cell, cutoff=20.0, kpoints=kpoints
            )  # ??????????????????????? 20 Angtrom
        elif klines is not None:
            self.periodic = Periodic(
                self.geometry, self.geometry.cell, cutoff=20.0, klines=klines
            )
        else:
            self.periodic = Periodic(
                self.geometry, self.geometry.cell, cutoff=20.0
            )

        self.kpoints = self.periodic.kpoints
        self.k_weights = self.periodic.k_weights.to(self.device)
        self.max_nk = torch.max(self.periodic.n_kpoints)
        self._original_kpoints = kpoints
        self._original_klines = klines
        # Cache for computed properties
        self._fermi_energy = None
        self._forces = None
        self._band_gap = None
        self._occupations = None

    @classmethod
    def load_model(cls, load_path):
        """
        Load the model using different methods

        Args:
            load_path: Path to load the model from
            method: 'state_dict', 'full_model', or 'universal_params'
            skf_directory: SKF directory (needed for some methods)
        """
        load_path = Path(load_path)
        t1 = time.time()
        load_file = Path(load_path).with_suffix(".pt")
        compact_data = torch.load(load_file)

        if not compact_data["metadata"].get("ultra_compact", False):
            raise ValueError("This is not an ultra-compact model file")

        metadata = compact_data["metadata"]
        state_dict = compact_data["trained_parameters"]
        skf_metadata = compact_data["skf_metadata"]

        # Create new instance
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)

        # Restore basic attributes
        instance.skf_directory = metadata["skf_directory"]
        instance.elements_in_system = set(metadata["elements_in_system"])
        instance.element_pairs = set(
            tuple(pair) for pair in metadata["element_pairs"]
        )

        # Recreate atomic number mapping
        from jarvis.core.specie import atomic_numbers_to_symbols

        zz = [i for i in range(1, 100)]
        z = atomic_numbers_to_symbols(zz)
        instance.atomic_num_to_symbol = dict(zip(zz, z))

        # Recreate SKF optimizers
        instance.skf_optimizers = nn.ModuleDict()

        for pair_key in metadata["available_pairs"]:
            # Create optimizer
            optimizer = SkfParameterOptimizer.__new__(SkfParameterOptimizer)
            nn.Module.__init__(optimizer)

            # Get the metadata (everything except hamiltonian/overlap)
            skf_dict = skf_metadata[pair_key].copy()

            # Extract trained parameters for this pair from state_dict
            h_params = {}
            s_params = {}

            for key, value in state_dict.items():
                if key.startswith(f"skf_optimizers.{pair_key}.h_params."):
                    param_name = key.replace(
                        f"skf_optimizers.{pair_key}.h_params.", ""
                    )
                    h_params[param_name] = value
                elif key.startswith(f"skf_optimizers.{pair_key}.s_params."):
                    param_name = key.replace(
                        f"skf_optimizers.{pair_key}.s_params.", ""
                    )
                    s_params[param_name] = value

            # Reconstruct full skf_dict with trained parameters
            skf_dict["hamiltonian"] = h_params
            skf_dict["overlap"] = s_params

            optimizer.skf_dict = skf_dict

            # Create parameter dicts
            optimizer.h_params = nn.ParameterDict(
                {k: nn.Parameter(v.clone()) for k, v in h_params.items()}
            )
            optimizer.s_params = nn.ParameterDict(
                {k: nn.Parameter(v.clone()) for k, v in s_params.items()}
            )

            # Set other attributes
            optimizer.grid = skf_dict.get("grid", None)
            optimizer.atomic_data = skf_dict.get("atomic_data", None)
            optimizer.atom_pair = skf_dict.get("atom_pair", None)
            optimizer.hs_cutoff = skf_dict.get("hs_cutoff", None)

            instance.skf_optimizers[pair_key] = optimizer

        # Load the state dict (this should work since we reconstructed the structure)
        instance.load_state_dict(state_dict)
        t2 = time.time()

        print(f"✅ Compact model loaded from: {load_file}")
        print("Time taken:", round(t2 - t1, 3))
        return instance

    def compute_hs_matrices(self):
        """Compute Hamiltonian and overlap matrices."""
        # print("Computing H and S matrices...")
        self.ham = hs_matrix(self.periodic, self.basis, self.h_feed)
        self.overlap = hs_matrix(self.periodic, self.basis, self.s_feed)
        self.ham = self.ham.to(self.device)
        self.overlap = self.overlap.to(self.device)

    def get_electronic_energy(self):
        if not self.ham:
            self.compute_hs_matrices()
        eigenvalues_list = []
        occupations_list = []
        for ik in range(self.max_nk):
            # self.ham: (batch_size,n_orb,n_orb,n_kpoints)
            h_k = self.ham[..., ik]
            s_k = self.overlap[..., ik]
            eigenvals, eigenvecs = eighb(h_k, s_k)
            # Get occupations (CHECK: does fermi preserve gradients?)
            occ, fermi_level = fermi(eigenvals, self.nelectron)
            eigenvalues_list.append(eigenvals)
            occupations_list.append(occ)
        eigenvalues = torch.stack(eigenvalues_list)
        occupations = torch.stack(occupations_list)
        electronic_energy = torch.sum(
            eigenvalues * occupations * self.k_weights.unsqueeze(-1)
        )
        if electronic_energy.is_complex():
            electronic_energy = torch.real(electronic_energy)
        return electronic_energy

    def evaluate_repulsive_spline(self, s, r):
        """Evaluate repulsive energy at distance r."""
        if r > s.r_spline.cutoff:
            return torch.tensor(0.0)
        elif r < s.r_spline.grid[0]:
            # Exponential: c + exp(-a*r + b)
            return s.r_spline.exp_coef[2] + torch.exp(
                -s.r_spline.exp_coef[0] * r + s.r_spline.exp_coef[1]
            )
        else:
            # Cubic spline: c0 + c1*dr + c2*dr^2 + c3*dr^3
            idx = torch.searchsorted(s.r_spline.grid, r) - 1
            idx = torch.clamp(idx, 0, len(s.r_spline.grid) - 2)
            c = s.r_spline.spline_coef[idx]
            dr = r - s.r_spline.grid[idx]
            return c[0] + c[1] * dr + c[2] * dr**2 + c[3] * dr**3

    def get_total_energy(self):
        rep = self.get_repulsive_energy()
        original_cell = self.geometry.cell.clone()

        original_positions = self.geometry.positions.clone()
        original_positions.requires_grad_(True)
        # rep_forces, _ = torch.autograd.grad(
        #        rep,
        #        self.geometry.positions,
        #        create_graph=True,
        #    )
        # print('rep forces',rep_forces)
        elec = self._calculate_electronic_energy()
        elec_forces = torch.autograd.grad(
            elec,
            self.periodic.positions,
            create_graph=True,
        )
        print("elec forces", elec_forces)
        print("rep", rep)
        print("elec", elec)
        return rep + elec

    def get_repulsive_energy(self):
        self.rep_energy = torch.zeros(self.periodic.n_atoms.shape)
        uan = self.periodic.unique_atomic_numbers()
        n_global = len(uan)
        uap = (
            torch.stack(
                [uan.repeat(n_global), uan.repeat_interleave(n_global)]
            ).T
        ).to(self.device)
        atom_pairs = self.basis.atomic_number_matrix("atomic").to(self.device)

        # Get the device from periodic.distances to ensure consistency
        device = self.device  # self.periodic.distances.device
        energy = torch.zeros(self.periodic.distances.shape, device=device)

        from jarvis.core.specie import atomic_numbers_to_symbols

        zz = [i for i in range(1, 100)]
        # self.periodic.distances=self.periodic.distances.to(device)
        z = atomic_numbers_to_symbols(zz)
        atomic_num_to_symbol = dict(zip(zz, z))
        dist_mat = self.periodic.distances.to(device)
        for iap in uap:
            # get rid of the same atom interaction
            mask_dist = dist_mat.ne(0)
            element_symbol_i = atomic_num_to_symbol.get(iap[0].item())
            element_symbol_j = atomic_num_to_symbol.get(iap[1].item())
            element_pair = "-".join(
                tuple(sorted([element_symbol_i, element_symbol_j]))
            )
            skf = self.model.get_updated_skfs()[element_pair]
            r_cutoff = skf.r_spline.cutoff

            # get mask for different atom pairs
            mask_cut = dist_mat.lt(r_cutoff)

            # Expand atom_pairs to match the periodic images dimension
            atom_pairs_expanded = atom_pairs.unsqueeze(-2).expand(
                -1, -1, -1, mask_dist.shape[-1], -1
            )
            print(
                "device = self.periodic.distances",
                self.periodic.distances.device,
            )
            # Now compare with iap
            mask = (
                ((iap == atom_pairs_expanded).sum(-1) == 2)
                * mask_dist
                * mask_cut
            )
            d_mask = dist_mat[mask]

            # 1. exponential repulsive
            r_a123 = skf.r_spline.exp_coef.to(
                device
            )  # Ensure coefficients are on the right device
            energy[mask] = (
                energy[mask]
                + torch.exp(-r_a123[0] * d_mask + r_a123[1])
                + r_a123[2]
            )

            # 2. spline repulsive
            r_table = skf.r_spline.spline_coef.to(device)
            grid = skf.r_spline.grid.to(device)

            mask2 = (
                dist_mat.le(grid[-1]) * dist_mat.ge(grid[0]) * mask
            ) * mask_dist
            ind1 = (torch.searchsorted(grid, dist_mat) - 1)[mask2]
            r_pol = r_table[ind1]
            deltar = dist_mat[mask2] - grid[ind1]
            energy[mask2] = (
                r_pol[..., 0]
                + r_pol[..., 1] * deltar
                + r_pol[..., 2] * deltar**2
                + r_pol[..., 3] * deltar**3
            )

            # 3. bounds distances spline repulsive
            r_table_l = skf.r_spline.tail_coef.to(device)
            grid_l = skf.r_spline.grid.to(device)
            mask_l = (
                dist_mat.le(grid_l[1]) * dist_mat.ge(grid_l[0]) * mask
            ) * mask_dist
            ind_l = (torch.searchsorted(grid_l, dist_mat) - 1)[mask_l]
            deltar_l = dist_mat[mask_l] - grid_l[ind_l]
            if mask_l.any():
                energy[mask_l] = (
                    r_table_l[0]
                    + r_table_l[1] * deltar_l
                    + r_table_l[2] * deltar_l**2
                    + r_table_l[3] * deltar_l**3
                    + r_table_l[4] * deltar_l**4
                    + r_table_l[5] * deltar_l**5
                )

        if not dist_mat.dim() == 4:
            return 0.5 * energy.sum(-1).sum(-1)
        else:
            return 0.5 * energy.sum(-1).sum(-1).sum(-1)

    def solve_kpoint(self, ik):
        """Solve eigenvalue problem for k-point ik."""
        # Get matrices for this k-point
        h_k = self.ham[..., ik]
        s_k = self.overlap[..., ik]
        # print('h_kkkk',h_k.shape)
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
        print("HERE Eigenvals", self.eigenvalue)
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

    def calculate_phonon_modes(
        self,
        line_density=5,
        write_fc=True,
        dim=[1, 1, 1],
        distance=0.05,
        electron_kpoints=[5, 5, 5],
    ):
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
        # dim = [1, 1, 1]
        # distance = 0.05
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
                kpoints=torch.tensor(electron_kpoints),
                # klines=klines,
                h_feed=self.h_feed,
                s_feed=self.s_feed,
                nelectron=self.nelectron,
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

        # Calculate BS
        lbls = kpoints.labels
        lbls_ticks = []
        freqs = []
        tmp_kp = []
        lbls_x = []
        count = 0
        for ii, k in enumerate(kpoints.kpts):
            k_str = ",".join(map(str, k))
            if ii == 0:
                tmp = []
                for i, freq in enumerate(self.phonon.get_frequencies(k)):
                    tmp.append(freq)
                freqs.append(tmp)
                tmp_kp.append(k_str)
                lbl = "$" + str(lbls[ii]) + "$"
                lbls_ticks.append(lbl)
                lbls_x.append(count)
                count += 1
                # lbls_x.append(ii)
            elif k_str != tmp_kp[-1]:
                tmp_kp.append(k_str)
                tmp = []
                for i, freq in enumerate(self.phonon.get_frequencies(k)):
                    tmp.append(freq)
                freqs.append(tmp)
                lbl = lbls[ii]
                if lbl != "":
                    lbl = "$" + str(lbl) + "$"
                    lbls_ticks.append(lbl)
                    # lbls_x.append(ii)
                    lbls_x.append(count)
                count += 1
        # lbls_x = np.arange(len(lbls_ticks))
        freq_conversion_factor = 33.3566830
        freqs = np.array(freqs)
        freqs = freqs * freq_conversion_factor
        # print('freqs',freqs,freqs.shape)
        # the_grid = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.0)
        # plt.rcParams.update({"font.size": 18})
        plt.figure(figsize=(10, 5))
        # plt.subplot(the_grid[0])
        for i in range(freqs.shape[1]):
            plt.plot(freqs[:, i], lw=2, c="b")
        for i in lbls_x:
            plt.axvline(x=i, c="black")
        plt.xticks(lbls_x, lbls_ticks)
        # print('lbls_x',lbls_x,len(lbls_x))
        # print('lbls_ticks',lbls_ticks,len(lbls_ticks))
        plt.ylabel("Frequency (cm$^{-1}$)")
        plt.xlim([0, max(lbls_x)])
        plt.savefig("phonon_bands.png")
        plt.close()

        # Calculate phonon DOS
        self.phonon.run_mesh(
            [40, 40, 40], is_gamma_center=True, is_mesh_symmetry=False
        )
        self.phonon.run_total_dos()
        tdos = self.phonon._total_dos
        freqs = tdos.frequency_points
        ds = tdos.dos
        # freqs, ds = tdos.get_dos()
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

    def get_total_energy_forces_dos(
        self, compute_forces=True, compute_dos=False, dos_params=None
    ):
        """
        Unified function to compute total energy, forces, and optionally DOS.

        Parameters:
        -----------
        compute_forces : bool
            Whether to compute forces
        compute_dos : bool
            Whether to compute DOS
        dos_params : dict, optional
            Parameters for DOS calculation (energy_range, num_points, sigma, etc.)

        Returns:
        --------
        dict : Dictionary containing:
            - 'total_energy': Total energy (Ha)
            - 'electronic_energy': Electronic contribution (Ha)
            - 'repulsive_energy': Repulsive contribution (Ha)
            - 'forces': Forces on atoms (Ha/Bohr) if compute_forces=True
            - 'dos_energy_grid': DOS energy grid if compute_dos=True
            - 'dos_values': DOS values if compute_dos=True
        """

        # Step 1: Enable gradients on positions if computing forces
        if compute_forces and not self.geometry.positions.requires_grad:
            self.geometry.positions.requires_grad_(True)

        # Step 2: Recreate Periodic with gradient-enabled positions
        print("Recreating Periodic with gradient-enabled positions...")
        periodic_kwargs = {}
        if self._original_kpoints is not None:
            periodic_kwargs["kpoints"] = self._original_kpoints
        if self._original_klines is not None:
            periodic_kwargs["klines"] = self._original_klines

        self.periodic = Periodic(
            self.geometry, self.geometry.cell, cutoff=20.0, **periodic_kwargs
        )

        # Update dependent attributes
        self.kpoints = self.periodic.kpoints
        self.k_weights = self.periodic.k_weights.to(self.device)
        self.max_nk = torch.max(self.periodic.n_kpoints)

        # Step 3: Compute H and S matrices with gradients
        self.compute_hs_matrices()

        # Step 4: Calculate electronic energy with proper gradient handling
        print("Computing electronic energy...")
        eigenvalues_list = []
        occupations_list = []

        for ik in range(self.max_nk):
            h_k = self.ham[..., ik]
            s_k = self.overlap[..., ik]

            # Solve eigenvalue problem
            eigenvals, eigenvecs = eighb(h_k, s_k)

            # Get occupations
            occ, fermi_level = fermi(eigenvals, self.nelectron)

            eigenvalues_list.append(eigenvals)
            occupations_list.append(occ)

        eigenvalues = torch.stack(eigenvalues_list)
        occupations = torch.stack(occupations_list)

        # Store for DOS calculation
        self.eigenvalue = eigenvalues.permute(1, 0, 2)
        self._occupations = occupations.permute(1, 0, 2)

        # Calculate electronic energy (ensure real part)
        electronic_energy = torch.sum(
            eigenvalues * occupations * self.k_weights.unsqueeze(-1)
        )
        if electronic_energy.is_complex():
            electronic_energy = torch.real(electronic_energy)

        # Step 5: Calculate repulsive energy
        print("Computing repulsive energy...")
        repulsive_energy = self.get_repulsive_energy()

        # Ensure both energies have the same dtype
        if electronic_energy.dtype != repulsive_energy.dtype:
            repulsive_energy = repulsive_energy.to(electronic_energy.dtype)

        # Step 6: Total energy
        total_energy = electronic_energy + repulsive_energy

        print(f"Electronic energy: {electronic_energy.item():.6f} Ha")
        print(f"Repulsive energy: {repulsive_energy.item():.6f} Ha")
        print(f"Total energy: {total_energy.item():.6f} Ha")

        # Initialize results dictionary
        results = {
            "total_energy": total_energy,
            "electronic_energy": electronic_energy,
            "repulsive_energy": repulsive_energy,
        }

        # Step 7: Calculate forces if requested
        if compute_forces:
            print("Computing forces...")
            if not total_energy.requires_grad:
                print("WARNING: Total energy doesn't require gradients!")
                forces = torch.zeros_like(self.geometry.positions)
            else:
                grad_outputs = torch.autograd.grad(
                    outputs=total_energy,
                    inputs=self.geometry.positions,
                    create_graph=False,
                    retain_graph=compute_dos,  # Retain if we need to compute DOS
                    allow_unused=False,
                )
                forces = -grad_outputs[0]

            results["forces"] = forces
            self._forces = forces

        # Step 8: Calculate DOS if requested
        if compute_dos:
            print("Computing DOS...")
            if dos_params is None:
                dos_params = {
                    "energy_range": (-10, 5),
                    "num_points": 5000,
                    "sigma": 0.1,
                    "fermi_shift": True,
                    "unit": "eV",
                }

            try:
                energy_grid, dos = self.calculate_dos(**dos_params)
                results["dos_energy_grid"] = energy_grid
                results["dos_values"] = dos
            except Exception as e:
                print(f"DOS calculation failed: {e}")
                results["dos_energy_grid"] = None
                results["dos_values"] = None

        return results

    def get_forcess(self):
        """Calculate forces via automatic differentiation."""

        # Step 1: Enable gradients on positions
        if not self.geometry.positions.requires_grad:
            self.geometry.positions.requires_grad_(True)

        # Step 2: Recreate Periodic to capture gradients
        print("Recreating Periodic with gradient-enabled positions...")

        # Build kwargs for Periodic
        periodic_kwargs = {}
        if self._original_kpoints is not None:
            periodic_kwargs["kpoints"] = self._original_kpoints
        if self._original_klines is not None:
            periodic_kwargs["klines"] = self._original_klines

        # Recreate Periodic object
        self.periodic = Periodic(
            self.geometry, self.geometry.cell, cutoff=20.0, **periodic_kwargs
        )

        # Update dependent attributes
        self.kpoints = self.periodic.kpoints
        self.k_weights = self.periodic.k_weights.to(self.device)
        self.max_nk = torch.max(self.periodic.n_kpoints)

        # Step 3: Recompute matrices with new Periodic
        self.compute_hs_matrices()

        # Step 4: Recompute eigenvalues
        print("Recomputing eigenvalues with gradients enabled...")
        self()

        # Step 5: Compute energy
        print("Computing total energy...")
        energy = self.get_total_energy()

        # Debug: Check for NaN in intermediate values
        print(f"\n=== Debugging ===")
        print(f"Ham has NaN: {torch.isnan(self.ham).any()}")
        print(f"Overlap has NaN: {torch.isnan(self.overlap).any()}")
        print(f"Eigenvalues has NaN: {torch.isnan(self.eigenvalue).any()}")
        print(f"Energy: {energy}")
        print(f"Energy requires_grad: {energy.requires_grad}")

        if not energy.requires_grad:
            raise RuntimeError("Energy doesn't require grad after recreation!")

        # Step 6: Calculate gradient
        print("Computing gradients...")
        try:
            # Use torch.autograd.grad with specific settings for stability
            grad_output = torch.autograd.grad(
                outputs=energy,
                inputs=self.geometry.positions,
                grad_outputs=None,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
                allow_unused=False,
            )[0]

            # Check for NaN/Inf
            if torch.isnan(grad_output).any():
                nan_count = torch.isnan(grad_output).sum().item()
                print(
                    f"WARNING: {nan_count} NaN values detected in gradients!"
                )

                # Print more debug info
                print(f"Gradient stats:")
                print(
                    f"  Min: {grad_output[~torch.isnan(grad_output)].min().item() if (~torch.isnan(grad_output)).any() else 'all NaN'}"
                )
                print(
                    f"  Max: {grad_output[~torch.isnan(grad_output)].max().item() if (~torch.isnan(grad_output)).any() else 'all NaN'}"
                )

                # Fall back to finite differences
                print("Falling back to finite differences...")
                self._forces = self._compute_forces_finite_diff()

            elif torch.isinf(grad_output).any():
                inf_count = torch.isinf(grad_output).sum().item()
                print(
                    f"WARNING: {inf_count} Inf values detected in gradients!"
                )
                print("Falling back to finite differences...")
                self._forces = self._compute_forces_finite_diff()

            else:
                self._forces = -grad_output
                max_force = torch.max(torch.abs(self._forces)).item()
                print(
                    f"✓ Forces computed successfully! Max component: {max_force:.6f}"
                )

        except RuntimeError as e:
            print(f"✗ Gradient computation failed: {e}")
            print("Falling back to finite differences...")
            self._forces = self._compute_forces_finite_diff()

        return self._forces

    def get_forcessX(self):
        """Calculate forces via automatic differentiation."""

        # Step 1: Enable gradients on positions
        if not self.geometry.positions.requires_grad:
            self.geometry.positions.requires_grad_(True)

        # Step 2: CRITICAL - Recreate Periodic to capture gradients
        # The old self.periodic.distances was computed before requires_grad was set!
        print("Recreating Periodic with gradient-enabled positions...")

        # Store original k-point configuration
        if hasattr(self, "kpoints") and self.kpoints is not None:
            kpoints = self.kpoints
        else:
            kpoints = None

        if hasattr(self, "klines") and self.klines is not None:
            klines = self.klines
        else:
            klines = None

        # Recreate Periodic object - this will compute distances from gradient-enabled positions
        if kpoints is not None and klines is not None:
            self.periodic = Periodic(
                self.geometry.to(self.device),
                self.geometry.cell.to(self.device),
                cutoff=20.0,
                kpoints=kpoints,
                klines=klines,
            )
        elif kpoints is not None:
            self.periodic = Periodic(
                self.geometry,
                self.geometry.cell,
                cutoff=20.0,
                kpoints=kpoints,
            )
        elif klines is not None:
            self.periodic = Periodic(
                self.geometry,
                self.geometry.cell,
                cutoff=20.0,
                klines=klines,
            )
        else:
            self.periodic = Periodic(
                self.geometry,
                self.geometry.cell,
                cutoff=20.0,
            )

        # Update dependent attributes
        self.kpoints = self.periodic.kpoints
        self.k_weights = self.periodic.k_weights.to(self.device)
        self.max_nk = torch.max(self.periodic.n_kpoints)

        # Step 3: Recompute matrices with new Periodic
        self.compute_hs_matrices()

        # Step 4: Recompute eigenvalues
        print("Recomputing eigenvalues with gradients enabled...")
        self()  # This calls __call__ which computes eigenvalues

        # Step 5: Now compute energy - should have gradients
        print("Computing total energy...")
        energy = self.get_total_energy()

        print(f"\n=== After recreation ===")
        print(
            f"Positions require_grad: {self.geometry.positions.requires_grad}"
        )
        print(
            f"Periodic distances require_grad: {self.periodic.distances.requires_grad}"
        )
        print(f"Energy: {energy}")
        print(f"Energy requires_grad: {energy.requires_grad}")
        print(f"Energy grad_fn: {energy.grad_fn}")

        if not energy.requires_grad:
            raise RuntimeError(
                "Energy still doesn't require grad after recreation!\n"
                "Check your Periodic class - distances might not depend on positions."
            )

        # Step 6: Calculate gradient
        print("Computing gradients...")
        grad_output = torch.autograd.grad(
            energy,
            self.geometry.positions,
            create_graph=True,
        )[0]

        self._forces = -grad_output

        print(
            f"Forces computed! Max component: {torch.max(torch.abs(self._forces)).item():.6f}"
        )

        return self._forces

    def get_forcessX(self):
        original_cell = self.geometry.cell.clone()

        original_positions = self.geometry.positions.clone()
        original_positions.requires_grad_(True)
        if self._forces is None:
            self._forces, _ = torch.autograd.grad(
                self.get_total_energy(),
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
                        cutoff=20.0,
                        kpoints=original_kpoints,
                        klines=original_klines,
                    )
                elif has_kpoints:
                    original_kpoints = self.kpoints.clone()
                    self.periodic = Periodic(
                        self.geometry,
                        self.geometry.cell,
                        cutoff=20.0,
                        kpoints=original_kpoints,
                    )
                elif has_klines:
                    original_klines = self.klines.clone()
                    self.periodic = Periodic(
                        self.geometry,
                        self.geometry.cell,
                        cutoff=20.0,
                        klines=original_klines,
                    )
                else:
                    self.periodic = Periodic(
                        self.geometry, self.geometry.cell, cutoff=20.0
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
    atoms = Atoms.from_poscar("tests/POSCAR").make_supercell_matrix([2, 2, 2])
    geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
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

    # Setup parameters
    shell_dict = generate_shell_dict_upto_Z65()
    path_to_skf = "tests/Si-Si.skf"
    path_to_skf = "tests/Al-Al.skf"
    from slakonet.skf import Skf
    from slakonet.interpolation import PolyInterpU

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
