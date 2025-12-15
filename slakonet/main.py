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
        # shell_dict=None,
        # h_feed=None,
        # s_feed=None,
        # nelectron=None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.geometry = geometry
        self.model = model
        self.max_Z = max_Z
        self.cutoff = cutoff
        self.repulsive = repulsive
        self.with_eigenvectors = with_eigenvectors
        self.kT = kT
        self.H2E = H2E

        # Setup basis and feeds
        self.shell_dict = generate_shell_dict_upto_Z65()
        self.basis = Basis(self.geometry.atomic_numbers, self.shell_dict)
        self.updated_skfs = self.model.get_updated_skfs()
        self.h_feed = create_feeds(self.updated_skfs, self.shell_dict, "H")
        self.s_feed = create_feeds(self.updated_skfs, self.shell_dict, "S")

        # Precompute electron lookup table (gradient-safe)
        self.electron_lookup = self._build_electron_lookup()
        self.nelectron = self._compute_nelectrons()
        print("self.geometry", self.geometry)
        print("self.geometry.cell", self.geometry.cell)
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
        """Solve H*c = E*S*c for all k-points."""
        n_kpoints = self.max_nk.item()
        eigenvalues_list = []
        eigenvecs_list = []
        occupations_list = []

        for ik in range(n_kpoints):
            h_k = H[..., ik]
            s_k = S[..., ik]

            # Solve generalized eigenvalue problem
            eigenvals, eigenvecs = eighb(h_k, s_k, scheme="chol")

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

        # Get unique atom pairs
        uan = self.periodic.unique_atomic_numbers()
        n_global = len(uan)
        uap = torch.stack(
            [uan.repeat(n_global), uan.repeat_interleave(n_global)]
        ).T.to(self.device)

        # Get distance matrix (gradient-preserving!)
        atom_pairs = self.basis.atomic_number_matrix("atomic").to(self.device)
        dist_vecs = self.periodic.distance_vectors.to(self.device)
        dist_mat = torch.sqrt((dist_vecs**2).sum(-1) + 1e-10)

        energy = torch.zeros_like(dist_mat)

        # Loop over unique atom pairs
        for iap in uap:
            mask_dist = dist_mat.gt(1e-8)

            element_i = atomic_num_to_symbol.get(iap[0].item())
            element_j = atomic_num_to_symbol.get(iap[1].item())

            if element_i is None or element_j is None:
                continue

            element_pair = "-".join(tuple(sorted([element_i, element_j])))

            if element_pair not in self.updated_skfs:
                continue

            skf = self.updated_skfs[element_pair]
            r_cutoff = skf.r_spline.cutoff
            mask_cut = dist_mat.lt(r_cutoff)

            # Atom pair mask
            mask_i = atom_pairs[..., 0] == iap[0]
            mask_j = atom_pairs[..., 1] == iap[1]
            mask_pair_central = mask_i & mask_j

            if dist_mat.shape[1] != mask_pair_central.shape[1]:
                mask_pair = mask_pair_central.unsqueeze(1)
            else:
                mask_pair = mask_pair_central

            mask = mask_pair & mask_dist & mask_cut

            if not mask.any():
                continue

            d_mask = dist_mat[mask]

            # 1. Exponential repulsive
            r_a123 = skf.r_spline.exp_coef.to(self.device)
            exp_contribution = (
                torch.exp(-r_a123[0] * d_mask + r_a123[1]) + r_a123[2]
            )

            energy_contrib = torch.zeros_like(dist_mat)
            energy_contrib[mask] = exp_contribution
            energy = energy + energy_contrib

            # 2. Spline repulsive
            r_table = skf.r_spline.spline_coef.to(self.device)
            grid = skf.r_spline.grid.to(self.device)

            mask2 = (
                dist_mat.le(grid[-1]) & dist_mat.ge(grid[0]) & mask & mask_dist
            )

            if mask2.any():
                d_mask2 = dist_mat[mask2]
                ind1 = torch.searchsorted(grid, d_mask2) - 1
                ind1 = torch.clamp(ind1, 0, len(grid) - 2)

                r_pol = r_table[ind1]
                deltar = d_mask2 - grid[ind1]

                spline_contribution = (
                    r_pol[..., 0]
                    + r_pol[..., 1] * deltar
                    + r_pol[..., 2] * deltar**2
                    + r_pol[..., 3] * deltar**3
                )

                energy_contrib2 = torch.zeros_like(dist_mat)
                energy_contrib2[mask2] = spline_contribution
                energy = energy + energy_contrib2

            # 3. Tail spline
            r_table_l = skf.r_spline.tail_coef.to(self.device)
            grid_l = skf.r_spline.grid.to(self.device)

            mask_l = (
                dist_mat.le(grid_l[1])
                & dist_mat.ge(grid_l[0])
                & mask
                & mask_dist
            )

            if mask_l.any():
                d_mask_l = dist_mat[mask_l]
                ind_l = torch.searchsorted(grid_l, d_mask_l) - 1
                ind_l = torch.clamp(ind_l, 0, len(grid_l) - 2)

                deltar_l = d_mask_l - grid_l[ind_l]

                tail_contribution = (
                    r_table_l[0]
                    + r_table_l[1] * deltar_l
                    + r_table_l[2] * deltar_l**2
                    + r_table_l[3] * deltar_l**3
                    + r_table_l[4] * deltar_l**4
                    + r_table_l[5] * deltar_l**5
                )

                energy_contrib3 = torch.zeros_like(dist_mat)
                energy_contrib3[mask_l] = tail_contribution
                energy = energy + energy_contrib3

        # Sum (0.5 to avoid double counting)
        if energy.dim() == 4:
            total_rep = 0.5 * energy.sum(-1).sum(-1).sum(-1)
        else:
            total_rep = 0.5 * energy.sum(-1).sum(-1)

        return total_rep

    def calculate(self, compute_forces=True):
        """Main calculation method."""
        # Enable gradient tracking for positions if forces needed
        if compute_forces:
            self.geometry.positions.requires_grad_(True)

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

        # Repulsive energy
        if self.repulsive:
            potential_energy = self._compute_repulsive_energy()
            total_energy = electronic_energy - potential_energy
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
        if compute_forces:
            forces = torch.autograd.grad(
                total_energy,
                self.geometry.positions,
                create_graph=True,
                # create_graph=False,
                retain_graph=False,
            )[0]

        # Store results
        self._results = {
            "energy": total_energy,
            "eigenvalues": eigenvalues_shifted,
            "fermi_energy": fermi_energy,
            "electronic_energy": electronic_energy,
            "potential_energy": potential_energy,
            "forces": forces,
            "bandgap": bandgap,
            "occupations": occupations,
        }

        if self.with_eigenvectors:
            self._results["eigenvectors"] = eigenvectors

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


import torch
from ase.calculators.calculator import Calculator, all_changes


class SimpleDftbCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model, kpoints=None, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.model = model  # .to(device).eval()
        self.device = device

    def calculate(
        self,
        atoms=None,
        properties=None,
        kpoints=None,
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        # ASE → Geometry
        device = "cpu"
        Z = torch.tensor(
            atoms.get_atomic_numbers(), device=device, dtype=torch.long
        )
        pos = torch.tensor(atoms.positions, device=device, dtype=torch.float32)
        cell = torch.tensor(atoms.cell, device=device, dtype=torch.float32)
        kpoints = torch.tensor([2, 2, 2], device=device)
        print("Z", Z)
        print("pos", pos)
        print("cell", cell)
        geometry = Geometry(Z, pos, cell, units="angstrom")
        # geometry.device = device
        geometry.positions.requires_grad_(True)

        # Run DFTB
        dftb = SimpleDftb(
            geometry,
            kpoints=kpoints,
            model=self.model,
        )

        results = dftb.calculate()

        energy = results["energy"].sum()
        forces = results["forces"][0]

        # Return to ASE
        self.results["energy"] = energy.detach().cpu().item()
        self.results["forces"] = forces.detach().cpu().numpy()


def run_calc(ase_atoms=None, model_path=None):
    model = MultiElementSkfParameterOptimizer.load_ultra_compact(model_path)
    geometry = Geometry.from_ase_atoms([ase_atoms])
    geometry.positions.requires_grad_(True)
    kpoints = torch.tensor([2, 2, 2])  # For DOS
    device = "cuda"
    model = model.to(device)
    model.eval()

    model = model.float()
    s = SimpleDftb(geometry, kpoints=kpoints, model=model)
    ##s = SimpleDftb(geometry,klines=klines,model=model)
    # print('ele',s.nelectron)
    res = s.calculate()
    print("res", res)
    return res


import torch
from ase.calculators.calculator import Calculator, all_changes
from slakonet.atoms import Geometry


class SlakoNetCalculatorX(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model_path, kpoints=[2, 2, 2], device="cuda", **kwargs):
        super().__init__(**kwargs)
        self.model = MultiElementSkfParameterOptimizer.load_ultra_compact(
            model_path
        )
        self.model = self.model.to(device).eval().float()
        self.device = device
        self.kpoints = torch.tensor(kpoints) if kpoints else None

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        # Convert ASE atoms to Geometry
        geometry = Geometry.from_ase_atoms([atoms]).to(self.device)
        geometry.positions.requires_grad_(True)

        # Run calculation
        dftb = SimpleDftb(geometry, kpoints=self.kpoints, model=self.model)
        results = dftb.calculate()

        # Store results
        self.results["energy"] = results["energy"].sum().detach().cpu().item()
        self.results["forces"] = results["forces"][0].detach().cpu().numpy()


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
