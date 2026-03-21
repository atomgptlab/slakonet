import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from slakonet.optim import (
    MultiElementSkfParameterOptimizer,
    get_atoms,
    kpts_to_klines,
    default_model,
)
from jarvis.core.kpoints import Kpoints3D as Kpoints
from slakonet.atoms import Geometry
from slakonet.main import generate_shell_dict_upto_Z65
import torch
from jarvis.core.atoms import Atoms
import argparse
from jarvis.io.vasp.inputs import Poscar
import argparse
import sys
import time

plt.rcParams.update({"font.size": 18})

H2E = 27.211
parser = argparse.ArgumentParser(description="SlakoNet Pretrained Models")
parser.add_argument(
    "--model_path",
    default=None,
    # default="slakonet/tests/slakonet_v1_sic",
    help="Provide model path ",
)

parser.add_argument(
    "--pairwise_cutoff_length",
    type=float,
    required=True,
    help="Maximum length for pairwise interactions measured in Angstroms",
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)

parser.add_argument(
    "--file_path",
    default=None,
    help="Path to atomic structure file.",
)

parser.add_argument(
    "--output_filename",
    default="slakonet_bands_dos.png",
    help="Path to desired output file name",
)

parser.add_argument(
    "--energy_range",
    default="-8 8",
    help="Energy range for bandstructure and DOS plots",
)


parser.add_argument(
    "--jid",
    default="JVASP-107",
    help="JARVIS-DFT Identifier",
)
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_trained_model(model_path, method="compact"):
    model = MultiElementSkfParameterOptimizer.load_ultra_compact(
        # model = MultiElementSkfParameterOptimizer.load_model(
        model_path
        # model_path, method="state_dict"
    )
    # model.float()
    # model=model.half()
    model.eval()
    return model


def get_properties(jid="", model=None, atoms=None, dataset=None, pairwise_cutoff_length=None):
    if atoms is None:
        atoms, opt_gap, mbj_gap = get_atoms(jid=jid, dataset=dataset)
    if model is None:
        model = default_model()

    # model=model.float()
    geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
    shell_dict = generate_shell_dict_upto_Z65()

    kpoints = Kpoints().kpath(atoms, line_density=20)
    klines = kpts_to_klines(kpoints.kpts, default_points=2)

    with torch.no_grad():
        properties, success = model.compute_multi_element_properties(
            geometry=geometry,
            shell_dict=shell_dict,
            klines=klines,
            get_fermi=True,
            with_eigenvectors=True,
            device=device,
            pairwise_cutoff_length=pairwise_cutoff_length,
        )

    if not success:
        raise RuntimeError("Failed to compute properties")

    return properties, atoms, kpoints


def _format_kpath_ticks(labels):
    """
    Make safe mathtext tick labels; skip empties and
    dedup repeats; normalize Gamma.
    """
    xticks, xtick_labels = [], []
    last = None
    for i, lbl in enumerate(labels):
        if not lbl or lbl.strip() == "":
            continue
        if lbl in ("G", r"\Gamma", "Γ"):
            show = r"$\Gamma$"
        else:
            show = rf"${lbl}$"
        if show != last:
            xticks.append(i)
            xtick_labels.append(show)
            last = show
    return xticks, xtick_labels


def compute_orbital_projected_dos(
    properties,
    geometry,
    sigma=0.1,
    energy_range=(-8, 6),
):
    """Compute orbital-projected DOS (s, p, d, f) from eigenvectors with Gaussian broadening."""
    # Eigen info from calculator (assumed in eV)
    fermi_eV = properties["fermi_energy_eV"]
    eigenvalues = properties["calc"].eigenvalue * H2E  # [1, nk, nb]
    eigenvectors = properties["calc"].eigenvectors  # [1, norb, nb, nk]

    # Get atom types from geometry
    atom_types = geometry.chemical_symbols[0]  # e.g., ['Zn', 'Zn', 'O', 'O']
    unique_atoms = list(dict.fromkeys(atom_types))

    # Get orbital info from basis
    basis = properties["calc"].basis
    orbs_per_atom = basis.orbs_per_atom[0].cpu().numpy()  # [9, 9, 4, 4]
    shells_per_atom = (
        basis.shells_per_atom[0].cpu().numpy()
    )  # [3, 3, 2, 2] for [spd, spd, sp, sp]
    on_atoms = (
        basis.on_atoms[0].cpu().numpy()
    )  # which atom each orbital belongs to

    # Map shell types: 0=s(1 orbital), 1=p(3 orbitals), 2=d(5 orbitals), 3=f(7 orbitals)
    shell_names = ["s", "p", "d", "f"]
    orbitals_per_shell = [1, 3, 5, 7]

    print(f"Atom types: {atom_types}")
    print(f"Orbitals per atom: {orbs_per_atom}")
    print(f"Shells per atom: {shells_per_atom}")

    # Create energy grid
    n_points = 1000
    energy_grid = torch.linspace(
        energy_range[0], energy_range[1], n_points, device=eigenvalues.device
    )
    energy_grid_eV = energy_grid + fermi_eV

    # Initialize orbital PDOS for each atom type
    orbital_pdos = {}
    for atom in unique_atoms:
        orbital_pdos[atom] = {
            shell: torch.zeros(n_points, device=eigenvalues.device)
            for shell in shell_names
        }

    # Build orbital-to-atom-and-shell mapping
    orbital_info = []  # List of (atom_idx, atom_type, shell_type)

    orbital_idx = 0
    for atom_idx, (atom_type, n_orbs, n_shells) in enumerate(
        zip(atom_types, orbs_per_atom, shells_per_atom)
    ):
        # Determine which shells this atom has based on n_orbs
        # For example: 9 orbitals = s(1) + p(3) + d(5), so shells [0,1,2]
        # For example: 4 orbitals = s(1) + p(3), so shells [0,1]

        remaining_orbs = n_orbs
        shell_idx = 0
        while remaining_orbs > 0 and shell_idx < len(orbitals_per_shell):
            n_shell_orbs = orbitals_per_shell[shell_idx]
            if remaining_orbs >= n_shell_orbs:
                # This shell is present
                for _ in range(n_shell_orbs):
                    orbital_info.append(
                        (atom_idx, atom_type, shell_names[shell_idx])
                    )
                    orbital_idx += 1
                remaining_orbs -= n_shell_orbs
            shell_idx += 1

    print(f"Total orbitals mapped: {len(orbital_info)}")
    print(f"Example orbital mapping (first 10): {orbital_info[:10]}")

    # Gaussian normalization
    norm_factor = 1.0 / (sigma * np.sqrt(2 * np.pi))

    # Loop over k-points and bands
    batch_size, n_kpoints, n_bands = eigenvalues.shape
    for k in range(n_kpoints):
        for b in range(n_bands):
            eigenval = eigenvalues[0, k, b]
            psi = eigenvectors[0, :, b, k]  # shape [n_orbitals]
            diff = energy_grid_eV - eigenval
            gaussian = norm_factor * torch.exp(-0.5 * (diff / sigma) ** 2)

            # Project onto each orbital
            for orb_idx, (atom_idx, atom_type, shell_type) in enumerate(
                orbital_info
            ):
                orbital_weight = torch.abs(psi[orb_idx]) ** 2
                orbital_pdos[atom_type][shell_type] += (
                    orbital_weight * gaussian
                )

    # Average over k-points
    for atom in orbital_pdos:
        for shell in orbital_pdos[atom]:
            orbital_pdos[atom][shell] /= n_kpoints

    # Convert to numpy
    energy_np = energy_grid.detach().cpu().numpy()
    orbital_pdos_np = {}
    for atom in orbital_pdos:
        orbital_pdos_np[atom] = {
            shell: pdos.detach().cpu().numpy()
            for shell, pdos in orbital_pdos[atom].items()
        }

    return energy_np, orbital_pdos_np, unique_atoms


def plot_orbital_projected_dos(
    energy_np,
    orbital_pdos_np,
    unique_atoms,
    fermi_eV=0.0,
    filename="orbital_pdos.png",
):
    """Plot orbital-projected DOS for each atom type."""
    import matplotlib.pyplot as plt

    n_atoms = len(unique_atoms)
    fig, axes = plt.subplots(n_atoms, 1, figsize=(8, 4 * n_atoms), sharex=True)

    if n_atoms == 1:
        axes = [axes]

    colors = {"s": "blue", "p": "red", "d": "green", "f": "purple"}

    for idx, atom in enumerate(unique_atoms):
        ax = axes[idx]

        for shell in ["s", "p", "d", "f"]:
            if shell in orbital_pdos_np[atom]:
                dos = orbital_pdos_np[atom][shell]
                if (
                    dos.max() > 1e-6
                ):  # Only plot if there's significant contribution
                    ax.fill_between(
                        energy_np,
                        dos,
                        alpha=0.5,
                        label=f"{atom}-{shell}",
                        color=colors[shell],
                    )
                    ax.plot(energy_np, dos, color=colors[shell], linewidth=1)

        ax.axvline(
            x=0, color="black", linestyle="--", linewidth=1, label="Fermi"
        )
        ax.set_ylabel(f"{atom} DOS (states/eV)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Energy - E_F (eV)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Orbital-projected DOS saved to {filename}")

    return fig


def compute_atom_projected_dos(
    properties,
    geometry,
    sigma=0.1,
    energy_range=(-8, 6),
    filename="slakonet_bands.png",
):
    """Compute atom-type projected DOS from eigenvectors with Gaussian broadening."""
    # Eigen info from calculator (assumed in eV)
    fermi_eV = properties["fermi_energy_eV"]
    eigenvalues = properties["calc"].eigenvalue * H2E  # [1, nk, nb]
    eigenvectors = properties["calc"].eigenvectors  # [1, norb, nb, nk]

    # Get atom types from geometry
    atom_types = geometry.chemical_symbols[0]  # e.g., ['Zn', 'Zn', 'O', 'O']
    unique_atoms = list(dict.fromkeys(atom_types))  # preserve order

    # Get orbital-to-atom mapping from basis
    basis = properties["calc"].basis
    orbs_per_atom = basis.orbs_per_atom[0].cpu().numpy()  # [9, 9, 4, 4]
    on_atoms = (
        basis.on_atoms[0].cpu().numpy()
    )  # which atom each orbital belongs to

    print(f"Atom types: {atom_types}")
    print(f"Orbitals per atom: {orbs_per_atom}")
    print(f"Unique atoms: {unique_atoms}")

    # Create energy grid (relative to Fermi for plotting)
    n_points = 1000
    energy_grid = torch.linspace(
        energy_range[0], energy_range[1], n_points, device=eigenvalues.device
    )
    energy_grid_eV = energy_grid + fermi_eV

    # Initialize PDOS
    atom_pdos = {
        atom: torch.zeros(n_points, device=eigenvalues.device)
        for atom in unique_atoms
    }

    # Map orbital indices to atom types using on_atoms
    orbital_to_atom_type = []
    for orb_idx in range(len(on_atoms)):
        atom_idx = on_atoms[orb_idx]
        orbital_to_atom_type.append(atom_types[atom_idx])

    # Create mapping from atom type to orbital indices
    atom_orbital_map = {atom: [] for atom in unique_atoms}
    for orb_idx, atom_type in enumerate(orbital_to_atom_type):
        atom_orbital_map[atom_type].append(orb_idx)

    # print(f"Orbital mapping: {atom_orbital_map}")

    # Gaussian normalization
    norm_factor = 1.0 / (sigma * np.sqrt(2 * np.pi))

    # Loop over k-points and bands
    batch_size, n_kpoints, n_bands = eigenvalues.shape
    for k in range(n_kpoints):
        for b in range(n_bands):
            eigenval = eigenvalues[0, k, b]
            psi = eigenvectors[0, :, b, k]  # shape [n_orbitals]
            diff = energy_grid_eV - eigenval
            gaussian = norm_factor * torch.exp(-0.5 * (diff / sigma) ** 2)

            for atom in unique_atoms:
                orbital_indices = atom_orbital_map[atom]
                atom_weight = torch.sum(torch.abs(psi[orbital_indices]) ** 2)
                atom_pdos[atom] += atom_weight * gaussian

    # Average over k-points
    for atom in atom_pdos:
        atom_pdos[atom] /= n_kpoints

    # Convert to numpy, energies relative to Fermi
    energy_np = energy_grid.detach().cpu().numpy()
    atom_pdos_np = {
        atom: pdos.detach().cpu().numpy() for atom, pdos in atom_pdos.items()
    }

    return energy_np, atom_pdos_np, unique_atoms


def plot_band_dos_atoms(
    jid=None,
    atoms=None,
    model=None,
    model_path="slakonet_v0",
    energy_range=(-10, 10),
    filename=None,
    pairwise_cutoff_length=None,
):
    if not model:
        model = load_trained_model(model_path)
    model = model.float()
    properties, atoms, kpoints = get_properties(
        jid=jid,
        model=model,
        atoms=atoms,
        pairwise_cutoff_length=pairwise_cutoff_length
    )
    properties["model"] = model
    if filename is None:
        filename = "slakonet_out.png"
    if jid is not None and filename is None:
        filename = str(jid) + "_slakonet_out.png"

    # Band structure data (assumed eV)
    eigenvalues = properties["calc"].eigenvalue * H2E  # [1, nk, nb], eV
    fermi_eV = float(properties["fermi_energy_eV"])  # scalar eV
    formula = atoms.composition.reduced_formula
    bandgap = float(properties["band_gap_eV"])
    print("bandgap", bandgap)
    # Geometry for PDOS
    geometry = properties["calc"].geometry

    # Compute atom-projected DOS
    energy_grid, atom_pdos, unique_atoms = compute_atom_projected_dos(
        properties, geometry, energy_range=energy_range
    )

    # K-point labels
    labels = kpoints.labels
    xticks, xtick_labels = _format_kpath_ticks(labels)

    # --- Plotting (constrained layout to avoid tight_layout warnings) ---
    fig = plt.figure(figsize=(16, 6), layout="constrained")
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[3, 1, 1.5])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Bands: energy relative to Fermi
    for i in range(eigenvalues.shape[-1]):
        y = eigenvalues[0, :, i].real.detach().cpu().numpy() - fermi_eV
        ax1.plot(y, linewidth=0.8)
    ax1.axhline(0, linestyle="--", alpha=0.7)
    ax1.set_xlabel("k-point")
    ax1.set_ylabel("Energy (eV)")
    # ax1.set_title(f"{jid}  {formula}\nGap: {bandgap:.2f} eV")
    title = "(a) Gap " + str(round(bandgap, 2))
    ax1.set_title(title)
    # print("title", title)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels)
    ax1.set_ylim(energy_range)
    # ax1.set_xlim([0,(eigenvalues.shape[-1])])
    ax1.grid(True, alpha=0.3)
    # Optional vertical guides at special k-points:
    # for x in xticks: ax1.axvline(x, linewidth=0.5, alpha=0.2)

    # Total DOS: (x vs y so the curve appears horizontal)
    dos_energies = (
        properties["dos_energy_grid_tensor"]
        .detach()
        .cpu()
        .numpy()  # - fermi_eV
    )
    dos_values = properties["dos_values_tensor"].detach().cpu().numpy()
    ax2.plot(dos_values, dos_energies, linewidth=1.5)
    ax2.axhline(0, linestyle="--", alpha=0.7)
    ax2.set_xlabel("Total DOS")
    ax2.set_ylim(energy_range)
    ax1.set_title("(b)")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(left=False, labelleft=False)

    # Atom-projected DOS
    for atom in unique_atoms:
        ax3.plot(
            atom_pdos[atom], np.array(energy_grid), linewidth=1.3, label=atom
        )
        # ax3.fill_betweenx(energy_grid, 0, atom_pdos[atom], alpha=0.25)
    ax3.axhline(0, linestyle="--", alpha=0.7)
    ax3.set_xlabel("Atom PDOS")
    # ax1.set_title("(a)")
    ax1.set_title(title)
    ax2.set_title("(b)")
    ax3.set_title("(c)")
    ax3.set_ylim(energy_range)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(left=False, labelleft=False)
    ax3.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    # plt.show()

    print(f"Fermi: {fermi_eV:.3f} eV | Gap: {bandgap:.3f} eV")
    # print(f"Atom types: {unique_atoms}")

    return fig, properties, atom_pdos, energy_grid


# Usage
if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    model_path = args.model_path
    model = None
    atoms = None
    if model_path is None:
        model = default_model()
    file_path = args.file_path
    file_format = args.file_format
    output_filename = args.output_filename
    pairwise_cutoff_length = args.pairwise_cutoff_length
    energy_range = np.array(args.energy_range.split(" "), dtype="float")
    jid = args.jid
    if file_path is not None:
        if file_format == "poscar":
            atoms = Atoms.from_poscar(file_path)
        elif file_format == "cif":
            atoms = Atoms.from_cif(file_path)
        elif file_format == "xyz":
            atoms = Atoms.from_xyz(file_path, box_size=500)
        elif file_format == "pdb":
            atoms = Atoms.from_pdb(file_path, max_lat=500)
        else:
            raise NotImplementedError(
                "File format not implemented", file_format
            )

    # fig, properties, atom_pdos, energy_grid = plot_band_dos_atoms(jid='JVASP-107')
    t1 = time.time()
    fig, properties, atom_pdos, energy_grid = plot_band_dos_atoms(
        atoms=atoms,
        model_path=model_path,
        model=model,
        jid=jid,
        energy_range=energy_range,
        filename=output_filename,
        pairwise_cutoff_length = args.pairwise_cutoff_length,
    )
    t2 = time.time()
    print("Time(s)", t2 - t1)
