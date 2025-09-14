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

H2E = 27.211
parser = argparse.ArgumentParser(description="SlakoNet Pretrained Models")
parser.add_argument(
    "--model_path",
    default=None,
    # default="slakonet/tests/slakonet_v1_sic",
    help="Provide model path ",
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


def load_trained_model(model_path):
    model = MultiElementSkfParameterOptimizer.load_model(
        model_path, method="state_dict"
    )
    model.eval()
    return model


def get_properties(jid="", model="", atoms=None):
    if atoms is None:
        atoms, opt_gap, mbj_gap = get_atoms(jid)
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
    eigenvalues = (
        properties["calc"].eigenvalue * H2E
    )  # - fermi_eV  # [1, nk, nb]
    eigenvectors = properties["calc"].eigenvectors  # [1, norb, nb, nk]

    # Atom symbols (list for the single geometry in batch)
    atom_types = geometry.chemical_symbols[
        0
    ]  # e.g., ['Si', 'Si', 'Si', 'Si', 'C', 'C', 'C', 'C']
    unique_atoms = list(dict.fromkeys(atom_types))  # preserve order

    # print(f"Atom types from geometry: {atom_types}")
    # print(f"Unique atoms: {unique_atoms}")

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

    # Infer orbitals_per_atom from eigenvectors
    n_orbitals = eigenvectors.shape[1]
    n_atoms = len(atom_types)
    if n_orbitals % n_atoms != 0:
        raise ValueError(
            f"n_orbitals ({n_orbitals}) not divisible by n_atoms ({n_atoms})."
        )
    orbitals_per_atom = n_orbitals // n_atoms

    # print(f"Computing PDOS for atom types: {unique_atoms}")
    # print(f"Total atoms: {len(atom_types)}")

    # Map atoms to orbital indices (contiguous blocks per atom)
    atom_orbital_map = {}
    orbital_idx = 0
    for at in atom_types:
        if at not in atom_orbital_map:
            atom_orbital_map[at] = []
        atom_orbital_map[at].extend(
            range(orbital_idx, orbital_idx + orbitals_per_atom)
        )
        orbital_idx += orbitals_per_atom

    # print(f"Orbital mapping: {atom_orbital_map}")

    # Gaussian normalization (scalar float is fine)
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
    jid="JVASP-107",
    atoms=None,
    model=None,
    model_path="slakonet_v1_sic",
    energy_range=(-10, 10),
    filename="slakonet_out.png",
):
    if not model:
        model = load_trained_model(model_path)

    properties, atoms, kpoints = get_properties(
        jid=jid, model=model, atoms=atoms
    )

    # Band structure data (assumed eV)
    eigenvalues = properties["calc"].eigenvalue * H2E  # [1, nk, nb], eV
    fermi_eV = float(properties["fermi_energy_eV"])  # scalar eV
    formula = atoms.composition.reduced_formula
    bandgap = float(properties["band_gap_eV"])
    # print("bandgap",bandgap)
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
    ax1.set_title(f"{jid}  {formula}\nGap: {bandgap:.2f} eV")
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels)
    ax1.set_ylim(energy_range)
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
    fig, properties, atom_pdos, energy_grid = plot_band_dos_atoms(
        atoms=atoms,
        model_path=model_path,
        model=model,
        jid=jid,
        energy_range=energy_range,
        filename=output_filename,
    )
