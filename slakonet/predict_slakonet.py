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
from jarvis.db.jsonutils import dumpjson
import pprint

plt.rcParams.update({"font.size": 14})
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
parser.add_argument(
    "--cutoff",
    default="10",
    help="Pairwise cutoff",
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_trained_model(model_path, method="compact", elements=None, prefer=None):
    """Load a SlakoNet model.

    Prefers the safetensors layout (lazy, mmap) when available next to
    `model_path`. Set `prefer="pt"` (or env SLAKONET_LOADER=pt) to force the
    legacy torch.load path. Pass `elements={"Si","C"}` to materialize only
    the relevant SKF pairs.
    """
    from slakonet.optim import _smart_load_slakonet_model
    model = _smart_load_slakonet_model(
        model_path, elements=elements, prefer=prefer
    )
    model.eval()
    return model


def get_properties(jid="", model=None, atoms=None, dataset=None, cutoff=None):
    if atoms is None:
        atoms, opt_gap, mbj_gap = get_atoms(jid=jid, dataset=dataset)
    if model is None:
        model = default_model()
    # model=model.float()
    geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
    shell_dict = generate_shell_dict_upto_Z65(model=model)
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
            cutoff=cutoff,
        )
    if not success:
        raise RuntimeError("Failed to compute properties")
    # print("properties",properties)
    return properties, atoms, kpoints


def _split_path_discontinuities(eigenvalues, labels):
    """Insert NaN rows at k-path discontinuities so band lines break cleanly.

    A label containing '|' marks a jump between non-adjacent high-symmetry
    points (end of segment N | start of segment N+1). We duplicate the
    k-index, replace eigenvalues at the inserted slot with NaN so the
    polyline is broken, and split the label into ``left`` and ``right``
    at neighboring indices.
    """
    import numpy as np

    if not any(isinstance(l, str) and "|" in l for l in labels):
        return eigenvalues, list(labels)

    new_labels = []
    insert_positions = []
    for i, lbl in enumerate(labels):
        if isinstance(lbl, str) and "|" in lbl:
            left, right = lbl.split("|", 1)
            new_labels.append(left)
            new_labels.append(right)
            insert_positions.append(i)
        else:
            new_labels.append(lbl)

    if eigenvalues is None or eigenvalues.size == 0 or not insert_positions:
        return eigenvalues, new_labels

    eig = np.asarray(eigenvalues)
    nan_row = np.full_like(eig[..., :1, :], np.nan)
    pieces = []
    last = 0
    for p in insert_positions:
        pieces.append(eig[..., last : p + 1, :])
        pieces.append(nan_row)
        last = p + 1
    pieces.append(eig[..., last:, :])
    return np.concatenate(pieces, axis=-2), new_labels


def _format_kpath_ticks(labels):
    """
    Make safe mathtext tick labels; skip empties and dedup repeats; normalize Gamma.

    When two non-empty high-symmetry labels land on adjacent k-indices
    (the result of a ``|`` discontinuity having been split), merge them
    into a single ``L|R`` tick rendered at the midpoint so they don't
    visually overlap.
    """
    def _render(lbl):
        if lbl in ("G", r"\Gamma", "Γ"):
            return r"$\Gamma$"
        return rf"${lbl}$"

    raw = []
    for i, lbl in enumerate(labels):
        if not lbl or lbl.strip() == "":
            continue
        raw.append((i, lbl))

    xticks, xtick_labels = [], []
    last_text = None
    j = 0
    while j < len(raw):
        i, lbl = raw[j]
        if j + 1 < len(raw) and raw[j + 1][0] == i + 1:
            i2, lbl2 = raw[j + 1]
            show = rf"${lbl}|{lbl2}$" if lbl != lbl2 else _render(lbl)
            pos = (i + i2) / 2.0
            if show != last_text:
                xticks.append(pos)
                xtick_labels.append(show)
                last_text = show
            j += 2
            continue
        show = _render(lbl)
        if show != last_text:
            xticks.append(i)
            xtick_labels.append(show)
            last_text = show
        j += 1
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
    eigenvalues = properties["calc"].eigenvalue  # * H2E  # [1, nk, nb]
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
    energy_grid_eV = energy_grid  # - fermi_eV

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
    plt.savefig(filename, dpi=100, bbox_inches="tight")
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
    fermi_eV = properties["fermi_energy"]  # .detach().cpu().numpy()
    eigenvalues = properties["eigenvalues"]  # * H2E  # [1, nk, nb]
    eigenvectors = properties["eigenvectors"]  # [1, norb, nb, nk]

    # Get atom types from geometry
    atom_types = geometry.chemical_symbols[0]  # e.g., ['Zn', 'Zn', 'O', 'O']
    unique_atoms = list(dict.fromkeys(atom_types))  # preserve order

    # Get orbital-to-atom mapping from basis
    basis = properties["basis"]
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
    energy_grid_eV = energy_grid  # - fermi_eV

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
    print(f"eigenvectors shape: {eigenvectors.shape}")
    print(f"n_kpoints: {n_kpoints}")
    print(f"n_bands: {n_bands}")

    for k in range(n_kpoints):
        for b in range(n_bands):
            eigenval = eigenvalues[0, k, b]
            psi = eigenvectors[0, k, b, :]  # shape [n_orbitals]
            # psi = eigenvectors[0, :, b, k]  # shape [n_orbitals]

            diff = energy_grid_eV - eigenval
            gaussian = norm_factor * torch.exp(-0.5 * (diff / sigma) ** 2)

            for atom in unique_atoms:
                orbital_indices = atom_orbital_map[atom]
                atom_weight = torch.sum(torch.abs(psi[orbital_indices]) ** 2)
                atom_pdos[atom] += (atom_weight * gaussian).squeeze()
                # atom_weight * gaussian

    # Average over k-points
    for atom in atom_pdos:
        atom_pdos[atom] /= n_kpoints

    # Convert to numpy, energies relative to Fermi
    energy_np = energy_grid.detach().cpu().numpy()
    atom_pdos_np = {
        atom: pdos.detach().cpu().numpy() for atom, pdos in atom_pdos.items()
    }

    return energy_np, atom_pdos_np, unique_atoms


def compute_atom_and_orbital_pdos(
    properties,
    geometry,
    sigma=0.1,
    energy_range=(-8, 6),
):
    """Compute atom-type and shell-resolved (s/p/d/f) PDOS in one pass.

    Uses the same eigenvector convention as compute_atom_projected_dos:
        eigenvectors : [batch, n_kpoints, n_bands, n_orbitals]
    and the basis-provided orbital mapping (on_atoms, on_shells, shell_ls),
    so no fragile shell inference from orbs_per_atom is needed.

    Returns
    -------
    energy_grid_np : ndarray [n_points]
    atom_pdos_np   : {atom_type: ndarray[n_points]}
    orbital_pdos_np: {atom_type: {'s':..., 'p':..., 'd':..., 'f':...}}
    unique_atoms   : list[str]  (preserves first-occurrence order)
    """
    eigenvalues = properties["eigenvalues"]   # [1, nk, nb]
    eigenvectors = properties["eigenvectors"] # [1, nk, nb, norb]

    atom_types = geometry.chemical_symbols[0]
    unique_atoms = list(dict.fromkeys(atom_types))

    basis = properties["basis"]
    on_atoms = basis.on_atoms
    on_shells = basis.on_shells
    shell_ls = basis.shell_ls
    if on_atoms.ndim == 2:
        on_atoms = on_atoms[0]
        on_shells = on_shells[0]
        shell_ls = shell_ls[0] if shell_ls.ndim == 2 else shell_ls

    on_atoms_np = on_atoms.cpu().numpy()
    # Shell l per orbital via the global shell index (on_shells is global).
    ls_per_orb = shell_ls[on_shells.long()].cpu().numpy()

    shell_names = ["s", "p", "d", "f"]

    # Build orbital index lists per (atom_type, shell_name).
    atom_orbital_map = {atom: [] for atom in unique_atoms}
    orbital_map = {
        atom: {sh: [] for sh in shell_names} for atom in unique_atoms
    }
    for orb_idx in range(len(on_atoms_np)):
        a_idx = int(on_atoms_np[orb_idx])
        if a_idx < 0:
            continue
        atype = atom_types[a_idx]
        l = int(ls_per_orb[orb_idx])
        if l < 0 or l >= len(shell_names):
            continue
        atom_orbital_map[atype].append(orb_idx)
        orbital_map[atype][shell_names[l]].append(orb_idx)

    n_points = 1000
    device = eigenvalues.device
    energy_grid = torch.linspace(
        energy_range[0], energy_range[1], n_points, device=device
    )

    atom_pdos = {
        atom: torch.zeros(n_points, device=device) for atom in unique_atoms
    }
    orbital_pdos = {
        atom: {
            sh: torch.zeros(n_points, device=device) for sh in shell_names
        }
        for atom in unique_atoms
    }

    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    _, n_kpoints, n_bands = eigenvalues.shape

    for k in range(n_kpoints):
        for b in range(n_bands):
            eigenval = eigenvalues[0, k, b]
            psi = eigenvectors[0, k, b, :]
            weights = (psi.conj() * psi).real if psi.is_complex() \
                else psi * psi
            diff = energy_grid - eigenval
            gaussian = norm * torch.exp(-0.5 * (diff / sigma) ** 2)

            for atype in unique_atoms:
                idx = atom_orbital_map[atype]
                if idx:
                    atom_pdos[atype] += weights[idx].sum() * gaussian
                for sh in shell_names:
                    sidx = orbital_map[atype][sh]
                    if sidx:
                        orbital_pdos[atype][sh] += weights[sidx].sum() * gaussian

    # Average over k-points
    for atype in unique_atoms:
        atom_pdos[atype] /= n_kpoints
        for sh in shell_names:
            orbital_pdos[atype][sh] /= n_kpoints

    energy_np = energy_grid.detach().cpu().numpy()
    atom_pdos_np = {
        a: p.detach().cpu().numpy() for a, p in atom_pdos.items()
    }
    orbital_pdos_np = {
        a: {sh: p.detach().cpu().numpy() for sh, p in d.items()}
        for a, d in orbital_pdos.items()
    }
    return energy_np, atom_pdos_np, orbital_pdos_np, unique_atoms


def plot_band_dos_plotly(
    eigenvalues,
    xticks,
    xtick_labels,
    dos_energies,
    dos_values,
    atom_pdos,
    orbital_pdos,
    energy_grid,
    unique_atoms,
    energy_range,
    bandgap,
    filename="slakonet_out.html",
):
    """Interactive 4-panel Plotly version of the band+DOS+PDOS figure.

    Saves a self-contained HTML file at `filename`. Returns the figure object
    so callers can further customise or export.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=4,
        shared_yaxes=True,
        column_widths=[0.4, 0.15, 0.2, 0.25],
        subplot_titles=(
            f"(a) Bands (Gap {bandgap:.2f} eV)",
            "(b) Total DOS",
            "(c) Atom PDOS",
            "(d) Orbital PDOS",
        ),
        horizontal_spacing=0.02,
    )

    # (a) Bands
    kx = list(range(eigenvalues.shape[1]))
    for ib in range(eigenvalues.shape[-1]):
        fig.add_trace(
            go.Scatter(
                x=kx, y=eigenvalues[0, :, ib].real,
                mode="lines", line=dict(width=1, color="steelblue"),
                showlegend=False, hoverinfo="y",
            ), row=1, col=1,
        )
    fig.add_hline(y=0.0, line_dash="dash", line_color="black",
                  opacity=0.5, row=1, col=1)

    # (b) Total DOS
    fig.add_trace(
        go.Scatter(
            x=dos_values, y=dos_energies, mode="lines",
            line=dict(color="steelblue", width=1.5),
            showlegend=False, name="Total DOS",
        ), row=1, col=2,
    )

    # (c) Atom PDOS
    atom_palette = ["tab:blue", "tab:orange", "tab:green",
                    "tab:red", "tab:purple"]
    for i, atype in enumerate(unique_atoms):
        fig.add_trace(
            go.Scatter(
                x=atom_pdos[atype], y=energy_grid, mode="lines",
                line=dict(width=1.5), name=atype, legendgroup=atype,
            ), row=1, col=3,
        )

    # (d) Orbital PDOS
    shell_colors = {"s": "#1f77b4", "p": "#d62728",
                    "d": "#2ca02c", "f": "#9467bd"}
    dash_per_atom = ["solid", "dash", "dot", "dashdot"]
    for iat, atype in enumerate(unique_atoms):
        dash = dash_per_atom[iat % len(dash_per_atom)]
        for sh in ["s", "p", "d", "f"]:
            pdos = orbital_pdos[atype][sh]
            if float(pdos.max()) < 1e-6:
                continue
            fig.add_trace(
                go.Scatter(
                    x=pdos, y=energy_grid, mode="lines",
                    line=dict(color=shell_colors[sh], width=1.5, dash=dash),
                    name=f"{atype}-{sh}",
                    legendgroup=f"{atype}-{sh}",
                ), row=1, col=4,
            )

    # axis labels + k-ticks
    fig.update_xaxes(
        tickmode="array", tickvals=xticks, ticktext=xtick_labels,
        row=1, col=1, title_text="k-point",
    )
    fig.update_xaxes(title_text="DOS", row=1, col=2)
    fig.update_xaxes(title_text="Atom PDOS", row=1, col=3)
    fig.update_xaxes(title_text="Orbital PDOS", row=1, col=4)
    fig.update_yaxes(title_text="Energy (eV)", row=1, col=1,
                     range=list(energy_range))
    for c in (2, 3, 4):
        fig.update_yaxes(range=list(energy_range), row=1, col=c)
    fig.update_layout(
        height=480, width=1300,
        template="plotly_white", font=dict(size=12),
        legend=dict(orientation="v", x=1.02, y=1.0),
        margin=dict(l=60, r=60, t=50, b=60),
    )
    fig.write_html(filename, include_plotlyjs="cdn", full_html=True)
    return fig


def plot_band_dos_atoms(
    jid=None,
    atoms=None,
    model=None,
    model_path="slakonet_v0",
    energy_range=(-10, 10),
    filename=None,
    cutoff=10.0,
    plotly_filename=None,
):
    if not model:
        elements_hint = None
        if atoms is not None:
            try:
                elements_hint = set(atoms.elements)
            except AttributeError:
                try:
                    elements_hint = set(atoms.get_chemical_symbols())
                except Exception:
                    elements_hint = None
        model = load_trained_model(model_path, elements=elements_hint)
        model = model.float()
    # print("MODEL PATHHHHH", model_path)
    properties, atoms, kpoints = get_properties(
        jid=jid, model=model, atoms=atoms, cutoff=cutoff
    )
    properties["model"] = model
    info = {}

    if filename is None:
        filename = "slakonet_out.png"
    if jid is not None and filename is None:
        filename = str(jid) + "_slakonet_out.png"

    # Band structure data (assumed eV)
    eigenvalues = (
        properties["eigenvalues"].detach().cpu().numpy()
    )  # * H2E  # [1, nk, nb], eV
    # eigenvalues = properties["calc"].eigenvalue * H2E  # [1, nk, nb], eV

    # Eigenvalues are already referenced to Fermi energy, so fermi_eV = 0 for plotting
    fermi_eV = 0.0  # Already subtracted in the eigenvalues

    formula = atoms.composition.reduced_formula
    bandgap = float(properties["bandgap"].detach().cpu().numpy())
    print(f"Bandgap: {bandgap:.3f} eV")
    cbm = float(properties["cbm"].detach().cpu().numpy())
    print(f"CBM: {cbm:.3f} eV")
    vbm = float(properties["vbm"].detach().cpu().numpy())
    print(f"VBM: {vbm:.3f} eV")
    info["cbm"] = cbm
    info["vbm"] = vbm
    info["kpoints"] = kpoints.to_dict()
    info["atoms"] = atoms.to_dict()

    # Geometry for PDOS
    geometry = properties["geometry"]

    # Compute atom- and orbital-projected DOS in one pass
    energy_grid, atom_pdos, orbital_pdos, unique_atoms = \
        compute_atom_and_orbital_pdos(
            properties, geometry, energy_range=energy_range
        )
    info["orbital_pdos"] = {
        a: {sh: p.tolist() for sh, p in d.items()}
        for a, d in orbital_pdos.items()
    }

    # K-point labels — split discontinuities so band lines break cleanly
    labels = kpoints.labels
    eigenvalues, labels = _split_path_discontinuities(eigenvalues, labels)
    xticks, xtick_labels = _format_kpath_ticks(labels)
    info["xticks"] = xticks
    info["xtick_labels"] = xtick_labels
    # --- Plotting ---
    fig = plt.figure(figsize=(13, 5), layout="constrained")
    gs = fig.add_gridspec(nrows=1, ncols=4, width_ratios=[3, 1, 1.5, 2])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    # Bands: eigenvalues already relative to Fermi (E_F = 0)
    for i in range(eigenvalues.shape[-1]):
        y = eigenvalues[0, :, i].real  # Already referenced to Fermi
        ax1.plot(y, linewidth=0.8)
    info["eigenvalues"] = eigenvalues[0, :, i].real.tolist()
    ax1.axhline(0, linestyle="--", alpha=0.7)
    ax1.set_xlabel("k-point")
    ax1.set_ylabel("Energy (eV)")
    # ax1.set_title(f"{jid} {formula}\nGap: {bandgap:.2f} eV")
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

    # Total DOS: already referenced to Fermi
    dos_energies = (
        properties["dos_energy_grid_tensor"]
        .detach()
        .cpu()
        .numpy()  # Already referenced to Fermi
    )
    info["dos_energies"] = dos_energies.tolist()
    dos_values = properties["dos_values_tensor"].detach().cpu().numpy()
    info["dos_values"] = dos_values.tolist()
    ax2.plot(dos_values, dos_energies, linewidth=1.5)
    ax2.axhline(0, linestyle="--", alpha=0.7)
    ax2.set_xlabel("Total DOS")
    ax2.set_ylim(energy_range)
    ax2.set_title("(b)")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(left=False, labelleft=False)
    # info['atom_pdos']=atom_pdos
    # Atom-projected DOS: already referenced to Fermi
    for atom in unique_atoms:
        ax3.plot(
            atom_pdos[atom],
            energy_grid,
            linewidth=1.3,
            label=atom,  # energy_grid already relative to Fermi
        )
        # ax3.fill_betweenx(energy_grid, 0, atom_pdos[atom], alpha=0.25)

    ax3.axhline(0, linestyle="--", alpha=0.7)
    ax3.set_xlabel("Atom PDOS")
    ax3.set_title("(c)")
    ax3.set_ylim(energy_range)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(left=False, labelleft=False)
    ax3.legend(loc="upper right")

    # Orbital-resolved PDOS per atom type
    shell_colors = {"s": "tab:blue", "p": "tab:red",
                    "d": "tab:green", "f": "tab:purple"}
    linestyles = ["-", "--", ":", "-."]
    for iat, atype in enumerate(unique_atoms):
        ls = linestyles[iat % len(linestyles)]
        for sh in ["s", "p", "d", "f"]:
            pdos = orbital_pdos[atype][sh]
            if pdos.max() < 1e-6:
                continue
            ax4.plot(
                pdos, energy_grid,
                linewidth=1.3, linestyle=ls,
                color=shell_colors[sh],
                label=f"{atype}-{sh}",
            )
    ax4.axhline(0, linestyle="--", alpha=0.7)
    ax4.set_xlabel("Orbital PDOS")
    ax4.set_title("(d)")
    ax4.set_ylim(energy_range)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(left=False, labelleft=False)
    ax4.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(
        filename, dpi=150, bbox_inches="tight"
    )  # Reduced DPI from 300 to 150
    plt.close()
    # plt.show()

    print(f"Gap: {bandgap:.3f} eV (Fermi level at E = 0)")
    # print(f"Atom types: {unique_atoms}")
    # pprint.pprint(info)
    dumpjson(data=info, filename="results.json")

    # Also emit interactive Plotly HTML
    if plotly_filename is None:
        if filename.endswith(".png"):
            plotly_filename = filename[:-4] + ".html"
        else:
            plotly_filename = filename + ".html"
    plotly_fig = None
    try:
        plotly_fig = plot_band_dos_plotly(
            eigenvalues=eigenvalues,
            xticks=xticks,
            xtick_labels=xtick_labels,
            dos_energies=dos_energies,
            dos_values=dos_values,
            atom_pdos=atom_pdos,
            orbital_pdos=orbital_pdos,
            energy_grid=energy_grid,
            unique_atoms=unique_atoms,
            energy_range=energy_range,
            bandgap=bandgap,
            filename=plotly_filename,
        )
        print(f"Plotly HTML saved to {plotly_filename}")
    except ImportError:
        print("plotly not installed; skipping interactive HTML output")

    return fig, properties, atom_pdos, energy_grid, orbital_pdos, plotly_fig


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
    cutoff = float(args.cutoff)

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
    fig, properties, atom_pdos, energy_grid, orbital_pdos, _plotly = plot_band_dos_atoms(
        atoms=atoms,
        model_path=model_path,
        model=model,
        jid=jid,
        energy_range=energy_range,
        filename=output_filename,
        cutoff=cutoff,
    )
    t2 = time.time()
    print("Time(s)", t2 - t1)
