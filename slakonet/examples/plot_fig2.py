import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from slakonet.atoms import Geometry
from jarvis.core.kpoints import Kpoints3D as Kpoints
from jarvis.db.figshare import get_jid_data
from slakonet.optim import get_atoms
from jarvis.io.vasp.outputs import Vasprun
import tempfile, os, zipfile, torch
from slakonet.main import SimpleDftb, generate_shell_dict_upto_Z65
from slakonet.optim import default_model, kpts_to_klines
import numpy as np
import requests

# Set up the figure with 4x2 subplots
fig = plt.figure(figsize=(20, 28))
gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.1)

plt.rcParams.update({"font.size": 24})
E_low = -4
E_high = 4

# List of materials to analyze
jids = ["JVASP-816", "JVASP-1002", "JVASP-107", "JVASP-1174"]
# model_best = default_model()


def add_descriptive_panel_label(
    ax, label, x=-0.12, y=1.05, fontsize=24, fontweight="bold"
):
    """
    Add descriptive panel label like (a) JVASP-816 - Al
    """
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        va="bottom",
        ha="left",
    )


def get_max_diff_with_scatter(sk_bands, sk_fermi, vasprun, energy_tol=4):
    """
    Fixed version that properly aligns energies and returns correct scatter plot data
    """
    # Get VASP bands relative to Fermi level
    vasp_eigs = (
        np.array([eig[:, 0] for eig in vasprun.eigenvalues[0]]).T
        - vasprun.efermi
    )
    vasp_eigs = vasp_eigs.T  # (n_kpoints, n_bands)

    # Convert SK bands to eV and adjust relative to SK Fermi level
    sk_bands_eV = sk_bands * 27.21  # Convert from Hartree to eV
    sk_adj = sk_bands_eV - sk_fermi

    differences = []
    energy_points = []
    vasp_energies = []

    for k in range(min(len(sk_adj), len(vasp_eigs))):
        sk_valid = sk_adj[k][
            (sk_adj[k] > -energy_tol) & (sk_adj[k] < energy_tol)
        ]
        vasp_valid = vasp_eigs[k][
            (vasp_eigs[k] > -energy_tol) & (vasp_eigs[k] < energy_tol)
        ]

        for sk_e in sk_valid:
            if len(vasp_valid) > 0:
                distances = np.abs(sk_e - vasp_valid)
                min_diff_idx = np.argmin(distances)
                min_diff = distances[min_diff_idx]
                closest_vasp_e = vasp_valid[min_diff_idx]

                # Only include reasonable matches (within 1 eV)
                if min_diff < 5.0:
                    differences.append(min_diff)
                    energy_points.append(
                        closest_vasp_e
                    )  # Use VASP energy as reference
                    vasp_energies.append(closest_vasp_e)

    max_diff = max(differences) if differences else 0.0
    return (
        max_diff,
        np.array(differences),
        np.array(energy_points),
        vasp_energies,
    )


def filter_close_labels(points, labels, min_distance=5):
    """Filter out labels that are too close to each other"""
    if len(points) == 0:
        return points, labels

    filtered_points = [points[0]]
    filtered_labels = [labels[0]]

    for i in range(1, len(points)):
        if points[i] - filtered_points[-1] >= min_distance:
            filtered_points.append(points[i])
            filtered_labels.append(labels[i])

    return np.array(filtered_points), filtered_labels


def process_material(jid, row_idx):
    """Process a single material and create plots with descriptive panel labels"""
    print(f"Processing {jid}...")

    try:
        # Get material data
        dat = get_jid_data(jid=jid, dataset="dft_3d")
        url = None
        for j in dat["raw_files"]:
            if "OPT-Bandst" in j:
                url = j.split(",")[2]
                break

        if url is None:
            print(f"No band structure URL found for {jid}")
            return

        atoms, opt_gap, mbj_gap = get_atoms(jid)

        # Set up k-points
        default_points = 2
        line_density = 20
        geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
        shell_dict = generate_shell_dict_upto_Z65()
        kpoints = Kpoints().kpath(atoms, line_density=line_density)

        # Process k-point labels
        labels = kpoints.labels
        xticks = []
        xtick_labels = []
        for ii, i in enumerate(labels):
            lbl = "$" + i + "$"
            if ii == 0 and lbl != "$$":
                xticks.append(ii * int(default_points / 2))
                xtick_labels.append(lbl)
            elif lbl != "$$" and labels[ii] != labels[ii - 1]:
                xticks.append(ii * int(default_points / 2))
                xtick_labels.append(lbl)

        formula = atoms.composition.reduced_formula
        klines = kpts_to_klines(kpoints.kpts, default_points=default_points)

        # Calculate SlaKoNet properties
        with torch.no_grad():
            properties, success = model_best.compute_multi_element_properties(
                geometry=geometry,
                shell_dict=shell_dict,
                klines=klines,
                get_fermi=True,
            )

        if not success:
            print(f"Failed to compute properties for {jid}")
            return

        eigenvalues = properties.get("eigenvalues", None)
        calc = properties["calc"]
        bandgap = (
            properties["bandgap"].squeeze().detach().cpu().numpy().tolist()
        )
        sk_fermi = properties.get("fermi_energy_eV", None)
        efermi = properties["efermi"].squeeze().detach().cpu().numpy().tolist()

        H2E = 27.21

        # Download and process VASP data
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Downloading VASP data for {jid}...")
            response = requests.get(url)
            response.raise_for_status()

            zip_path = os.path.join(temp_dir, "vasp_data.zip")
            with open(zip_path, "wb") as f:
                f.write(response.content)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # Find vasprun.xml and KPOINTS files
            vasprun_path = None
            kpoints_file_path = None

            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file == "vasprun.xml":
                        vasprun_path = os.path.join(root, file)
                        vrun_bands = Vasprun(vasprun_path)
                    elif file == "KPOINTS":
                        kpoints_file_path = os.path.join(root, file)

            if vasprun_path is None:
                print(f"No vasprun.xml found for {jid}")
                return

            # Process KPOINTS labels
            kp_labels = []
            kp_labels_points = []
            if kpoints_file_path:
                with open(kpoints_file_path, "r") as f:
                    lines = f.read().splitlines()
                for ii, line in enumerate(lines):
                    if ii > 2:
                        tmp = line.split()
                        if len(tmp) == 5:
                            lbl = "$" + str(tmp[4]) + "$"
                            if len(kp_labels) == 0:
                                kp_labels.append(lbl)
                                kp_labels_points.append(ii - 3)
                            elif lbl != kp_labels[-1]:
                                kp_labels.append(lbl)
                                kp_labels_points.append(ii - 3)

            # Get VASP band structure data
            indir_gap = float(vrun_bands.get_indir_gap[0])
            vasp_efermi = float(vrun_bands.efermi)

            # Create descriptive panel labels
            panel_letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
            left_letter = panel_letters[
                row_idx * 2
            ]  # For left column: a, c, e, g
            right_letter = panel_letters[
                row_idx * 2 + 1
            ]  # For right column: b, d, f, h

            left_label = f"({left_letter}) {jid} - {formula}"
            right_label = f"({right_letter}) Energy vs |Difference|"

            # PLOT 1: Band Structure Comparison
            ax1 = fig.add_subplot(gs[row_idx, 0])

            # Add descriptive panel label to band structure plot
            add_descriptive_panel_label(
                ax1, left_label, x=0.2, y=1.02, fontsize=18
            )
            # add_descriptive_panel_label(ax1, left_label, x=-0.12, y=1.02, fontsize=18)

            # Plot VASP bands
            count = 0
            for i, band_data in enumerate(
                vrun_bands.eigenvalues[0][:, :, 0].T - vasp_efermi
            ):
                if count == 0:
                    ax1.plot(
                        np.arange(len(band_data)),
                        band_data,
                        color="r",
                        label="OPT",
                        linewidth=1,
                        alpha=0.8,
                    )
                    count += 1
                else:
                    ax1.plot(
                        np.arange(len(band_data)),
                        band_data,
                        color="r",
                        linewidth=1,
                        alpha=0.8,
                    )

            # Plot SlaKoNet bands
            for i in range(eigenvalues.shape[-1]):
                sk_band = (
                    eigenvalues[0, :, i].real.detach().cpu().numpy() * H2E
                    - efermi
                )
                if i == 0:
                    ax1.plot(
                        sk_band,
                        c="b",
                        label="SlaKoNet",
                        linewidth=1,
                        alpha=0.8,
                    )
                else:
                    ax1.plot(sk_band, c="b", linewidth=1, alpha=0.8)

            ax1.set_ylim([E_low, E_high])
            ax1.axhline(y=0, linestyle="--", color="gray", alpha=0.7)

            # Set x-axis labels
            if kp_labels_points and kp_labels:
                kp_labels_points_filtered, kp_labels_filtered = (
                    filter_close_labels(
                        kp_labels_points, kp_labels, min_distance=10
                    )
                )
                ax1.set_xticks(kp_labels_points_filtered)
                ax1.set_xticklabels(kp_labels_filtered)

            ax1.set_xlim([0, len(vrun_bands.kpoints._kpoints)])
            ax1.set_ylabel("E - E$_f$ (eV)")
            if indir_gap < 0.05:
                indir_gap = 0.0

            # Remove individual subplot titles since we have descriptive panel labels
            # ax1.set_title(f"{jid} - {formula}\nVASP Gap: {indir_gap:.2f} eV, SK Gap: {bandgap:.2f} eV")

            # Add gap information as text annotation instead
            if bandgap < 0.06:
                bandgap = 0.0
            ax1.text(
                0.02,
                0.98,
                f"VASP Gap: {indir_gap:.2f} eV\nSK Gap: {bandgap:.2f} eV",
                transform=ax1.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                fontsize=20,
            )

            if row_idx == 0:  # Only show legend for the first row
                ax1.legend(fontsize=16, loc="upper right")

            # PLOT 2: Max Difference Analysis
            ax2 = fig.add_subplot(gs[row_idx, 1])
            ax2.set_xlim([0, 2])  # Adjusted x-limit for better visualization

            # Add descriptive panel label to difference plot
            add_descriptive_panel_label(
                ax2, right_label, x=0.2, y=1.02, fontsize=18
            )
            # add_descriptive_panel_label(ax2, right_label, x=-0.12, y=1.02, fontsize=18)

            # Get max difference data
            max_diff, differences, energy_points, vvv = (
                get_max_diff_with_scatter(
                    eigenvalues[0].cpu().numpy(), sk_fermi, vrun_bands
                )
            )

            if len(differences) > 0:
                # Create scatter plot using actual energy points
                scatter = ax2.scatter(
                    differences,
                    energy_points,  # Now using actual VASP energies
                    c="black",
                    alpha=0.3,
                    s=8,
                    rasterized=True,
                )
                ax2.set_xlabel("δ = |E$_{VASP}$ - E$_{SK}$| (eV)")
                # ax2.set_ylabel("E - E$_f$ (eV)")
                ax2.set_ylim(E_low, E_high)
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
                ax2.axhline(
                    y=1, color="green", linestyle="-.", linewidth=2, alpha=0.7
                )
                ax2.axhline(
                    y=-1, color="green", linestyle="-.", linewidth=2, alpha=0.7
                )

                # Add statistics text
                mae = np.mean(differences)
                rms = np.sqrt(np.mean(differences**2))
                ax2.text(
                    0.02,
                    0.02,  # Position at bottom left instead of top
                    f"MAE: {mae:.2f} eV\nRMS: {rms:.2f} eV\nMax: {max_diff:.2f} eV",
                    transform=ax2.transAxes,
                    # verticalalignment="bottom",
                    verticalalignment="baseline",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                    fontsize=20,
                )
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "No matching bands found",
                    transform=ax2.transAxes,
                    ha="center",
                    va="center",
                    fontsize=20,
                )

            print(
                f"Completed {jid}: Max diff = {max_diff:.4f} eV, Points = {len(differences)}"
            )

    except Exception as e:
        print(f"Error processing {jid}: {str(e)}")

        # Create error plots with panel labels
        panel_letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
        left_letter = panel_letters[row_idx * 2]
        right_letter = panel_letters[row_idx * 2 + 1]

        left_label = f"({left_letter}) {jid} - Error"
        right_label = f"({right_letter}) Error"

        ax1 = fig.add_subplot(gs[row_idx, 0])
        add_descriptive_panel_label(
            ax1, left_label, x=-0.12, y=1.02, fontsize=20
        )
        ax1.text(
            0.5,
            0.5,
            f"Error processing {jid}",
            transform=ax1.transAxes,
            ha="center",
            va="center",
        )

        ax2 = fig.add_subplot(gs[row_idx, 1])
        add_descriptive_panel_label(
            ax2, right_label, x=-0.12, y=1.02, fontsize=20
        )
        ax2.text(
            0.5,
            0.5,
            f"Error processing {jid}",
            transform=ax2.transAxes,
            ha="center",
            va="center",
        )

    try:
        return max_diff, differences, energy_points, vrun_bands, vvv
    except:
        return None, None, None, None, None


# Process all materials
results = []
for i, jid in enumerate(jids):
    result = process_material(jid, i)
    results.append(result)

# Adjust layout to accommodate panel labels
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)

# Save the figure
plt.savefig(
    "multi_material_analysis_with_labels.png", dpi=300, bbox_inches="tight"
)
plt.savefig("multi_material_analysis_with_labels.pdf", bbox_inches="tight")
plt.show()

print("Analysis complete!")
print("Panel labels:")
print("(a) JVASP-816 - Al, (b) Energy vs |Difference|")
print("(c) JVASP-1002 - Si, (d) Energy vs |Difference|")
print("(e) JVASP-107 - SiC, (f) Energy vs |Difference|")
print("(g) JVASP-1174 - GaAs, (h) Energy vs |Difference|")
