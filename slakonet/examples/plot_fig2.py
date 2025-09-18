import matplotlib, requests
import matplotlib.pyplot as plt
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

# the_grid = GridSpec(2, 2)
plt.rcParams.update({"font.size": 18})
plt.figure(figsize=(10, 8))
E_low = -2
E_high = 2

jid = "JVASP-1002"
# jid = "JVASP-107"
model_best = default_model()


def get_max_diff(tb3_bands, tb3_fermi, vasprun, energy_tol=4):
    # Get VASP bands relative to Fermi level
    vasp_eigs = (
        np.array([eig[:, 0] for eig in vasprun.eigenvalues[0]]).T
        - vasprun.efermi
    )
    vasp_eigs = vasp_eigs.T  # (n_kpoints, n_bands)

    # Adjust TB3 bands
    tb3_adj = tb3_bands - tb3_fermi

    differences = []
    x = []
    y = []
    for k in range(min(len(tb3_adj), len(vasp_eigs))):
        tb3_valid = tb3_adj[k][
            (tb3_adj[k] > -energy_tol) & (tb3_adj[k] < energy_tol)
        ]
        vasp_valid = vasp_eigs[k][
            (vasp_eigs[k] > -energy_tol) & (vasp_eigs[k] < energy_tol)
        ]

        for tb3_e in tb3_valid:
            if len(vasp_valid) > 0:
                differences.append(np.min(np.abs(tb3_e - vasp_valid)))
                x.append(np.min(np.abs(tb3_e - vasp_valid)))
                y.append(k)
    max_diff = max(differences) if differences else 0.0
    return max_diff, x, y  # max(differences) if differences else 0.0


def filter_close_labels(points, labels, min_distance=5):
    """
    Filter out labels that are too close to each other
    """
    if len(points) == 0:
        return points, labels

    filtered_points = [points[0]]
    filtered_labels = [labels[0]]

    for i in range(1, len(points)):
        # Check if current point is far enough from the last added point
        if points[i] - filtered_points[-1] >= min_distance:
            filtered_points.append(points[i])
            filtered_labels.append(labels[i])

    return np.array(filtered_points), filtered_labels


dat = get_jid_data(jid=jid, dataset="dft_3d")
url = None
for j in dat["raw_files"]:
    if "OPT-Bandst" in j:
        url = j.split(",")[2]


if url is None:
    print("Check URL")

atoms, opt_gap, mbj_gap = get_atoms(jid)


default_points = 2
line_density = 20
geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
# Generate shell dictionary
shell_dict = generate_shell_dict_upto_Z65()
kpoints = Kpoints().kpath(atoms, line_density=line_density)
labels = kpoints.labels
xticks = []
xtick_labels = []
kps = []
for ii, i in enumerate(labels):
    kps.append(kpoints.kpts[ii])
    lbl = "$" + i + "$"
    # lbl=lbl.replace("\\G","\G")
    if ii == 0 and lbl != "$$":
        xticks.append(ii * int(default_points / 2))
        xtick_labels.append(lbl)

    if lbl != "$$" and labels[ii] != labels[ii - 1]:
        xticks.append(ii * int(default_points / 2))
        xtick_labels.append(lbl)
        # kps.append(kpoints.kpts[ii])

# print(xtick_labels)
formula = atoms.composition.reduced_formula
klines = kpts_to_klines(kpoints.kpts, default_points=default_points)


# Calculate properties using the trained model
with torch.no_grad():  # No gradients needed for inference
    properties, success = model_best.compute_multi_element_properties(
        geometry=geometry, shell_dict=shell_dict, klines=klines, get_fermi=True
    )

if not success:
    raise RuntimeError("Failed to compute properties")

# Extract energy information
fermi_energy = properties.get("fermi_energy_eV", None)
total_energy = properties.get("total_energy_eV", None)
eigenvalues = properties.get("eigenvalues", None)
calc = properties["calc"]
bandgap = properties["bandgap"].squeeze().detach().cpu().numpy().tolist()

H2E = 27.21
plot = True

figshare_url = "https://ndownloader.figshare.com/files/23713301"
with tempfile.TemporaryDirectory() as temp_dir:
    # Download the ZIP file
    print("Downloading ZIP file...")
    response = requests.get(figshare_url)
    response.raise_for_status()

    zip_path = os.path.join(temp_dir, "vasp_data.zip")
    with open(zip_path, "wb") as f:
        f.write(response.content)

    # Extract ZIP file
    print("Extracting ZIP file...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    # Find vasprun.xml and KPOINTS files
    vasprun_path = None
    kpoints_path = None

    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file == "vasprun.xml":
                vasprun_path = os.path.join(root, file)
                vrun_bands = Vasprun(vasprun_path)
            elif file == "KPOINTS":
                kpoints_file_path = os.path.join(root, file)
                kp_labels = []
                kp_labels_points = []
                f = open(kpoints_file_path, "r")
                lines = f.read().splitlines()
                f.close()
                for ii, i in enumerate(lines):
                    if ii > 2:
                        tmp = i.split()
                        if len(tmp) == 5:
                            tmp = str("$") + str(tmp[4]) + str("$")
                            if len(kp_labels) == 0:
                                kp_labels.append(tmp)
                                kp_labels_points.append(ii - 3)
                            elif tmp != kp_labels[-1]:
                                kp_labels.append(tmp)
                                kp_labels_points.append(ii - 3)


spin = 0
zero_efermi = True
plot = True

tmp = 0.0
info = {}
indir_gap = float(vrun_bands.get_indir_gap[0])
print("gap=", indir_gap)
info["indir_gap"] = indir_gap

info["efermi"] = float(vrun_bands.efermi)
if zero_efermi:
    tmp = float(vrun_bands.efermi)

spin_up_bands_x = []
spin_up_bands_y = []
spin_down_bands_x = []
spin_down_bands_y = []
for i, ii in enumerate(vrun_bands.eigenvalues[spin][:, :, 0].T - tmp):
    # plt.plot(ii, color="r")
    spin_up_bands_x.append([np.arange(0, len(ii))])
    spin_up_bands_y.append([ii])
if vrun_bands.is_spin_polarized:
    for i, ii in enumerate(vrun_bands.eigenvalues[1][:, :, 0].T - tmp):
        # plt.plot(ii, color="b")
        spin_down_bands_x.append([np.arange(0, len(ii))])
        spin_down_bands_y.append([ii])

info["spin_up_bands_x"] = spin_up_bands_x
info["spin_up_bands_y"] = spin_up_bands_y
info["spin_down_bands_x"] = spin_down_bands_x
info["spin_down_bands_y"] = spin_down_bands_y

info["kp_labels_points"] = list(kp_labels_points)
info["kp_labels"] = list(kp_labels)
count = 0
if plot:
    for i, j in zip(info["spin_up_bands_x"], info["spin_up_bands_y"]):
        if count == 0:
            plt.plot(
                np.array(i).flatten(),
                np.array(j).flatten(),
                color="r",
                label="OPT",
            )
            count += 1
        else:
            plt.plot(np.array(i).flatten(), np.array(j).flatten(), color="r")
    # plt.plot(0,0,'-',label='OPT',color='red')
    # if self.is_spin_polarized:
    #     for i, j in zip(
    #         info["spin_down_bands_x"], info["spin_down_bands_y"]
    #     ):
    #         plt.plot(
    #             np.array(i).flatten(), np.array(j).flatten(), color="r"
    #         )

    plt.ylim([E_low, E_high])
    kp_labels_points, kp_labels = filter_close_labels(
        kp_labels_points, kp_labels
    )
    plt.xticks(kp_labels_points, kp_labels)
    plt.xlim([0, len(vrun_bands.kpoints._kpoints)])
    # plt.legend()
    # plt.xlabel(r"$\mathrm{Wave\ Vector}$")
    ylabel = (
        r"$\mathrm{E\ -\ E_f\ (eV)}$"
        if zero_efermi
        else r"$\mathrm{Energy\ (eV)}$"
    )
    plt.ylabel(ylabel)
# plt.tight_layout()
# Intel Xenon, E5V4, 56 cores, 2 TB
# Savethe server (Media tech)
with torch.no_grad():  # No gradients needed for inference
    properties, success = model_best.compute_multi_element_properties(
        geometry=geometry, shell_dict=shell_dict, klines=klines, get_fermi=True
    )

fermi_energy = properties.get("fermi_energy_eV", None)
total_energy = properties.get("total_energy_eV", None)
eigenvalues = properties.get("eigenvalues", None)
calc = properties["calc"]
bandgap = properties["bandgap"].squeeze().detach().cpu().numpy().tolist()

H2E = 27.21
plot = True
# print("eigenvalues",eigenvalues)
if plot:
    # %matplotlib inline

    # plt.figure(figsize=(8, 6))
    # plt.rcParams.update({'font.size': 22})
    efermi = properties["efermi"].squeeze().detach().cpu().numpy().tolist()
    for i in range(eigenvalues.shape[-1]):  # Plot each band
        if i == 0:
            plt.plot(
                eigenvalues[0, :, i].real.detach().cpu().numpy() * H2E
                - efermi,
                c="b",
                label="SlaKoNet",
            )
        else:
            plt.plot(
                eigenvalues[0, :, i].real.detach().cpu().numpy() * H2E
                - efermi,
                c="b",
            )

    # props = calc.get_properties_dict()

    # fermi = props["fermi_energy_eV"]/H2E #.item()
    # plt.xlabel("k-point")
    plt.legend()
    plt.axhline(y=0, linestyle="-.")
    gap = (
        jid
        + "_"
        + atoms.composition.reduced_formula
        + " Bandgap: "
        + str(round(bandgap, 2))
    )
    # plt.title(gap)
    # plt.xticks(xticks,xtick_labels)
    plt.ylabel("Energy (eV)")
    plt.xlim([0, calc.kpoints.shape[1]])
    # plt.title("Band Structure")
    plt.tight_layout()
plt.savefig("Fig2.png")
plt.close()
# plt.show()

sk_fermi = properties.get("fermi_energy_eV", None)
max_diff, x, y = get_max_diff(
    eigenvalues[0].cpu().numpy(), sk_fermi, vrun_bands
)
print("max_diff", max_diff)
plt.plot(x, y, ".")
plt.savefig("diff.png")
plt.close()
