import matplotlib
from jarvis.db.figshare import get_jid_data
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
from jarvis.core.kpoints import Kpoints3D as Kpoints
from slakonet.atoms import Geometry
import torch
import numpy as np
from slakonet.optim import (
    MultiElementSkfParameterOptimizer,
    get_atoms,
    kpts_to_klines,
    default_model,
)
import torch
import numpy as np
from pathlib import Path
from jarvis.core.atoms import Atoms
from slakonet.main import SimpleDftb, generate_shell_dict_upto_Z65
from slakonet.atoms import Geometry
from jarvis.core.specie import atomic_numbers_to_symbols
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_trained_model(model_path, method="state_dict"):
    """
    Load a trained model

    Args:
        model_path: Path to the saved model
        method: 'state_dict' or 'universal_params'

    Returns:
        Loaded model
    """
    if method == "state_dict":
        model = MultiElementSkfParameterOptimizer.load_model(
            model_path, method="state_dict"
        )
    """
    elif method == 'universal_params':
        # You need to provide the original SKF directory for this method
        skf_directory = "tests/"  # Replace with your SKF directory
        model = MultiElementSkfParameterOptimizer.load_model(
            model_path,
            method='universal_params',
            skf_directory=skf_directory
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    """
    # Set model to evaluation mode
    model.eval()
    return model


def get_gap(
    jid="JVASP-1002", model=None, plot=False, default_points=2, line_density=20
):
    if model is None:
        model = default_model()
    # jid='JVASP-14636'
    atoms, opt_gap, mbj_gap = get_atoms(
        jid
    )  # Atoms.from_dict(get_jid_data(jid=jid,dataset='dft_3d')['atoms'])
    # atoms=Atoms.from_poscar("tests/POSCAR")
    # atoms=Atoms.from_poscar("tests/POSCAR-SiC.vasp")
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
    # print(klines)
    # kpoints = {
    # "\\Gamma": np.array([0.0, 0.0, 0.0]),
    # "K": np.array([3.0 / 8.0, 3.0 / 8.0, 3.0 / 4.0]),
    # "L": np.array([0.5, 0.5, 0.5]),
    # "U": np.array([5.0 / 8.0, 1.0 / 4.0, 5.0 / 8.0]),
    # "W": np.array([0.5, 1.0 / 4.0, 3.0 / 4.0]),
    # "X": np.array([0.5, 0.0, 0.5]),
    #  }
    """
    klines2=torch.tensor(
            [
                [0.0, 0.0, 0.0, 3.0 / 8.0, 3.0 / 8.0, 3.0 / 4.0, 10],
                [3.0 / 8.0, 3.0 / 8.0, 3.0 / 4.0, 0.5, 0.5, 0.5, 10],
                [0.5, 0.5, 0.5,5.0 / 8.0, 1.0 / 4.0, 5.0 / 8.0,10],
                [5.0 / 8.0, 1.0 / 4.0, 5.0 / 8.0,0.5, 1.0 / 4.0, 3.0 / 4.0,10],
                [0.5, 1.0 / 4.0, 3.0 / 4.0,0.5, 0.0, 0.5,10]]
        
        )
    """
    # print("device1",model.device)
    # Calculate properties using the trained model
    with torch.no_grad():  # No gradients needed for inference
        properties, success = model.compute_multi_element_properties(
            geometry=geometry,
            shell_dict=shell_dict,
            klines=klines,
            get_fermi=True,
            device=device,
        )

    if not success:
        raise RuntimeError("Failed to compute properties")

    # Extract energy information
    fermi_energy = properties.get("fermi_energy_eV", None)
    total_energy = properties.get("total_energy_eV", None)
    eigenvalues = properties.get("eigenvalues", None)
    calc = properties["calc"]
    # bandgap = properties["bandgap"].squeeze().detach().numpy().tolist()
    # NEW:
    # print("properties",properties)
    bandgap = properties["bandgap"].squeeze().detach().cpu().numpy().tolist()
    H2E = 27.21
    # print("eigenvalues",eigenvalues)
    if plot:

        plt.figure(figsize=(8, 6))
        plt.rcParams.update({"font.size": 22})
        efermi = properties["efermi"].squeeze().detach().cpu().numpy().tolist()
        for i in range(eigenvalues.shape[-1]):  # Plot each band
            plt.plot(
                eigenvalues[0, :, i].real.detach().cpu().numpy() * H2E
                - efermi,
                c="blue",
            )

        # props = calc.get_properties_dict()

        # fermi = props["fermi_energy_eV"]/H2E #.item()
        plt.xlabel("k-point")
        plt.axhline(y=0, linestyle="-.")
        gap = (
            jid
            + "_"
            + atoms.composition.reduced_formula
            + " Bandgap: "
            + str(round(bandgap, 2))
        )
        # plt.title(gap)
        plt.xticks(xticks, xtick_labels)
        plt.ylabel("Energy (eV)")
        plt.xlim([0, calc.kpoints.shape[1]])
        # plt.title("Band Structure")
        plt.tight_layout()
        plt.savefig("bands_slako.png")
        plt.close()
    return bandgap, opt_gap, mbj_gap, calc, formula


if __name__ == "__main__":
    model_path = "slakonet_v1_sic"
    model_path = "slakonet_v1"
    model = MultiElementSkfParameterOptimizer.load_model(
        model_path, method="state_dict"
    )
    bandgap, opt_gap, mbj_gap, calc, formula = get_gap(
        jid="JVASP-75464", model=model, plot=True
    )
    print(
        "bandgap, opt_gap, mbj_gap, calc, formula",
        bandgap,
        opt_gap,
        mbj_gap,
        calc,
        formula,
    )
    # get_gap(jid="JVASP-1002", model=model, plot=True)
