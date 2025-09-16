from slakonet.get_bands import get_gap
from slakonet.optim import (
    MultiElementSkfParameterOptimizer,
    get_atoms,
    kpts_to_klines,
    default_model,
)
import os
import glob
from slakonet.atoms import Geometry
from slakonet.main import SimpleDftb, generate_shell_dict_upto_Z65
import torch

test_dir = os.path.dirname(os.path.abspath(__file__))
model = default_model()


def test_basic():
    jid = "JVASP-107"
    jid = "JVASP-1002"

    atoms, opt_gap, mbj_gap = get_atoms(
        jid
    )  # Atoms.from_dict(get_jid_data(jid=jid,dataset='dft_3d')['atoms'])
    # atoms=Atoms.from_poscar("tests/POSCAR")
    # atoms=Atoms.from_poscar("tests/POSCAR-SiC.vasp")
    geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
    # Generate shell dictionary
    shell_dict = generate_shell_dict_upto_Z65()

    updated_skfs = model.get_updated_skfs()

    # Create comprehensive HS feeds that include ALL element pairs
    h_feed = model._create_comprehensive_feed(updated_skfs, shell_dict, "H")
    s_feed = model._create_comprehensive_feed(updated_skfs, shell_dict, "S")

    # Calculate total electron count for the system
    nelectron = model._calculate_system_electrons(geometry, updated_skfs)

    kpoints = torch.tensor([5, 5, 5])
    device = "cpu"
    with_eigenvectors = True
    calc = SimpleDftb(
        geometry,
        shell_dict=shell_dict,
        kpoints=kpoints,
        h_feed=h_feed,
        s_feed=s_feed,
        nelectron=nelectron,
        device=device,
        with_eigenvectors=with_eigenvectors,
    )

    info_ev = calc.calculate_ev_curve(
        method="polynomial",
    )
    eigenvalues = calc()
    energy = calc._calculate_electronic_energy()
    forces = calc._compute_forces_finite_diff()
    # freqs,ds = calc.calculate_phonon_modes()
    print("energy", energy)
    print("forces", forces)
    print("eigenvalues", eigenvalues)
    print("ev info", info_ev)
    import sys

    # sys.exit()
    """
    properties, success = model.compute_multi_element_properties(
        geometry=geometry,
        shell_dict=shell_dict,
        kpoints=kpoints,
        get_fermi=True,
        get_bulk_mod=True,
        get_forces=True,
        device=device,
        with_eigenvectors=True,
    )
    """


def test_si():
    # Find the test file directory
    bandgap, opt_gap, mbj_gap, calc, formula = get_gap(
        jid="JVASP-1002", model=model, plot=True
    )
    print(
        "bandgap, opt_gap, mbj_gap, calc, formula",
        bandgap,
        opt_gap,
        mbj_gap,
        calc,
        formula,
    )


def test_training():
    vasprun_files = [
        os.path.join(test_dir, "vasprun-107.xml"),
        os.path.join(test_dir, "vasprun-1002.xml"),
        os.path.join(test_dir, "vasprun-1002.xml"),
        os.path.join(test_dir, "vasprun-1002.xml"),
    ]
    # vasprun_files = []
    # for i in glob.glob("vasprun/*.xml"):
    #    vasprun_files.append(i)
    print("vasprun_files", vasprun_files)
    multi_vasp_training(vasprun_files, model=model, batch_size=2)


# test_basic()
# test_si()
# test_training()
