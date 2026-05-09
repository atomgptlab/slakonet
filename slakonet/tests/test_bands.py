import os
import pytest
import torch
from slakonet.get_bands import get_gap
from slakonet.optim import (
    MultiElementSkfParameterOptimizer,
    get_atoms,
    kpts_to_klines,
)
from slakonet.atoms import Geometry
from slakonet.main import SimpleDftb, generate_shell_dict_upto_Z65

test_dir = os.path.dirname(os.path.abspath(__file__))


def test_basic(default_model_fixture):
    model = default_model_fixture
    jid = "JVASP-1002"

    atoms, opt_gap, mbj_gap = get_atoms(jid)
    geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
    shell_dict = generate_shell_dict_upto_Z65()

    updated_skfs = model.get_updated_skfs()
    h_feed = model._create_comprehensive_feed(updated_skfs, shell_dict, "H")
    s_feed = model._create_comprehensive_feed(updated_skfs, shell_dict, "S")
    nelectron = model._calculate_system_electrons(geometry, updated_skfs)

    kpoints = torch.tensor([3, 3, 3])
    calc = SimpleDftb(
        geometry,
        model,
        kpoints=kpoints,
        device="cpu",
        with_eigenvectors=True,
    )
    calc.calculate()
    energy = calc.energy
    print("energy", energy)
    assert energy is not None


def test_si(default_model_fixture):
    bandgap, opt_gap, mbj_gap, calc, formula = get_gap(
        jid="JVASP-1002", model=default_model_fixture, plot=True
    )
    print("bandgap, opt_gap, mbj_gap, formula", bandgap, opt_gap, mbj_gap, formula)
    assert bandgap is not None


def test_training(minimal_train_cfg, default_model_fixture, tmp_path):
    """Train for 1 epoch on the test vasprun files using the Hydra config."""
    from slakonet.train_slakonet import run_training
    from omegaconf import OmegaConf

    cfg = OmegaConf.merge(
        minimal_train_cfg,
        OmegaConf.create({"training": {"save_directory": str(tmp_path / "out")}}),
    )
    # Use preloaded model to avoid repeated downloads
    import unittest.mock as mock
    with mock.patch("slakonet.train_slakonet.default_model", return_value=default_model_fixture):
        result = run_training(cfg)
    assert result is not None
