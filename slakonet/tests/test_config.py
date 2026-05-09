"""Config schema and composition tests — no GPU or network required."""
import os
import pytest
from omegaconf import OmegaConf

from slakonet.conf import (
    SlakoNetTrainConfig,
    SlakoNetPredictConfig,
    TrainingConfig,
    OptimizerConfig,
)


def test_train_config_defaults():
    cfg = OmegaConf.structured(SlakoNetTrainConfig)
    assert cfg.training.cutoff == pytest.approx(10.0)
    assert list(cfg.optimizer.kpoints) == [5, 5, 5]
    assert cfg.optimizer.hs_regularization == pytest.approx(1e-10)
    assert cfg.optimizer.rep_regularization == pytest.approx(1e-8)
    assert cfg.optimizer.scheduler_factor == pytest.approx(0.8)
    assert cfg.optimizer.scheduler_patience == 10
    assert cfg.optimizer.random_seed == 42
    assert cfg.model.max_Z == 100
    assert cfg.model.kT == pytest.approx(0.025)


def test_training_config_defaults():
    cfg = OmegaConf.structured(TrainingConfig)
    assert cfg.num_epochs == 100
    assert cfg.learning_rate == pytest.approx(1e-5)
    assert cfg.batch_size is None
    assert cfg.target_property == "bandgap"
    assert cfg.force_weight == pytest.approx(0.0)
    assert cfg.energy_weight == pytest.approx(1.0)
    assert cfg.bandgap_weight == pytest.approx(1.0)


def test_predict_config_energy_range():
    cfg = OmegaConf.structured(SlakoNetPredictConfig)
    er = list(cfg.prediction.energy_range)
    assert er == pytest.approx([-8.0, 8.0])
    assert cfg.prediction.file_format == "poscar"
    assert cfg.prediction.cutoff == pytest.approx(10.0)
    assert cfg.prediction.jid == "JVASP-107"


def test_config_merge_override():
    cfg = OmegaConf.structured(SlakoNetTrainConfig)
    override = OmegaConf.create(
        {"training": {"num_epochs": 5, "learning_rate": 0.0001}}
    )
    merged = OmegaConf.merge(cfg, override)
    assert merged.training.num_epochs == 5
    assert merged.training.learning_rate == pytest.approx(0.0001)
    assert merged.optimizer.kpoints == [5, 5, 5]  # unaffected


def test_config_to_yaml_roundtrip():
    cfg = OmegaConf.structured(SlakoNetTrainConfig)
    # Set the required MISSING field before serializing
    cfg.data.xml_folder_path = "/tmp/test"
    yaml_str = OmegaConf.to_yaml(cfg)
    loaded = OmegaConf.create(yaml_str)
    assert loaded.training.num_epochs == cfg.training.num_epochs
    assert loaded.optimizer.scheduler_factor == pytest.approx(
        cfg.optimizer.scheduler_factor
    )


def test_hydra_yaml_compose():
    """Integration test: load real YAML files via hydra.compose."""
    from hydra import initialize_config_dir, compose
    from slakonet.conf import register_configs

    conf_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "conf")
    )
    register_configs()
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["data.xml_folder_path=slakonet/tests"],
        )
    assert cfg.optimizer.scheduler_factor == pytest.approx(0.8)
    assert cfg.optimizer.rep_regularization == pytest.approx(1e-8)
    assert cfg.training.early_stopping_patience == 20
    assert cfg.data.xml_folder_path == "slakonet/tests"
