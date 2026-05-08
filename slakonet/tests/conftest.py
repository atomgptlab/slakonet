import pytest
import os
from omegaconf import OmegaConf

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def minimal_train_cfg():
    """Structured train config with test-dir data — no Hydra runtime or network needed."""
    from slakonet.conf import SlakoNetTrainConfig
    cfg = OmegaConf.structured(SlakoNetTrainConfig)
    cfg.data.xml_folder_path = TEST_DIR
    cfg.training.num_epochs = 1
    cfg.training.batch_size = 2
    cfg.training.target_property = "bandgap"
    cfg.optimizer.random_seed = 0
    return cfg


@pytest.fixture
def minimal_predict_cfg():
    """Structured predict config pointing at a known JID — no Hydra runtime needed."""
    from slakonet.conf import SlakoNetPredictConfig
    cfg = OmegaConf.structured(SlakoNetPredictConfig)
    cfg.prediction.jid = "JVASP-1002"
    cfg.prediction.cutoff = 10.0
    return cfg


@pytest.fixture(scope="session")
def default_model_fixture():
    """Session-scoped model download so tests share a single download."""
    from slakonet.optim import default_model
    return default_model()
