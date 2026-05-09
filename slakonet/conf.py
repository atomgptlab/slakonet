from dataclasses import dataclass, field
from typing import Optional, List, Any
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class EigsolverConfig:
    """Base config for all eigensolver variants. Not used directly."""
    solver_name: str = MISSING
    eps: float = 1e-8


@dataclass
class CholeskyEighConfig(EigsolverConfig):
    """Generalized eigenproblem via Cholesky whitening + torch.linalg.eigh."""
    solver_name: str = "cholesky_eigh"


@dataclass
class LanczosConfig(EigsolverConfig):
    """Lanczos algorithm: finds the m lowest eigenvalues via a Krylov subspace."""
    solver_name: str = "lanczos"
    m: int = 20
    tol: float = 1e-6
    max_iter: int = 300
    reorthogonalization: bool = True
    reorthogonalization_type: str = "modified_gram_schmidt"


@dataclass
class VqeConfig(EigsolverConfig):
    """Variational Quantum Eigensolver — stub pending paper implementation."""
    solver_name: str = "vqe"
    n_layers: int = 2
    n_shots: int = 1000


@dataclass
class DataConfig:
    xml_folder_path: str = MISSING
    initial_model_path: str = ""


@dataclass
class TrainingConfig:
    num_epochs: int = 100
    learning_rate: float = 1e-5
    batch_size: Optional[int] = None
    save_directory: str = "out"
    plot_frequency: int = 5
    weight_by_system_size: bool = True
    early_stopping_patience: int = 20
    target_property: str = "bandgap"
    force_weight: float = 0.0
    energy_weight: float = 1.0
    bandgap_weight: float = 1.0
    cutoff: float = 10.0


@dataclass
class OptimizerConfig:
    kpoints: List[int] = field(default_factory=lambda: [5, 5, 5])
    scheduler_factor: float = 0.8
    scheduler_patience: int = 10
    hs_regularization: float = 1e-10
    rep_regularization: float = 1e-8
    random_seed: int = 42


@dataclass
class ModelConfig:
    max_Z: int = 100
    kT: float = 0.025
    alpha: float = 0.1
    beta: float = 0.1
    use_float32: bool = True


@dataclass
class PredictionConfig:
    model_path: Optional[str] = None
    file_format: str = "poscar"
    file_path: Optional[str] = None
    output_filename: str = "slakonet_bands_dos.png"
    energy_range: List[float] = field(default_factory=lambda: [-8.0, 8.0])
    jid: str = "JVASP-107"
    cutoff: float = 10.0


@dataclass
class SlakoNetTrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eigsolver: Any = field(default_factory=CholeskyEighConfig)


@dataclass
class SlakoNetPredictConfig:
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eigsolver: Any = field(default_factory=CholeskyEighConfig)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="train_config", node=SlakoNetTrainConfig)
    cs.store(name="predict_config", node=SlakoNetPredictConfig)
    cs.store(group="data", name="default", node=DataConfig)
    cs.store(group="training", name="default", node=TrainingConfig)
    cs.store(group="optimizer", name="default", node=OptimizerConfig)
    cs.store(group="model", name="default", node=ModelConfig)
    cs.store(group="prediction", name="default", node=PredictionConfig)
    cs.store(group="eigsolver", name="cholesky_eigh", node=CholeskyEighConfig)
    cs.store(group="eigsolver", name="lanczos", node=LanczosConfig)
    cs.store(group="eigsolver", name="vqe", node=VqeConfig)
