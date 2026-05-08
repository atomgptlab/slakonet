import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from slakonet.conf import register_configs
from slakonet.optim import (
    MultiElementSkfParameterOptimizer,
    train_multi_vasp_skf_parameters,
    default_model,
    set_random_seed,
)

H2E = 27.211

register_configs()


def run_training(cfg: DictConfig):
    """Execute training from a DictConfig. Callable without the Hydra runtime."""
    set_random_seed(cfg.optimizer.random_seed)

    xml_folder = to_absolute_path(cfg.data.xml_folder_path)
    xml_files = [
        os.path.join(xml_folder, f)
        for f in os.listdir(xml_folder)
        if f.endswith(".xml")
    ]
    print(f"Found {len(xml_files)} XML files in {xml_folder}")

    model_path = cfg.data.initial_model_path
    if not model_path:
        print("Loading default model...")
        model = default_model()
    else:
        print(f"Loading model from {model_path}...")
        model = MultiElementSkfParameterOptimizer.load_model(
            to_absolute_path(model_path),
            method="compact",
        )

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70 + "\n")

    trained_optimizer, history, data_loader = train_multi_vasp_skf_parameters(
        multi_element_optimizer=model,
        vasprun_paths=xml_files,
        num_epochs=cfg.training.num_epochs,
        learning_rate=cfg.training.learning_rate,
        batch_size=cfg.training.batch_size,
        plot_frequency=cfg.training.plot_frequency,
        save_directory=to_absolute_path(cfg.training.save_directory),
        weight_by_system_size=cfg.training.weight_by_system_size,
        early_stopping_patience=cfg.training.early_stopping_patience,
        target_property=cfg.training.target_property,
        force_weight=cfg.training.force_weight,
        energy_weight=cfg.training.energy_weight,
        bandgap_weight=cfg.training.bandgap_weight,
        cutoff=cfg.training.cutoff,
        kpoints=list(cfg.optimizer.kpoints),
        scheduler_factor=cfg.optimizer.scheduler_factor,
        scheduler_patience=cfg.optimizer.scheduler_patience,
        hs_regularization=cfg.optimizer.hs_regularization,
        rep_regularization=cfg.optimizer.rep_regularization,
    )
    return trained_optimizer, history, data_loader


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_training(cfg)
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
