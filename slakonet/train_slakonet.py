from slakonet.get_bands import get_gap
from slakonet.optim import (
    MultiElementSkfParameterOptimizer,
    train_multi_vasp_skf_parameters,
    default_model,
)
import os
import glob
import argparse
import sys
from pydantic_settings import BaseSettings
from jarvis.db.jsonutils import loadjson


class SlakoNetConfig(BaseSettings):
    # Required paths
    initial_model_path: str = ""
    xml_folder_path: str = ""

    # Training parameters
    num_epochs: int = 3
    batch_size: int = None
    save_directory: str = "out"
    learning_rate: float = 0.001
    plot_frequency: int = 5
    weight_by_system_size: bool = True
    early_stopping_patience: int = 20

    # Loss configuration
    target_property: str = (
        "bandgap"  # "energy", "forces", "bandgap", or "both"
    )
    # target_property: str = "bandgap"  # "energy", "forces", "bandgap", or "both"
    force_weight: float = 0.0
    energy_weight: float = 1.0
    bandgap_weight: float = 1.0


H2E = 27.211

parser = argparse.ArgumentParser(description="SlakoNet Train Model")
parser.add_argument(
    "--config_name",
    default="slakonet/examples/config_example.json",
    help="Path of the config file",
)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    config_dict = loadjson(args.config_name)
    config = SlakoNetConfig(**config_dict)

    # Collect XML files
    xml_folder_path = config.xml_folder_path
    xml_files = []
    for i in os.listdir(xml_folder_path):
        if ".xml" in i:
            pth = os.path.join(config.xml_folder_path, i)
            xml_files.append(pth)

    print(f"Found {len(xml_files)} XML files in {xml_folder_path}")

    # Load model
    model_path = config.initial_model_path
    if model_path == "":
        print("Loading default model...")
        model = default_model()
    else:
        print(f"Loading model from {model_path}...")
        model = MultiElementSkfParameterOptimizer.load_model(
            model_path,
            method="compact",
        )

    # Print training configuration
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model: {model_path if model_path else 'default'}")
    print(f"XML files: {len(xml_files)}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size or 'all'}")
    print(f"Target property: {config.target_property}")
    print(
        f"Weights: energy={config.energy_weight}, bandgap={config.bandgap_weight}, force={config.force_weight}"
    )
    print(f"Save directory: {config.save_directory}")
    print("=" * 70 + "\n")

    # Train model
    trained_optimizer, history, data_loader = train_multi_vasp_skf_parameters(
        multi_element_optimizer=model,
        vasprun_paths=xml_files,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        plot_frequency=config.plot_frequency,
        save_directory=config.save_directory,
        weight_by_system_size=config.weight_by_system_size,
        early_stopping_patience=config.early_stopping_patience,
        target_property=config.target_property,
        force_weight=config.force_weight,
        energy_weight=config.energy_weight,
        bandgap_weight=config.bandgap_weight,
    )

    print("\n✅ Training completed successfully!")
