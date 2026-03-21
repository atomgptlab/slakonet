from slakonet.get_bands import get_gap
from slakonet.optim import (
    MultiElementSkfParameterOptimizer,
    multi_vasp_training,
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
    initial_model_path: str = ""
    xml_folder_path: str = ""
    num_epochs: int = 3
    batch_size: int = None
    save_directory: str = "out"
    learning_rate: float = 0.001
    plot_frequency: int = 5
    weight_by_system_size: bool = True
    early_stopping_patience: int = 20
    pairwise_cutoff_length: float = 7


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
    xml_folder_path = config.xml_folder_path
    xml_files = []
    for i in os.listdir(xml_folder_path):
        if ".xml" in i:
            pth = os.path.join(config.xml_folder_path, i)
            xml_files.append(pth)

    model_path = config.initial_model_path
    if model_path == "":
        model = default_model()
    else:
        model = MultiElementSkfParameterOptimizer.load_model(
            model_path, method="state_dict"
        )

    trained_optimizer, history, data_loader = train_multi_vasp_skf_parameters(
        multi_element_optimizer=model,
        vasprun_paths=xml_files,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,  # Use all datasets each epoch
        plot_frequency=config.plot_frequency,
        save_directory=config.save_directory,
        weight_by_system_size=config.weight_by_system_size,
        early_stopping_patience=config.early_stopping_patience,
        pairwise_cutoff_length=config.pairwise_cutoff_length
    )
