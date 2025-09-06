from slakonet.get_bands import get_gap
from slakonet.optim import (
    MultiElementSkfParameterOptimizer,
    example_multi_vasp_training,
)
import os
import glob


model_path = os.path.join(os.path.dirname(__file__), "slakonet_v1_sic")


def test_si():

    model = MultiElementSkfParameterOptimizer.load_model(
        model_path, method="state_dict"
    )
    get_gap(jid="JVASP-1002", model=model, plot=True)


def test_training():
    vasprun_files = [
        os.path.join(os.path.dirname(__file__), "vasprun-1002.xml"),
        os.path.join(os.path.dirname(__file__), "vasprun-107.xml"),
    ]
    example_multi_vasp_training(vasprun_files)


# test_si()
