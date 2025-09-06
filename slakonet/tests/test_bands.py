from slakonet.get_bands import get_gap
from slakonet.optim import (
    MultiElementSkfParameterOptimizer,
    example_multi_vasp_training,
)
import os
import glob


test_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(test_dir, "slakonet_v1_sic")

# Debug: print the path being used
print(f"Test directory: {test_dir}")
print(f"Model path: {model_path}")
print(f"Model path exists: {os.path.exists(model_path)}")


model = MultiElementSkfParameterOptimizer.load_model(
    model_path, method="state_dict"
)


def test_si():
    # Find the test file directory
    get_gap(jid="JVASP-1002", model=model, plot=True)


def test_training():
    vasprun_files = [
        os.path.join(test_dir, "vasprun-1002.xml"),
        os.path.join(test_dir, "vasprun-107.xml"),
    ]
    print("vasprun_files", vasprun_files)
    example_multi_vasp_training(vasprun_files, model=model)


# test_si()
# test_training()
