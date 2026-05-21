# Installation

SlaKoNet is a Python package built on PyTorch. It runs on CPU and, for
larger or high-throughput workloads, on NVIDIA GPUs.

## Requirements

- Python **3.10+**
- PyTorch (CPU or CUDA build)
- A few scientific-Python packages (NumPy, SciPy, ASE, jarvis-tools) —
  installed automatically as dependencies.

## Install from PyPI

The simplest route:

```bash
pip install slakonet
```

This pulls in everything needed to load the trained model and run
calculations.

## Install from source

For the latest development version, or to modify the code:

```bash
git clone https://github.com/atomgptlab/slakonet.git
cd slakonet
pip install -e .
```

The `-e` (editable) flag means changes to the source take effect without
reinstalling.

## Recommended: a clean environment

A dedicated conda environment avoids dependency clashes:

```bash
conda create --name slakonet python=3.10 -y
conda activate slakonet
pip install slakonet
```

## GPU support

SlaKoNet uses whatever PyTorch build is installed. For GPU acceleration,
install a CUDA-enabled PyTorch **before** installing SlaKoNet, following
the [official PyTorch instructions](https://pytorch.org/get-started/locally/)
for your CUDA version. SlaKoNet then automatically uses the GPU when one
is available; you can always force a device explicitly:

```python
from slakonet.ase_calc import SlaKoNetCalculator
calc = SlaKoNetCalculator(model, device="cpu")   # or "cuda"
```

## Verify the installation

```python
from slakonet.optim import default_model

model = default_model()       # downloads & caches the trained model
print("SlaKoNet model ready")
```

The first call downloads the trained model and caches it locally
(subsequent calls are instant). If this prints without error, you are
ready — head to the [Quickstart](getting-started.md).

!!! tip "First run is slower"
    The very first `default_model()` call fetches the model weights and
    caches them under `~/.cache`. Every later call simply loads the cache.
