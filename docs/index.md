---
hide:
  - navigation
---

# SlaKoNet

**Differentiable Slater-Koster tight binding for fast, accurate electronic
structure across the periodic table.**

[![PyPI](https://img.shields.io/pypi/v/slakonet.svg)](https://pypi.org/project/slakonet/)
[![Downloads](https://static.pepy.tech/badge/slakonet)](https://pepy.tech/project/slakonet)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/atomgptlab/slakonet/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/atomgptlab/slakonet?style=social)](https://github.com/atomgptlab/slakonet)

[Open in Colab :material-rocket-launch:](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/slakonet_example.ipynb){ .md-button .md-button--primary }
[Get started :material-book-open-variant:](getting-started.md){ .md-button }

---

## What is SlaKoNet?

SlaKoNet is a **parameter-optimization framework** that learns the
Slater-Koster Hamiltonian matrix elements of tight-binding theory **across
65 elements** of the periodic table using **automatic differentiation**.
The parameters are optimized against density-functional-theory band
structures from the JARVIS-DFT database (Tran-Blaha modified
Becke-Johnson level), spanning more than 20,000 materials.

The result combines the **speed and physical interpretability** of
tight-binding with **accuracy approaching hybrid-functional DFT** — band
gaps reach a mean absolute error of **0.74 eV against experiment**,
better than standard GGA functionals (1.14 eV).

![SlaKoNet schematic](https://raw.githubusercontent.com/atomgptlab/slakonet/main/slakonet/examples/sk_schematic.png)

## Why SlaKoNet?

Traditional Slater-Koster tight binding suffers from limited
transferability, painstaking manual parameterization, and training on
low-fidelity data. Machine-learning surrogates, on the other hand, often
fail to produce *detailed* electronic structure (bands, DOS, Fermi
surfaces). SlaKoNet addresses both:

<div class="grid cards" markdown>

-   :material-atom: **Universal**

    One model covers 65 elements and their combinations — no per-system
    fitting.

-   :material-flash: **Fast**

    Quantum-level band structures in seconds; GPU-accelerated for
    high-throughput screening.

-   :material-chart-line: **Detailed**

    Band structures, density of states, band gaps, Fermi surfaces,
    orbital projections — not just a single scalar.

-   :material-function-variant: **Differentiable**

    Built on PyTorch autograd end-to-end — energies, forces and stresses
    flow naturally; the model itself is trainable.

</div>

## A 30-second taste

```python
from ase.build import bulk
from slakonet.optim import default_model
from slakonet.ase_calc import SlaKoNetCalculator

# load the trained model once, reuse it for everything
model = default_model().float()
calc = SlaKoNetCalculator(model, kpoints=(3, 3, 3))

si = bulk("Si", "diamond", a=5.43)
si.calc = calc

print(si.get_potential_energy())   # eV
print(calc.get_bandgap())          # eV
print(calc.get_fermi_level())      # eV
```

Want a band structure or a Fermi surface? See the
[user guide](guide/band-structure.md).

## Highlights

| | |
|---|---|
| **Elements** | 65 (H through the lanthanides) |
| **Band-gap MAE vs experiment** | 0.74 eV |
| **Properties** | bands, DOS, gaps, Fermi surfaces, energies, forces, stress |
| **Interface** | native ASE `Calculator` |
| **Backend** | PyTorch (CPU & GPU) |

## Get going

<div class="grid cards" markdown>

-   [:material-download: **Install**](installation.md) — `pip install slakonet`
-   [:material-rocket: **Quickstart**](getting-started.md) — your first calculation
-   [:material-flask: **Examples & Colab**](examples.md) — ready-to-run notebooks
-   [:material-cog: **How it works**](methodology.md) — the method behind it

</div>

## Citation

If SlaKoNet helps your research, please cite the project. See
[the repository](https://github.com/atomgptlab/slakonet) for the current
reference.
