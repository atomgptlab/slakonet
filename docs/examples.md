# Examples & Colab

The fastest way to try SlaKoNet — **no installation required** — is the
interactive Colab notebook. Worked example scripts that ship with the
repository are listed further down.

## :material-rocket-launch: Run it in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/slakonet_example.ipynb)

| Notebook | Open | What it covers |
| --- | --- | --- |
| **SlaKoNet — getting started** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/slakonet_example.ipynb) | Loading the model, computing a band structure and DOS, extracting the band gap — entirely in the browser. |

!!! tip "First cell"
    The notebook installs SlaKoNet with `pip install slakonet` and then
    downloads the trained model on first use. Give the first cell a
    minute.

## Example scripts in the repository

The [`slakonet/examples/`](https://github.com/atomgptlab/slakonet/tree/main/slakonet/examples)
directory contains runnable, self-contained scripts. Highlights:

### Using the model

| Script | What it does |
| --- | --- |
| `slakonet_calculator_example.py` | End-to-end `SlaKoNetCalculator` demo — energy, forces, stress, band structure, DOS; reuses one loaded model across structures. |
| `chipstb_bandgaps.py` | High-throughput band-gap benchmark over a JARVIS-DFT material set; compares two k-sampling schemes and computes formation energies. |
| `mgb2_fermi_bands.py` | MgB₂ worked example: band structure + DOS, 3D band structure, and 2D / 3D Fermi surfaces. |
| `predict_bands_from_poscar.py` | Predict a band structure directly from a POSCAR file. |

### Validation & correctness

| Script | What it does |
| --- | --- |
| `check_autograd_forces.py` | Verifies forces match a finite-difference of the energy. |
| `check_stress.py` | Verifies stress against numerical strain. |

To run any of them:

```bash
git clone https://github.com/atomgptlab/slakonet.git
cd slakonet/slakonet/examples
python slakonet_calculator_example.py
```

A full annotated index lives in
[`slakonet/examples/README.md`](https://github.com/atomgptlab/slakonet/blob/main/slakonet/examples/README.md).

## Minimal copy-paste examples

### Band gap of a single material

```python
from ase.build import bulk
from slakonet.optim import default_model
from slakonet.ase_calc import SlaKoNetCalculator

calc = SlaKoNetCalculator(default_model().float())
si = bulk("Si", "diamond", a=5.43); si.calc = calc
print("Si gap:", calc.get_bandgap(), "eV")
```

### Band structure to a PNG

```python
calc.band_structure(si, path="GXWKGL", npoints=20,
                    savefig="si_bands.png")
```

### Screen many materials with one loop

```python
model = default_model().float()
calc = SlaKoNetCalculator(model, kpoints=(3, 3, 3),
                          compute_forces=False)   # fast path

for atoms in my_structures:          # any iterable of ASE Atoms
    atoms.calc = calc
    atoms.get_potential_energy()
    print(atoms.get_chemical_formula(), calc.get_bandgap())
```

See [Band-gap screening](guide/bandgap-screening.md) for a complete,
benchmarked screening tutorial.
