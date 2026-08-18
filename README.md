# SlakoNet

SlaKoNet learns Slater-Koster tight-binding Hamiltonian matrix elements
across 65 elements using automatic differentiation, trained on JARVIS-DFT
data with the Tran-Blaha modified Becke-Johnson (TBmBJ) functional
(>20,000 materials). It reaches 0.74 eV MAE for band gaps against
experiment, versus 1.14 eV for standard GGA, while keeping the cost and
interpretability of tight binding.

![SlakoNet schematic](https://github.com/atomgptlab/slakonet/blob/main/slakonet/examples/sk_schematic.png)

## Key Features

- **Universal parameterization**: 65 elements and their combinations
- **Physics-informed**: Slater-Koster tight-binding formalism
- **Accurate**: 0.74 eV MAE for band gaps vs experiment
- **Scalable**: GPU-accelerated, >10,000 atoms with the sparse solver
- **Comprehensive**: band structures, DOS, band gaps, orbital projections
- **ASE-compatible**: energy, forces and stress through a standard calculator

## Installation

```bash
pip install slakonet
```

Or create a conda environment and install SlaKoNet in editable mode. To
do so, first install [miniforge](https://github.com/conda-forge/miniforge):

```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
```

Based on your system requirements, you'll get a file something like
'Miniforge3-XYZ'.

```
bash Miniforge3-$(uname)-$(uname -m).sh
```

Now, make a conda environment:

```
conda create --name slakonet python=3.10 -y
conda activate slakonet
```

```
git clone https://github.com/atomgptlab/slakonet.git
cd slakonet
pip install uv; uv pip install -e .
```

## Quick Start

### Google Colab example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/slakonet_example.ipynb)

### Example of Training Models

```bash
python slakonet/train_slakonet.py --config_name slakonet/examples/config_example.json
```

### Example of Inference

```bash
python slakonet/predict_slakonet.py  --file_path slakonet/examples/POSCAR-JVASP-107.vasp 
```

![SlakoNet output](https://github.com/atomgptlab/slakonet/blob/main/slakonet/examples/slakonet_bands_dos.png)

### Available Parameter Sets

Parameter sets are downloaded from
[Figshare](https://figshare.com/articles/dataset/SlakoNet_parameters/30122215)
on first use and cached under `~/.cache/atomgptlab/slakonet/`.

| Name | Elements | Description |
| --- | --- | --- |
| `slakonet_v0` | 64 | Original universal parameter set (paper v1) |
| `slakonet_v1` | 75 | Second-generation universal parameter set |
| `slakonet_v1a` | 64 | Refined v1 parameter set (default) |
| `slakonet_base75` | 75 | Untrained reference tables; built locally |
| `slakonet_v1a_full` | 75 | `v1a` over the full range; built locally |
| `slakonet_v2` | 75 | `v1a_full` + on-site shifts fitted to ChIPS-TB gaps; built locally |

`slakonet_v1a` is an untrained Slater-Koster set: its H/S are
bit-identical to the reference tables it was built from, and its
repulsive is too apart from 496 pairs that were refit. Nothing was
retrained away from anything else, which is what keeps its equations of
state physical — Si `B0` = 101 GPa against 89 GPa DFT, where
`slakonet_v1`'s retrained H/S give 9261 GPa for Al against 69 GPa DFT.
What `v1a` lacks is reach: it stops at the 64 elements with Z ≤ 65,
while the reference tables cover 75.

`slakonet_v1a_full` closes that gap without giving up `v1a`: untrained
H/S for all 5625 pairs, untouched repulsive everywhere except the 496
pairs `v1a` refit. It reproduces `v1a` energies exactly on shared
chemistry and adds He, Ne, Ar, Kr, Xe, Rn, Po, At, Ra, Th and Lu. Build
it from the reference skf files
([Zenodo](https://zenodo.org/records/14289468) → `complete_set/`):

```bash
python slakonet/examples/build_v1a_extended.py \
    --skf-dir /path/to/complete_set --name slakonet_base75
python slakonet/examples/build_v1a_full.py
```

Without the skf files, `--base slakonet_v1` merges from the cached sets
instead, at the cost of `v1`'s retrained parameters for those elements.

`slakonet_v2` adds per-element on-site energy shifts fitted against
ChIPS-TB band gaps (`slakonet/examples/fit_onsite_gaps.py`). On materials
held out of the fit it cuts gap MAE from 0.63 to 0.41 eV. It does **not**
improve the equation of state — bulk-modulus MAE is flat and the
stiffness bias grows — so use `slakonet_v1a_full` where energetics
matter, until the repulsive is refit to match. Score any set with:

```bash
python slakonet/examples/chipstb_eval.py --model slakonet_v2 --kspacing 0.2
python slakonet/examples/chipstb_compare.py --ours chipstb_eval_slakonet_v2.csv
```

```python
from slakonet.optim import default_model

model = default_model(model_name="slakonet_v1a")
```

`default_model()` with no arguments uses `slakonet_v1a`; set the
`SLAKONET_MODEL` environment variable to change the default globally, and
`--model_path slakonet_v1a` selects a set from the command line:

```bash
SLAKONET_MODEL=slakonet_v1a python slakonet/predict_slakonet.py --jid JVASP-107
python slakonet/predict_slakonet.py --model_path slakonet_v1a --jid JVASP-107
```

### Using Pretrained Models in Python

```python
from slakonet.optim import (
    MultiElementSkfParameterOptimizer,
    get_atoms,
    kpts_to_klines,
    default_model,
)
import torch
from slakonet.atoms import Geometry
from slakonet.main import generate_shell_dict_upto_Z65

model = default_model()

# Get structure (example with JARVIS ID)
atoms, opt_gap, mbj_gap = get_atoms("JVASP-107")  
geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
shell_dict = generate_shell_dict_upto_Z65(model=model)

# Compute electronic properties
with torch.no_grad():
    properties, success = model.compute_multi_element_properties(
        geometry=geometry,
        shell_dict=shell_dict,
        get_fermi=True,
        device="cuda"
    )

# Access results (all tensors; .item() for scalars)
print(f"Band gap: {properties['bandgap'].item():.3f} eV")
print(f"Fermi energy: {properties['fermi_energy'].item():.3f} eV")

# Plot band structure and DOS
eigenvalues = properties["eigenvalues"]
dos_values = properties['dos_values_tensor']
dos_energies = properties['dos_energy_grid_tensor']
```

### ASE Calculator

`SlaKoNetCalculator` exposes SlaKoNet through the standard ASE
`Calculator` API. The trained model is **loaded once** and injected into
the calculator, then reused for every structure and every call (no
per-call reload). Energy, forces and stress use the usual ASE methods;
band structure and DOS are dedicated methods.

```python
from ase.build import bulk
from slakonet.optim import default_model
from slakonet.ase_calc import SlaKoNetCalculator

# load the trained model ONCE
model = default_model().float()

calc = SlaKoNetCalculator(model, kpoints=(3, 3, 3))

si = bulk("Si", "diamond", a=5.43)
si.calc = calc
si.get_potential_energy()        # eV
si.get_forces()                  # eV/Ang, shape (N, 3)
si.get_stress()                  # eV/Ang^3, Voigt(6)

# band structure (-> PNG) and total DOS, same loaded model
bs = calc.band_structure(si, path="GXWKGL", npoints=20,
                         savefig="si_bands.png")
e, dos = calc.dos(si)
print(calc.get_bandgap(), calc.get_fermi_level())

# Hamiltonian and overlap, (n_kpoints, n_orbitals, n_orbitals)
H, S = calc.get_HS(si)

# reuse on another structure with NO model reload
ge = bulk("Ge", "diamond", a=5.66); ge.calc = calc
ge.get_potential_energy()
```

`get_bandstructure()` and `get_dos()` are aliases of `band_structure()`
and `dos()`. The same three accessors exist on
`slakonet.main.SlakoNetCalculator`.

`get_HS` returns the k-resolved Hamiltonian and overlap. **H is in
Hartree** and the basis is non-orthogonal, so band energies come from
the generalized eigenproblem:

```python
import scipy.linalg as sla
from ase.build import bulk
from slakonet.optim import default_model
from slakonet.ase_calc import SlaKoNetCalculator

calc = SlaKoNetCalculator(default_model().float(), kpoints=(3, 3, 3))
si = bulk("Si", "diamond", a=5.43)
si.calc = calc
si.get_potential_energy()                       # sets the Fermi level

H, S = calc.get_HS(si)
w = sla.eigh(H[0], S[0], eigvals_only=True)     # k-point 0
eigenvalues_eV = w * 27.211 - calc.get_fermi_level()
```

Toggles (constructor keywords): `compute_forces`, `compute_stress`,
`use_scc`, `include_dos`, `kpoints`, `cutoff`, `kT`, `alpha`, `beta`,
`device`. Setting `compute_forces=False` gives a fast energy-only path
for high-throughput screening.

Notes: `alpha` scales the band-structure energy and `beta` the forces;
both default to `1.0`, which gives the standard DFTB total energy
`E = E_band + E_rep` together with its exact gradient. Energy, forces
and stress have been checked against finite differences (agreement
better than 0.5% for bulk Si and SiC), so cell relaxation with
`ExpCellFilter` is supported. A full runnable demo is in
`slakonet/examples/slakonet_calculator_example.py`. See also the ASE docs
page *Calculators -> SlaKoNet*.

## Supported Materials

- **Elements**: Z = 1-65
- **Material classes**: Oxides, carbides, nitrides, chalcogenides, halides, intermetallics
- **Crystal structures**: All major structure types 

## Performance Benchmarks

Accuracy: 0.76 eV MAE for band gaps (vs 0.38 eV for reference TB-mBJ
DFT), validated on 50 semiconductor/insulator compounds.

### Scaling

Time per diagonalization, with peak GPU memory in brackets (GB). The
dense `eigh` path is limited to roughly 7,000 orbitals; beyond that the
sparse solver is the only option.

| atoms | Norb | dense eigh (s) | sparse solve (s) |
| ---: | ---: | ---: | ---: |
| 128 | 1,152 | 0.15 [2.6] | 0.12 [2.6] |
| 1,024 | 9,216 | – (Norb > 7k) | 3.71 [3.4] |
| 3,456 | 31,104 | – | 56.9 [5.3] |
| 8,192 | 73,728 | – | 403 [10.0] |
| 11,664 | 104,976 | – | 956 [19.2] |
| 16,000 | 144,000 | – | > 30 min (timeout) |

![SlakoNet timing](https://github.com/atomgptlab/slakonet/blob/main/slakonet/examples/timing.png)

## Output Properties

- Band structures along high-symmetry k-paths
- Total, atom-projected and orbital-projected DOS (s/p/d)
- Band gaps (direct/indirect) and band edges
- Fermi energy
- Hamiltonian and overlap matrices

## Dataset

- [Figshare TBmBJ dataset](https://figshare.com/projects/JARVIS-DFT_TBmBJ/84020)

## Methodology

SlakoNet employs a neural network to learn distance-dependent Slater-Koster parameters:
- **Basis set**: sp³d tight-binding orbitals
- **Training data**: JARVIS-DFT with TB-mBJ functional
- **Loss function**: Combined DOS + band gap optimization
- **Framework**: PyTorch with GPU acceleration
- **Cutoff radius**: 7 Å for orbital interactions

## Limitations

- Limited to elements Z ≤ 65
- Trained on specific meta-GGA DFT (TBmBJ)
- Discrepancies in conduction band descriptions
- No self-consistent cycle
- No spin-orbit coupling or magnetic properties

## Citation

If you use SlakoNet in your research, please cite:

```bibtex
@article{choudhary2025slakonet,
  title={SlaKoNet: A Unified Slater-Koster Tight-Binding Framework Using Neural Network Infrastructure for the Periodic Table},
  author={Choudhary, Kamal},
  journal={ChemRxiv},
  doi={https://doi.org/10.26434/chemrxiv-2025-4vjr9-v2},
  year={2025}
}
```

