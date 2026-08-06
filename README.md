# SlakoNet

Accurate and efficient prediction of electronic band structures is essential for designing materials with targeted properties. However, existing machine learning models often lack universality and struggle to predict detailed electronic structures, while traditional tight-binding models based on the Slater-Koster (SK) formalism suffer from (i) limited transferability, (ii) the need for manual parameterization, and (iii) training on low-fidelity electronic structure data. To address these challenges, I introduce SlaKoNet, a parameter optimization framework that learns SK-based Hamiltonian matrix elements across 65 elements of the periodic table using automatic differentiation. SlaKoNet is trained on density functional theory data from the JARVIS-DFT database using the Tran-Blaha modified Becke-Johnson (TBmBJ), encompassing over 20000 materials. The framework achieves a mean absolute error (MAE) of 0.74 eV for bandgap predictions against experimental data, representing a reasonable improvement over standard GGA functionals (MAE = 1.14 eV) while preserving the computational advantages and physical interpretability of tight-binding methods. SlaKoNet demonstrates promising scalability with up to 8.4× speedup on GPUs, enabling rapid electronic structure screening for materials discovery.


![SlakoNet schematic](https://github.com/atomgptlab/slakonet/blob/main/slakonet/examples/sk_schematic.png)

## Key Features

- **Universal parameterization**: Works across 65 elements and their combinations
- **Physics-informed**: Based on Slater-Koster tight-binding formalism
- **High accuracy**: Mean absolute error of 0.74 eV for band gaps vs experimental values
- **Scalable**: GPU-accelerated calculations for systems up to 2000 atoms
- **Comprehensive properties**: Predicts band structures, DOS, band gaps, and orbital projections

## Installation
Install via pip:
```bash
pip install slakonet
```

Or create a conda environment and install SlaKoNet in editable mode. To do so, first, install miniforge https://github.com/conda-forge/miniforge. For example: 

```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
```

Based on your system requirements, you'll get a file something like 'Miniforge3-XYZ'.

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

[Open in Colab](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/slakonet_example.ipynb)

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

| Name | Description |
| --- | --- |
| `slakonet_v0` | Original universal parameter set (paper v1) |
| `slakonet_v1` | Second-generation universal parameter set |
| `slakonet_v1a` | Refined v1 parameter set |

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

- **Accuracy**: 0.76 eV MAE for band gaps (vs 0.38 eV for reference TB-mBJ DFT)
- **Speed**: <10 seconds for 1000-atom systems on GPU
- **Scalability**: Efficient with GPU acceleration
- **Coverage**: Validated on 50 semiconductor/insulator compounds for experiments

![SlakoNet timing](https://github.com/atomgptlab/slakonet/blob/main/slakonet/examples/timing.png)


## Output Properties

SlakoNet predicts comprehensive electronic properties including:

- Electronic band structures along high-symmetry k-paths
- Total and projected density of states (DOS)
- Band gaps (direct/indirect) and band edges
- Fermi energy and electronic structure topology
- Atom-projected and orbital-projected DOS (s/p/d contributions)

## Applications

- High-throughput materials screening
- Electronic structure prediction without expensive DFT
- Band structure and DOS calculations for device design
- Semiconductor and quantum materials discovery
- Educational tools for solid-state physics


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

