# SlakoNet

A unified Slater-Koster tight-binding neural network framework for predicting electronic band structures across the periodic table. SlakoNet combines the computational efficiency of tight-binding methods with the accuracy of density functional theory through deep learning.

## Overview

SlakoNet learns Slater-Koster Hamiltonian matrix elements from DFT data to predict electronic structures for materials containing elements Z ≤ 65. The model is trained on ~20,000 materials from the JARVIS-DFT database using the Tran-Blaha modified Becke-Johnson (TB-mBJ) functional.

## Key Features

- **Universal parameterization**: Works across 65 elements and their combinations
- **Physics-informed**: Based on Slater-Koster tight-binding formalism
- **High accuracy**: Mean absolute error of 0.81 eV for band gaps vs experimental values
- **Scalable**: GPU-accelerated calculations for systems up to 2000 atoms
- **Comprehensive properties**: Predicts band structures, DOS, band gaps, and orbital projections

## Installation

```bash
git clone https://github.com/atomgptlab/slakonet.git
cd slakonet
pip install -e .
```

## Quick Start

### Example of Training Models

```bash
python slakonet/train_slakonet.py --config_name slakonet/examples/config_example.json
```

### Example of Inference

```bash
python slakonet/pretrained.py --model_path slakonet/tests/slakonet_v1_sic --file_path slakonet/examples/POSCAR-JVASP-107.vasp
```

### Using Pretrained Models in Python

```python
from slakonet.optim import MultiElementSkfParameterOptimizer, get_atoms
from slakonet.atoms import Geometry
from slakonet.main import generate_shell_dict_upto_Z65

# Load pretrained model
model = MultiElementSkfParameterOptimizer.load_model("slakonet_v1_sic", method="state_dict")
model.eval()

# Get structure (example with JARVIS ID)
atoms, opt_gap, mbj_gap = get_atoms("JVASP-107")  
geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
shell_dict = generate_shell_dict_upto_Z65()

# Compute electronic properties
with torch.no_grad():
    properties, success = model.compute_multi_element_properties(
        geometry=geometry,
        shell_dict=shell_dict,
        get_fermi=True,
        device="cuda"
    )

# Access results
print(f"Band gap: {properties['band_gap_eV']:.3f} eV")
print(f"Fermi energy: {properties['fermi_energy_eV']:.3f} eV")

# Plot band structure and DOS
eigenvalues = properties["eigenvalues"]
dos_values = properties['dos_values_tensor']
dos_energies = properties['dos_energy_grid_tensor']
```

## Supported Materials

- **Elements**: Z = 1-65 (H to Tb)
- **Material classes**: Oxides, carbides, nitrides, chalcogenides, halides, intermetallics
- **Crystal structures**: All major structure types with up to 2000 atoms

## Performance Benchmarks

- **Accuracy**: 0.81 eV MAE for band gaps (vs 0.37 eV for reference TB-mBJ DFT)
- **Speed**: <10 seconds for 1000-atom systems on GPU
- **Scalability**: Efficient up to 2000 atoms with GPU acceleration
- **Coverage**: Validated on 50 semiconductor/insulator compounds

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

## Methodology

SlakoNet employs a neural network to learn distance-dependent Slater-Koster parameters:
- **Basis set**: sp³d tight-binding orbitals
- **Training data**: JARVIS-DFT with TB-mBJ functional
- **Loss function**: Combined DOS + band gap optimization
- **Framework**: PyTorch with GPU acceleration
- **Cutoff radius**: 7 Å for orbital interactions

## Limitations

- Limited to elements Z ≤ 65
- Trained on specific DFT functional (TB-mBJ)
- Performance depends on similarity to training data
- Discrepancies in conduction band descriptions
- No self-consistent cycle
- No spin-orbit coupling or magnetic properties

## Citation

If you use SlakoNet in your research, please cite:

```bibtex
@article{choudhary2025slakonet,
  title={SlaKoNet: A Unified Slater-Koster Tight-Binding Neural Network for the Periodic Table},
  author={Choudhary, Kamal},
  journal={arXiv preprint},
  year={2025}
}
```

