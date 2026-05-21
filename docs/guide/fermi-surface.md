# Fermi surfaces

For a metal, the **Fermi surface** is the locus of **k**-points where
electronic bands cross the Fermi level. Its shape governs conductivity,
superconductivity and many transport properties. SlaKoNet can map it in
both 2D (a Brillouin-zone slice) and 3D (isosurfaces).

## A complete worked example

The script
[`mgb2_fermi_bands.py`](https://github.com/atomgptlab/slakonet/blob/main/slakonet/examples/mgb2_fermi_bands.py)
runs four analyses on **MgB₂** — the 39 K superconductor, a textbook
Fermi-surface case — using only matplotlib:

| Analysis | Output |
| --- | --- |
| Band structure + DOS | `MgB2_bands_dos.png` |
| 3D band structure over the Brillouin zone | `MgB2_bands3d.png` |
| 2D Fermi surface (contours at `E = E_F`) | `MgB2_fermi2d.png` |
| 3D Fermi surface (isosurfaces) | `MgB2_fermi3d.png` |

Run it:

```bash
cd slakonet/slakonet/examples
python mgb2_fermi_bands.py
```

## How it works

All four analyses share one helper that evaluates SlaKoNet on a
**Cartesian k-mesh**:

1. **2D mesh** (k_z = 0 plane) — eigenvalues on a grid of (k_x, k_y).
2. **3D mesh** — eigenvalues on a full (k_x, k_y, k_z) box.

The Fermi surface is then extracted:

- **2D**: contour the Fermi-crossing bands at energy `0` (the Fermi
  level) — `matplotlib.contour`.
- **3D**: marching cubes on each Fermi-crossing band gives triangulated
  isosurfaces (`skimage.measure.marching_cubes`).

## Minimal 2D Fermi surface

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from slakonet.optim import default_model
from slakonet.atoms import Geometry
from slakonet.main import generate_shell_dict_upto_Z65
from slakonet.optim import kpts_to_klines

model = default_model()
shell_dict = generate_shell_dict_upto_Z65(model=model)

# build a 2D Cartesian k-grid in the k_z = 0 plane, evaluate, and
# contour the bands that cross E_F -- see mgb2_fermi_bands.py for the
# full, ready-to-run implementation.
```

The example script is the recommended starting point — it handles the
Brillouin-zone geometry, band selection and plotting for you.

## When does a material have a Fermi surface?

Only **metals**. For a semiconductor or insulator no band crosses the
Fermi level, so the Fermi surface is empty — which is why the worked
example uses a metal (MgB₂). To check first, compute the band gap:

```python
from slakonet.ase_calc import SlaKoNetCalculator
calc = SlaKoNetCalculator(model)
gap = calc.get_bandgap()      # ~0 eV  -> metal, has a Fermi surface
```

## Requirements

The 3D isosurface step needs `scikit-image`:

```bash
pip install scikit-image
```

The 2D contour analysis needs only matplotlib (already a dependency).
