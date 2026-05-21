# Density of states

The density of states (DOS) counts how many electronic states are
available at each energy. It is the natural companion to a band
structure and is often easier to interpret for gaps, metallicity and
orbital character.

## Computing the DOS

```python
from ase.build import bulk
from slakonet.optim import default_model
from slakonet.ase_calc import SlaKoNetCalculator

calc = SlaKoNetCalculator(default_model().float())

si = bulk("Si", "diamond", a=5.43)
energies, dos = calc.dos(si)
```

`calc.dos()` returns two 1-D arrays:

- `energies` — energy grid in eV, **referenced to the Fermi level**
  (so `0.0` is the Fermi energy);
- `dos` — the density of states on that grid.

## `dos()` arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `atoms` | — | the ASE `Atoms` object |
| `energy_range` | `(-10, 10)` | energy window (eV, relative to Fermi) |
| `num_points` | `3000` | number of energy grid points |
| `sigma` | `0.1` | Gaussian broadening (eV) |

Smaller `sigma` resolves sharp features; larger `sigma` smooths noise
from a finite k-mesh.

## Plotting it

```python
import matplotlib.pyplot as plt

energies, dos = calc.dos(si)

plt.plot(energies, dos, lw=1.0)
plt.axvline(0.0, ls="--", c="k", lw=0.6)   # Fermi level
plt.xlabel(r"$E - E_\mathrm{F}$ (eV)")
plt.ylabel("DOS")
plt.xlim(-10, 10)
plt.savefig("si_dos.png", dpi=200)
```

For a semiconductor like silicon you will see a clean gap around
`E = 0`; for a metal the DOS is finite at the Fermi level.

## Band structure + DOS together

The `slakonet_calculator_example.py` and `mgb2_fermi_bands.py` scripts
in [`slakonet/examples/`](https://github.com/atomgptlab/slakonet/tree/main/slakonet/examples)
produce a combined band-structure-plus-DOS figure — a good starting
point for a publication-style panel.

## Reading the DOS

| Feature | Interpretation |
| --- | --- |
| Gap of zero DOS at `E = 0` | semiconductor / insulator; gap width = band gap |
| Finite DOS at `E = 0` | metal |
| Sharp peaks | flat bands / localized (often d- or f-) states |
| Broad features | dispersive (often s/p) bands |

See [Fermi surfaces](fermi-surface.md) for the **k**-resolved view of
states at the Fermi level in metals.
