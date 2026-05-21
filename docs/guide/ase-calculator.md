# ASE calculator

`SlaKoNetCalculator` is the main entry point: a standard
[ASE](https://wiki.fysik.dtu.dk/ase/) `Calculator` that wraps a loaded
SlaKoNet model. Attach it to an `ase.Atoms` object and use the usual ASE
API, plus dedicated methods for band structure and DOS.

## The basic pattern

```python
from ase.build import bulk
from slakonet.optim import default_model
from slakonet.ase_calc import SlaKoNetCalculator

model = default_model().float()                 # load ONCE
calc = SlaKoNetCalculator(model, kpoints=(3, 3, 3))

si = bulk("Si", "diamond", a=5.43)
si.calc = calc
si.get_potential_energy()
```

The trained model is **injected** into the calculator and reused for
every structure and every call — it is never reloaded. This is the
recommended high-throughput pattern.

## Constructor options

| Argument | Default | Meaning |
| --- | --- | --- |
| `model` | — | a loaded SlaKoNet model (required) |
| `kpoints` | `(3, 3, 3)` | Monkhorst-Pack grid for energy / gap |
| `cutoff` | `10.0` | interaction cutoff (Bohr) |
| `kT` | `0.025` | Fermi smearing (eV) |
| `alpha` | `0.1` | electronic-energy mixing weight |
| `beta` | `0.1` | force scaling — see note below |
| `use_scc` | `False` | self-consistent charges (slower) |
| `compute_forces` | `True` | evaluate forces (autograd) |
| `compute_stress` | `True` | evaluate stress (needs forces + a cell) |
| `include_dos` | `False` | also compute DOS during `calculate()` |
| `device` | auto | `"cpu"` or `"cuda"` |

A configuration can also be supplied as a `SlaKoNetConfig` object, a
plain `dict`, or a path to a JSON file:

```python
from slakonet.ase_calc import SlaKoNetCalculator, SlaKoNetConfig

cfg = SlaKoNetConfig(kpoints=[4, 4, 4], use_scc=True)
calc = SlaKoNetCalculator(model, config=cfg)
```

Explicit keyword arguments always override the config, so existing call
sites keep working.

## Standard properties

```python
si.get_potential_energy()    # eV
si.get_forces()              # eV / Ang, shape (N, 3)
si.get_stress()              # eV / Ang^3, Voigt 6-vector
calc.get_bandgap()           # eV
calc.get_fermi_level()       # eV
```

!!! warning "Forces are scaled by `beta`"
    SlaKoNet returns `forces = -beta * dE/dx`. With the default
    `beta = 0.1` the forces are scaled down 10×. For **physically
    meaningful forces** (relaxation, molecular dynamics, elastic
    constants) construct the calculator with `beta=1.0`:

    ```python
    calc = SlaKoNetCalculator(model, beta=1.0)
    ```

    Forces are autograd-derived and, with `beta=1.0`, agree with a
    finite-difference of the energy to finite-difference precision.

## The fast path

For high-throughput band-gap screening you usually do not need forces or
stress. Turn them off:

```python
calc = SlaKoNetCalculator(model, kpoints=(3, 3, 3),
                          compute_forces=False,
                          compute_stress=False)
```

This skips the autograd force/stress evaluation and is substantially
faster. The band gap is still produced.

## Band structure and DOS

Two dedicated methods go beyond the standard ASE properties:

```python
bs = calc.band_structure(si, path="GXWKGL", npoints=120,
                         savefig="si_bands.png")
# bs["energies"]  -> (n_k, n_band) eV, referenced to mid-gap
# bs["gap"], bs["vbm"], bs["cbm"], bs["path"]

energies, dos = calc.dos(si)   # Fermi-referenced energies, DOS
```

Full details: [Band structure](band-structure.md) and
[Density of states](dos.md).

## Two band-gap definitions

SlaKoNet can report a band gap two ways, and they are **not identical**:

- `calc.get_bandgap()` — gap from the **Monkhorst-Pack grid** used by
  `calculate()`. Fast; good for a quick metallic / non-metallic screen.
- `calc.band_structure(...)["gap"]` — gap from **VBM/CBM bracketing
  along a high-symmetry path**. More accurate for actual gap values,
  because a coarse uniform grid can miss the true band extrema.

For band-gap accuracy, prefer the band-path value. For a fast screen,
the grid value is fine. See
[Band-gap screening](bandgap-screening.md) for a quantitative comparison.

## Reusing across structures

```python
for atoms in structures:
    atoms.calc = calc           # same calculator, same model
    atoms.get_potential_energy()
```

No reload, no re-initialization — this is the intended workflow.
