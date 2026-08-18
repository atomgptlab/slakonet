# API reference

The most useful public entry points. For the full source, see the
[GitHub repository](https://github.com/atomgptlab/slakonet).

## `slakonet.optim`

### `default_model()`

```python
from slakonet.optim import default_model
model = default_model()
```

Loads (and caches on first use) the trained SlaKoNet model covering 65
elements. Call once; reuse the returned object everywhere. Often used as
`default_model().float()`.

Pass `model_name=` to pick a specific parameter set:

| Name | Elements | Notes |
| --- | --- | --- |
| `slakonet_v0` | 64 | Original universal set (paper v1) |
| `slakonet_v1` | 75 | Second-generation set; widest coverage, over-stiff EOS |
| `slakonet_v1a` | 64 | Default. Untrained H/S, untouched repulsive bar 496 refit pairs |
| `slakonet_base75` | 75 | Untrained reference tables over their full element range |
| `slakonet_v1a_full` | 75 | `v1a` overlaid on `slakonet_base75` |

The last two are built locally rather than downloaded, from the
reference skf files on [Zenodo](https://zenodo.org/records/14289468) —
see `slakonet/examples/build_v1a_extended.py` and `build_v1a_full.py`.
They land in the same cache directory as the downloaded sets.

### `default_mu(full=False, model_name=None)`

```python
from slakonet.optim import default_mu
mu = default_mu()              # {element: chemical potential, eV}
meta = default_mu(full=True)   # full record incl. calibration metadata
```

Returns the per-element chemical potentials bundled with SlaKoNet, used
for formation energies. They are calibrated per parameter set, so
`model_name` must match the set the energies come from. See
[Formation energies](guide/formation-energy.md).

## `slakonet.ase_calc`

### `SlaKoNetCalculator`

```python
from slakonet.ase_calc import SlaKoNetCalculator
calc = SlaKoNetCalculator(model, kpoints=(3, 3, 3))
```

A standard ASE `Calculator`. Constructor options are documented in the
[ASE calculator guide](guide/ase-calculator.md).

**Standard ASE properties**

| Call | Returns |
| --- | --- |
| `atoms.get_potential_energy()` | total energy, eV |
| `atoms.get_forces()` | forces, eV/Å, shape `(N, 3)` |
| `atoms.get_stress()` | stress, eV/Å³, Voigt 6-vector |

**SlaKoNet-specific methods**

| Method | Returns |
| --- | --- |
| `calc.get_bandgap()` | band gap (eV), from the MP grid |
| `calc.get_fermi_level()` | Fermi level (eV) |
| `calc.band_structure(atoms, path=None, npoints=80, savefig=None)` | dict with `energies`, `kpts`, `labels`, `path`, `gap`, `vbm`, `cbm` |
| `calc.dos(atoms, energy_range=(-10, 10), num_points=3000, sigma=0.1)` | `(energies, dos)` arrays, Fermi-referenced |

### `SlaKoNetConfig`

```python
from slakonet.ase_calc import SlaKoNetConfig
cfg = SlaKoNetConfig(kpoints=[4, 4, 4], use_scc=True)
calc = SlaKoNetCalculator(model, config=cfg)
```

A declarative configuration object (pydantic). Accepts the same fields
as the calculator constructor; also constructible from a `dict` or a
JSON file. Explicit keyword arguments to `SlaKoNetCalculator` override
the config.

## `slakonet.predict_slakonet`

### `plot_band_dos_atoms(...)`

```python
from slakonet.predict_slakonet import plot_band_dos_atoms
plot_band_dos_atoms(atoms=atoms, model=model,
                    filename="bands_dos.png")
```

Convenience function: computes and plots a combined band-structure +
DOS figure for a structure.

## Quick map of the package

| Module | Purpose |
| --- | --- |
| `slakonet.optim` | model loading (`default_model`), chemical potentials (`default_mu`) |
| `slakonet.ase_calc` | the ASE `SlaKoNetCalculator` and `SlaKoNetConfig` |
| `slakonet.predict_slakonet` | band-structure / DOS prediction helpers |
| `slakonet.main` | core driver (`SimpleDftb`, shell-dict helpers) |
| `slakonet.slaterkoster` | Slater-Koster Hamiltonian / overlap construction |
| `slakonet.atoms`, `slakonet.basis` | geometry and basis-set containers |

!!! tip
    For day-to-day use you only need `slakonet.optim` and
    `slakonet.ase_calc` — the rest is internal machinery.
