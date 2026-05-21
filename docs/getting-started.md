# Quickstart

This page walks through your first SlaKoNet calculations. Five minutes,
copy-paste-able. If you have not installed SlaKoNet yet, see
[Installation](installation.md).

## 1. Load the model once

SlaKoNet ships a trained model covering 65 elements. Load it **once** and
reuse it — loading is the only slow step.

```python
from slakonet.optim import default_model

model = default_model().float()
```

## 2. Attach the ASE calculator

`SlaKoNetCalculator` is a standard
[ASE](https://wiki.fysik.dtu.dk/ase/) `Calculator`. Build it once and
attach it to as many structures as you like — the model is never
reloaded.

```python
from ase.build import bulk
from slakonet.ase_calc import SlaKoNetCalculator

calc = SlaKoNetCalculator(model, kpoints=(3, 3, 3))

si = bulk("Si", "diamond", a=5.43)
si.calc = calc
```

## 3. Get properties

```python
si.get_potential_energy()    # total energy, eV
calc.get_bandgap()           # band gap, eV
calc.get_fermi_level()       # Fermi level, eV
si.get_forces()              # eV / Ang   (N, 3)
si.get_stress()              # eV / Ang^3 (Voigt 6)
```

That is the whole standard-properties workflow.

!!! note "Forces & stress"
    Forces and stress are evaluated with PyTorch autograd. Forces are
    scaled by `beta` (see the [ASE calculator guide](guide/ase-calculator.md));
    pass `beta=1.0` for physically meaningful forces.

## 4. Band structure and DOS

A band structure along the standard high-symmetry path, plus the density
of states:

```python
bs = calc.band_structure(si, path="GXWKGL", npoints=120,
                         savefig="si_bands.png")
print("gap:", bs["gap"], "eV")

energies, dos = calc.dos(si)
```

See [Band structure](guide/band-structure.md) and
[Density of states](guide/dos.md) for the details.

## 5. Reuse across structures

The same calculator works for any material — no reload:

```python
ge = bulk("Ge", "diamond", a=5.66)
ge.calc = calc
print(ge.get_potential_energy(), calc.get_bandgap())
```

This reuse pattern is what makes SlaKoNet fast for screening — see
[Band-gap screening](guide/bandgap-screening.md) for a 50-material
benchmark in one loop.

## Where to next

<div class="grid cards" markdown>

-   [:material-cog: **ASE calculator**](guide/ase-calculator.md) — every
    option, explained.
-   [:material-chart-line: **Band structure**](guide/band-structure.md) —
    high-symmetry paths and plots.
-   [:material-flask: **Examples & Colab**](examples.md) — run it in your
    browser, no install.
-   [:material-school: **How it works**](methodology.md) — the method.

</div>
