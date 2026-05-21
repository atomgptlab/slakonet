# FAQ

## How accurate is SlaKoNet?

Band gaps reach a mean absolute error of **0.74 eV** against
experimental values — better than standard GGA functionals (1.14 eV).
Band *shapes* and *gaps* are reliable; absolute band positions are less
so, as with any tight-binding method.

## Which elements are supported?

SlaKoNet covers **65 elements** of the periodic table and their
combinations. Compounds containing unsupported elements (mostly heavy
lanthanides/actinides) cannot be evaluated.

## Why are my forces 10× too small?

The calculator returns `forces = -beta * dE/dx`, and `beta` defaults to
`0.1`. For physically meaningful forces build the calculator with
`beta=1.0`:

```python
calc = SlaKoNetCalculator(model, beta=1.0)
```

## Why do `get_bandgap()` and `band_structure()["gap"]` disagree?

They use different Brillouin-zone sampling:

- `get_bandgap()` — a 3×3×3 Monkhorst-Pack grid;
- `band_structure()["gap"]` — VBM/CBM bracketing along a high-symmetry
  path.

The band-path value is more accurate; a coarse uniform grid can miss the
true band extrema and overestimate the gap. Use the band-path gap for
accurate values, the grid gap for a quick screen. See
[Band-gap screening](guide/bandgap-screening.md).

## The first calculation is slow — is that normal?

Yes. The first `default_model()` call downloads and caches the trained
model. After that it loads instantly. Loading the model is the only slow
step — reuse the loaded model and calculator across all your structures.

## Can it run on a GPU?

Yes. Install a CUDA-enabled PyTorch and the calculator uses the GPU
automatically, or force it with `device="cuda"`. GPU gives up to **8.4×
speedup** for the electronic-structure solve.

## How do I screen many materials quickly?

Load the model once, build one calculator with `compute_forces=False`,
and loop. See [Band-gap screening](guide/bandgap-screening.md) for the
full pattern and a benchmark.

## Does it give forces and stress?

Yes — via PyTorch autograd. Use `beta=1.0` for correct force magnitudes.
Stress is reported in ASE units (eV/Å³, Voigt). Both require
`compute_forces=True` (the default) and, for stress, a periodic cell.

## What is `alpha` for?

`alpha` weights the electronic energy in the total energy. It does **not**
affect eigenvalues, so band gaps are independent of it. Use `alpha=1.0`
for formation energies (the bundled chemical potentials are calibrated
for it).

## Should I use `use_scc`?

Only for charge-sensitive systems. `use_scc=True` adds a
self-consistent-charge correction at extra cost. The default
non-self-consistent solve is appropriate for most band-gap and screening
work.

## How do I cite SlaKoNet?

See the [repository](https://github.com/atomgptlab/slakonet) for the
current reference.

## Where do I report a bug or ask a question?

Open an issue on
[GitHub](https://github.com/atomgptlab/slakonet/issues).
