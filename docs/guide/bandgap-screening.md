# Band-gap screening

One of SlaKoNet's strengths is **high-throughput** electronic-structure:
load the model once, then evaluate hundreds or thousands of materials in
a single loop. This guide shows the screening pattern and a benchmarked
comparison of k-sampling choices.

## The screening loop

```python
from slakonet.optim import default_model
from slakonet.ase_calc import SlaKoNetCalculator

model = default_model().float()
# forces/stress off -> fast energy + gap only
calc = SlaKoNetCalculator(model, kpoints=(3, 3, 3),
                          compute_forces=False,
                          compute_stress=False)

results = {}
for atoms in my_structures:        # any iterable of ASE Atoms
    atoms.calc = calc
    atoms.get_potential_energy()
    results[atoms.get_chemical_formula()] = calc.get_bandgap()
```

The model is loaded once and reused — each material then costs only a
single eigensolve.

## Worked benchmark

[`chipstb_bandgaps.py`](https://github.com/atomgptlab/slakonet/blob/main/slakonet/examples/chipstb_bandgaps.py)
benchmarks a curated set of JARVIS-DFT materials and compares the
predicted gaps to MBJ reference values. It evaluates each gap **two
ways**:

1. **3×3×3 Monkhorst-Pack grid** — `calc.get_bandgap()`. One uniform
   mesh, fast.
2. **High-symmetry band path** — `calc.band_structure()`. The standard
   k-path; samples band extrema along the symmetry lines.

On the benchmark set the band path is clearly more accurate:

| k-sampling | Band-gap MAE vs MBJ |
| --- | --- |
| 3×3×3 Monkhorst-Pack grid | 0.95 eV |
| High-symmetry band path | **0.68 eV** |

**Why the path wins:** a coarse uniform grid frequently *misses* the
true valence/conduction band extrema and therefore **overestimates** the
gap. The high-symmetry path samples exactly where the VBM and CBM
usually sit.

!!! tip "Practical recommendation"
    Use the **MP grid** for a quick metallic / non-metallic screen, and
    the **band-path** gap when you need accurate gap *values*. As few as
    20 path points already bracket the gap well.

## Formation energies in the same pass

If you also want stability information, compute formation energies in
the same loop — see [Formation energies](formation-energy.md). The
benchmark script does this automatically using SlaKoNet's bundled
per-element chemical potentials.

## Performance notes

- **`compute_forces=False`** is the single biggest speed-up for a pure
  gap screen — it skips the autograd force/stress evaluation entirely.
- **GPU**: the MP-grid solve runs well on GPU. Set `device="cuda"`.
- **Reuse the calculator** — never rebuild it inside the loop.

## Output

The benchmark writes:

- `chipstb_bandgaps.csv` — per-material MP-grid gap, band-path gap, MBJ
  reference, and formation energy;
- `chipstb_parity.png` — a parity plot of both schemes against MBJ.
