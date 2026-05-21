# Performance

SlaKoNet is built for **speed at scale** — quantum-level electronic
structure cheap enough to screen thousands of materials.

## Accuracy

| Quantity | SlaKoNet | Reference |
| --- | --- | --- |
| Band-gap MAE vs experiment | **0.74 eV** | GGA functionals: 1.14 eV |
| Training data | JARVIS-DFT, TBmBJ level | 20,000+ materials |
| Element coverage | 65 elements | — |

The MBJ-level training data is what lets a tight-binding model reach
gap accuracy competitive with much more expensive methods.

## Speed

- **Per material**: a band structure or band gap takes seconds, not the
  minutes-to-hours of a DFT calculation.
- **GPU acceleration**: up to **8.4× speedup** on GPU versus CPU for the
  electronic-structure solve.
- **High throughput**: load the model once, then screen hundreds of
  materials in a single loop — see
  [Band-gap screening](guide/bandgap-screening.md).

## k-sampling matters

How you sample the Brillouin zone affects the predicted band gap. On a
benchmark set of JARVIS-DFT materials:

| k-sampling | Band-gap MAE vs MBJ |
| --- | --- |
| 3×3×3 Monkhorst-Pack grid | 0.95 eV |
| High-symmetry band path | **0.68 eV** |

A coarse uniform grid can miss the true band extrema and overestimates
the gap; a high-symmetry path samples where the valence/conduction edges
actually sit. **Use the band-path gap for accurate values**, the grid
gap for a fast metallic / non-metallic screen.

## Choosing settings for your workload

| Goal | Recommended settings |
| --- | --- |
| Fast band-gap screen | `compute_forces=False`, MP grid `(3,3,3)` |
| Accurate band gap | `band_structure()` with a high-symmetry path |
| Forces / relaxation | `beta=1.0`, `compute_forces=True` |
| Formation energies | `alpha=1.0`, bundled `default_mu()` |
| Charge-sensitive systems | `use_scc=True` (slower) |

## Tips

- **Reuse the calculator.** Loading the model is the only slow step;
  never rebuild the calculator inside a loop.
- **Turn off what you do not need.** `compute_forces=False` and
  `compute_stress=False` skip the autograd evaluation for a pure
  band-gap screen.
- **Use the GPU** for large or high-throughput runs — install a
  CUDA-enabled PyTorch and pass `device="cuda"`.
- **Fewer band-path points** are fine for the gap — `npoints=20` already
  brackets the VBM/CBM; reserve large `npoints` for publication figures.
