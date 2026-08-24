# SlaKoNet examples

Runnable scripts that exercise the major user-facing features. Most
scripts load a trained model once and reuse it; some download a model
from Figshare on first use. The "Sparse SK" and "Validation" groups
were added during the million-atom scaling work and are the
recommended starting points.

---

## ASE calculator

| script | what it does |
| --- | --- |
| `slakonet_calculator_example.py` | End-to-end demo of `SlaKoNetCalculator`: load the model once, get energy/forces/stress on bulk Si, run a band structure (`-> si_v1_bands.png`) and DOS (`-> si_v1_dos.png`), then reuse the same calculator on Ge with no model reload. |
| `chipstb_bandgaps.py` | High-throughput tutorial over the ChIPS-TB JARVIS-DFT set with one reused `SlaKoNetCalculator`. Computes each band gap **two ways** — 3×3×3 Monkhorst-Pack grid vs high-symmetry band path — and reports which has the lower MAE vs MBJ (parity plot `-> chipstb_parity.png`). Also computes formation energies using the chemical potentials bundled with slakonet (`slakonet.optim.default_mu()`; override via `MU_JSON`). Output: `chipstb_bandgaps.csv`. |

## Sparse Slater–Koster (assembly + interior eigensolver)

The vectorized sparse path is bit-exact vs the validated dense path and
is the building block for the million-atom regime.

| script | what it does |
| --- | --- |
| `validate_sparse_sk.py`     | Finite H and S from `hs_matrix_sparse` match the dense reference to **exactly 0.0** on Si4 (homonuclear, s/p/d) and Si3C2 (heteronuclear). |
| `validate_sparse_solver.py` | `solve_near_gap` (shift-invert Lanczos) interior eigenvalues vs the dense reference, ~1e-7 Ha on Si4 and Si3C2. |
| `validate_sparse_periodic.py` | Periodic complex H(k)/S(k) vs dense `eighb` on bulk Si over an 8-k MP grid, ~3.6e-7 Ha. |
| `check_direct_assembly.py`  | `assembly="direct"` (vectorized SK→COO scatter) vs `assembly="pairwise"` (dense reuse) — bit-exact AND 10–75× faster on the periodic case. |
| `benchmark_scaling.py`      | Dense vs sparse scaling: per-size assemble + eigensolve + peak RSS table across Si cluster sizes. |
| `max_atoms_sparse.py`       | Stress test: ramp finite-Si cluster size until per-size assembly time or available memory is exceeded; logs incrementally to `max_atoms_sparse_log.csv`. |

## Correctness checks (forces / stress / autograd)

| script | what it does |
| --- | --- |
| `check_autograd_forces.py` | Verifies `SlaKoNetCalculator` forces come from `torch.autograd` and match a central finite-difference of the energy to FD-truncation level (~0.04% on a distorted Si dimer). |
| `check_stress.py`          | Applies ±ε strains in the 6 Voigt components, finite-differences the energy (ASE tensile-positive convention σ = −(1/V) ∂E/∂ε), and compares to the calculator's stress. |

## Property prediction / I-O

| script | what it does |
| --- | --- |
| `predict_bands_from_poscar.py` | Predict the band structure for a structure from a POSCAR. |
| `predict_formation_energy.py`  | Predict formation energy via a trained model. |
| `run_inference.py`             | Generic inference driver. |
| `serve_universal_v1.py`        | Lightweight HTTP/serving wrapper for the universal model. |
| `nacl_scc.py`, `ni_spin_bands.py`, `nio_spin_bands.py` | Worked examples of SCC and spin-polarised bands. |
| `si_eos.py`, `si_eos_check.py`, `si_eos_fit.py`, `si_eos_refit.py` | Si equation-of-state workflows (also useful for stress/bulk-modulus validation). |

## Band structure & Fermi surfaces

| script | what it does |
| --- | --- |
| `mgb2_fermi_bands.py` | MgB2 (the 39 K superconductor) end-to-end demo of four analyses, matplotlib-only: **(1)** band structure + DOS along a k-path (`-> MgB2_bands_dos.png`); **(2)** 3D band structure — bands near E_F as surfaces over the Brillouin zone (`-> MgB2_bands3d.png`); **(3)** 2D Fermi surface — E=0 contours of the Fermi-crossing bands at kz=0 (`-> MgB2_fermi2d.png`); **(4)** 3D Fermi surface — isosurfaces extracted with marching cubes on a full 3D k-mesh (`-> MgB2_fermi3d.png`, needs `scikit-image`). One shared `kmesh_eigs` helper runs SlaKoNet on a Cartesian k-grid; the model is loaded once. Mirrors the analyses in the SlaKoNet web backend. |

## SlaKoNetDB (Hamiltonian/overlap database)

| script | what it does |
| --- | --- |
| `slakonetdb_record.py` | Builds one database record: real-space `H(R)`/`S(R)` plus bands, DOS, gap, VBM/CBM, Fermi level, total and formation energy. Archives `H(R)`, not `H(k)`: a uniform Gamma-centred mesh is invertible, so the real-space blocks reproduce **any** k afterwards, while a band path only ever reproduces itself. Two gates decide whether a record is usable, and both are stored. `recon_err` rebuilds `H` at an off-mesh k and compares against a direct evaluation -- the mesh is grown until that holds, since the geometric estimate (`N_i >= 2*cutoff/d_i + 1`, from the *interplanar spacing*, not `|a_i|`) is a lower bound rather than a guarantee. `min_eig_S` rejects a non-positive-definite overlap: `H c = e S c` is then not a valid eigenproblem, and the solver returns eigenvalues reaching 1e5 eV alongside a perfectly plausible band gap. |
| `slakonetdb_run.py` | Shard runner: one Slurm array task processes a stride slice of the hull-stable set, resumable (skips existing records), logging one JSON line per structure. |
| `slakonetdb.slurm` | Slurm array template. Pins `OMP_NUM_THREADS` per task -- torch and BLAS otherwise each grab every core and the tasks thrash -- and sets `--mem` per task, without which Slurm hands each task the whole node and only one runs per node. |

## Parameter-set building

| script | what it does |
| --- | --- |
| `build_v1a_extended.py` | Builds a model straight from a directory of reference `.skf` files ([Zenodo](https://zenodo.org/records/14289468)), over all 75 elements they cover. `--check slakonet_v1a` verifies the shared pairs against the cached v1a (they match to 0.000e+00 — v1a's H/S are the untrained tables). |
| `build_v1a_full.py` | Builds `slakonet_v1a_full` by overlaying `slakonet_v1a` onto the 75-element untrained set: `v1a` wins on all 4096 pairs it defines (so energies match `v1a` exactly, and its 496 refit repulsive splines are preserved), the base fills the other 1529. Result: untrained H/S everywhere, over the full 75-element range. Writes into the slakonet cache, so `default_model("slakonet_v1a_full")` finds it. |
| `recalibrate_mu.py` | Recalibrate per-element chemical potentials against a given parameter set (they are model-specific). |

## Training / model building / SKF tooling (developer-oriented)

These build or refit Slater–Koster files, repulsive splines, and
universal models. They are research scripts rather than user
interfaces — read each one before running.

```
al_repulsive_refit.py   Aluminum repulsive-spline refit
jarvis_*.py      JARVIS-driven fits (bulk modulus, etc.)
papers_*.py      Joint/targeted fits used for the SlaKoNet paper
si_*.py          Silicon-only fits and clean-up workflows
universal_*.py   Universal-model build and regression
build_hybrid_v5.py
                 Merge SlaKoNet model checkpoints into a hybrid bundle
merge_models.py, merge_v3_into_v2.py
                 Merge model checkpoints
convert_to_safetensors.py
                 Convert .pt -> .safetensors
calibrate_mu.py  Calibrate chemical potentials for formation-energy
plot_fig*.py     Figures used in publications
```

## Tips

- A trained model loads once (~20 s); reuse the same `model` object
  across all subsequent calls — `SlaKoNetCalculator` does this for you.
- The sparse path defaults to `assembly="direct"`; pass
  `assembly="pairwise"` to fall back to the slow but bit-equivalent
  reference if you need to debug.
- For forces and stress, the `SimpleDftb` Bohr→Å unit fix is applied
  automatically. Use `beta=1.0` if you want physically meaningful
  forces (default `0.1` scales them down ×10).
