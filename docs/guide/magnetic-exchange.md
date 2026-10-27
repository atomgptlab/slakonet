# Magnetic exchange and magnons (TB2J)

`slakonet.tb2j_interface` connects the collinear spin-polarized slakonet
model to [TB2J](https://github.com/mailhexu/TB2J), which computes the
Heisenberg exchange constants J_ij with the magnetic force theorem
(Liechtenstein–Katsnelson–Antropov–Gubanov Green's-function method) and,
from them, the magnon spectrum. You give a crystal structure; slakonet
builds and converges the spin-polarized tight-binding Hamiltonian and TB2J
consumes it in memory. No DFT run and no intermediate files are needed.

!!! note "Scope"
    The default workflow is collinear: isotropic J_ij and TB2J's
    biquadratic term. DMI and anisotropic exchange come from the
    spin–orbit workflow described in *Spin–orbit coupling* below.

## Installation

```bash
pip install -e ./slakonet
git clone https://github.com/mailhexu/TB2J.git
pip install --no-deps -e ./TB2J        # TB2J lists `pypao`, which is not on PyPI
pip install "HamiltonIO>=0.3.8" sympair supercellmap tomli tomli-w pathos \
            tqdm pyyaml "sisl>=0.16.2" progressbar2
```

## Quick start

```python
from ase.io import read
from slakonet.optim import default_model
from slakonet.tb2j_interface import gen_exchange_slakonet, compute_magnon_bands

atoms = read("POSCAR")                     # e.g. NiO, AFM-II cell (2 Ni)
exchange, spin_model = gen_exchange_slakonet(
    atoms,
    magnetic_elements=["Ni"],
    model=default_model(),
    initial_moments=[2, -2, 0, 0],         # signed moments set the order
                                           # default model: Stoner + U + SCC,
                                           # no parameter to choose
    scf_kmesh=(6, 6, 6),                   # spin self-consistency mesh
    kmesh=(6, 6, 6),                       # TB2J mesh
    Rcut=6.0,
    output_path="TB2J_results",
)
print(spin_model.moments, spin_model.efermi)
compute_magnon_bands("TB2J_results", kpath="GXWKGL")
```

From the command line:

```bash
slakonet2J POSCAR --elements Ni --magmoms 2 -2 0 0 \
           --scf-kmesh 6 6 6 --kmesh 6 6 6 --rcut 6 --magnon --kpath GXWKGL
```

`TB2J_results/` holds the usual TB2J output (`exchange.out`,
`TB2J.pickle`, Multibinit/Vampire/TomASD files), plus
`magnon_bands.png/.json` when magnons are requested.

## What happens

1. **Exact real-space matrices.** slakonet assembles Bloch sums H(k),
   S(k). The Hamiltonian has a finite support in real space, the image
   cells inside the Slater–Koster cutoff with |n_i| ≤ n_max, so H(k) on a
   Γ-centred grid of 2 n_max + 1 points per direction is inverted *exactly*
   to H(R), S(R) (`extract_real_space_hs`). The phase convention,
   H(k) = Σ_R H(R) e^{2πik·R}, is the one TB2J uses.
2. **Spin model.** By default (`magnetic_model="hubbard"`, see *Default
   model* below) the atomic Stoner exchange of `slakonet.magnetism` plus an
   orbital-resolved +U correction and SCC charges. With
   `magnetic_model="stoner"`, only the Stoner exchange, optionally with the
   SCC charge shift of `slakonet.scc` (see *Charge self-consistency*):

       H_σ(R) = H0(R) + ½ S(R)∘(Δε_μ + Δε_ν) + σ δ_{R,0} diag(V_μ),
       V_μ = −½ I_l(Z) m_a

   Moments (and charges with `use_scc=True`) are converged on a uniform
   Monkhorst–Pack mesh directly from these real-space matrices
   (`build_spin_model`), so TB2J receives exactly the converged model.
3. **TB2J.** The two spin channels become HamiltonIO `LCAOHamiltonian`
   objects (eV, non-orthogonal basis) and run through `ExchangeCL2`. The
   exchange splitting TB2J uses is Δ = H↑(R=0) − H↓(R=0) = I_l m_a on the
   magnetic shells.

!!! warning "All bands in the Green's function"
    TB2J's `TBGreen` keeps only the bands below E_F + 5.1 eV. G(E) below
    E_F still needs the empty bands: superexchange comes from the mixing of
    occupied and empty states. When the empty minority d band lies higher
    (large I·m), J then collapses and depends on where E_F sits in the gap.
    For MnO with I_d = 0.12 Ha, J1 is −0.00, 0.16 or 0.84 meV depending on
    E_F; with all bands it is 0.34 meV at every E_F.

    The interface therefore keeps every band (`keep_all_bands=True`, cheap
    for a minimal basis). TB2J is left unchanged and restored on exit. For
    insulators the Fermi level is placed at mid-gap. With the atomic Stoner
    values the effect is small: below 1 % for NiO, MnO, MnF2 and FeF2, and
    7 % for Cr2O3 J1.

## Default model: Stoner + U by ligand class + SCC

!!! note "Update (2026-10-02): U depends on the ligand"
    Since 2026-10-02 the default U depends on the ligand class of each 3d
    site (`u_table="anion"`): O 2.74, F 9.19, Cl/Br/I 4.06, S/Se/Te 7.10
    and metallic 1.98 eV. In leave-one-out it gives 57/129 couplings
    within ×2 (universal U: 51) and 88/129 signs (94).

    DFT+U U-scans support it: the slakonet U equivalent to DFT U = 5 eV is
    about 3 eV for oxides and about 12.5 eV for fluorides. The error comes
    from the F parameters of SlaKoNet, not from U. `u_table="universal"`
    restores the single U = 2.97 eV described below.

## Universal U (u_table="universal")

No parameter has to be chosen by the user. The default magnetic model
(`magnetic_model="hubbard"`, `slakonet.hubbard_spin`) combines three terms:

- the atomic Stoner exchange (the LSDA-like term, which gives Hund's-rule
  high-spin states);
- a Dudarev +U correction on the full d occupation matrix of every 3d
  site, with one universal U = 2.97 eV (`UNIVERSAL_U_EV`);
- periodic SCC charges.

The single value of U was fitted once to 129 experimental exchange
constants of 62 compounds (ExpMagDB). The fit used a symmetric log-ratio
loss, so over- and under-estimates cost the same. Leave-one-compound-out
validation:

| model | within ×2 | correct sign | median \|J_calc/J_ref\| | MAE | no physical solution |
| --- | --- | --- | --- | --- | --- |
| **Stoner + U = 2.97 eV + SCC (default)** | **51/129** | **94/129** | **0.88** | **9.4 meV** | **16** |
| Stoner + per-class U table | 36/129 | 83/129 | 0.50 | 11.0 meV | 25 |
| Stoner, per-site I table (`magnetic_model="stoner"`) | 28/124 | 83/124 | 0.27 | 10.5 meV | 27 |
| Stoner, atomic I | 20/124 | 45/124 | 0.00 | 11.7 meV | 66 |

A single U generalises better than per-element or per-class tables, which
overfit. With the best U chosen compound by compound (an upper bound), 69
of 131 couplings fall within a factor of 2. That is the ceiling of this
tight-binding + U description.

## Stoner model: parameters fitted to experiment

With `magnetic_model="stoner"` (`build_spin_model`), every 3d site gets
by default (`stoner_table="fitted"`) the d-shell I of
`slakonet.stoner_tables`. That table was learned from 124 experimental
exchange constants of 61 compounds (ExpMagDB: INS spin-wave fits and
MAGNDATA structures), each computed in its experimental magnetic order.
The value is chosen by the automatic class of the site
(`slakonet.magnetic_sites`):

- **environment:** ionic when most nearest neighbours are anions,
  otherwise metallic;
- **formal oxidation state:** by charge balance with fixed valences,
  including structural rules (S–S and P–P dimers, oxo-anions). When that
  is undetermined (mixed valence, intermetallics), it comes from the
  Wyckoff-resolved oxidation-state atlas of Alexandria, if the
  `oxstate_atlas` package is available.

The lookup goes from (element, environment, oxidation) to (element,
environment) to (element); elements outside the table (Co, V, 4d/5d) use
the atomic values.

Leave-one-compound-out validation (couplings with |J S²| ≥ 1 meV):

| | atomic I | fitted table |
| --- | --- | --- |
| no physical magnetic solution | 66 / 124 | 27 / 124 |
| correct sign | 45 / 124 | 83 / 124 |
| within a factor of 2 | 20 / 124 | 28 / 124 |
| median \|J_calc / J_ref\| | 0.00 | 0.27 |
| mean absolute error | 11.7 meV | 10.5 meV |

The rigid Stoner model on a non-self-consistent tight-binding
Hamiltonian, not the table, limits the accuracy: the default +U model
is better (see above).
Explicit `stoner_I` values always take precedence over the table;
`stoner_table="atomic"` restores the atomic values.

## Stoner parameter: other options

The Stoner parameter I_l sets the exchange splitting and therefore
J (roughly J ∝ (I m)²). `slakonet.magnetism.DEFAULT_STONER_I` holds the
atomic values (~0.024–0.037 Ha for 3d shells). With them:

| system | moment (μB) | experiment |
| --- | --- | --- |
| bcc Fe | 2.43 | 2.22 |
| hcp Co | 1.32 | 1.6 |
| fcc Ni | 0.31 | 0.6 |
| MnO (AFM-II) | 4.66 | 4.6 |
| NiO | collapses (≈0) | 1.7–1.9 |

Late-transition-metal Mott/charge-transfer insulators (NiO) need an
effective, larger I_d (0.08–0.12 Ha), which plays the role of a Hubbard U
that the tight-binding model lacks. Always report the I values you used,
and check the moments in `spin_model.moments` before trusting J.

### Calibrating I to a target moment

Instead of choosing I by hand, `calibrate_stoner` finds, element by
element, the d-shell I that reproduces a target moment. It brackets I by
stepping up from the atomic value, accepting only converged solutions,
and then bisects:

```python
from slakonet.tb2j_interface import calibrate_stoner, gen_exchange_slakonet

stoner, spin_model = calibrate_stoner(atoms, {"Ni": 1.8}, model,
                                      initial_moments=[2, -2, 0, 0],
                                      kmesh=(6, 6, 6))
print(stoner)            # {28: {2: 0.142...}}

# or directly
gen_exchange_slakonet(atoms, ["Ni"], target_moments={"Ni": 1.8}, ...)
```

```bash
slakonet2J POSCAR --elements Ni --magmoms 2 -2 0 0 --target-moment Ni=1.8
```

Only the signs of `initial_moments` are used, to set the collinear
order. The calibrated I is stored in `spin_model.stoner_I` and written into
the header of `exchange.out`.

## +U model (orbital-resolved exchange)

`slakonet.hubbard_spin.build_hubbard_model` adds a Dudarev-type Hubbard
correction on the full 5×5 d occupation matrix of each site, on top of the
atomic Stoner term, which plays the LSDA role (Hund's rule high spin). SCC
charges are included:

    H_σ(R) = H0(R) + SCC − σ δ_{R,0} ½ I_l m_a + δ_{R,0} Σ_a U_a (½·1 − n_a^σ)

Unlike the rigid Stoner field, +U acts orbital by orbital: it polarises a
single hole (Cu d9), opens Mott-like gaps and sets the occupied–empty
splitting that controls superexchange (J ∝ t²/U). The SCF starts from the
high-spin configuration of the formal ion. It is the default model
(universal U = 2.97 eV, see above); `hubbard_U` accepts per-element or
per-site values. With Materials Project U values, for reference:

- CrI3 J1 becomes ferromagnetic (−3.1 meV vs −4.5 meV from INS).
- La2CuO4 J1 is 21 meV (INS 26–35 meV); the Stoner model has no physical
  solution there.
- MnO, FeO and NiO give high-spin moments, but superexchange about 1.5–2.5
  times too large, as expected for too small a U.

The fit is reproducible with
`examples/fit_stoner_table.py {dataset,precompute,fit} --spin-model hubbard
--grouping global`.

## Spin–orbit coupling: DMI and anisotropic exchange

`slakonet.tb2j_ncl.gen_exchange_slakonet_soc` gives the full exchange
tensor (J_iso, symmetric anisotropic J_ani, DMI vectors):

```python
from slakonet.tb2j_ncl import gen_exchange_slakonet_soc

merged, spin_model = gen_exchange_slakonet_soc(
    atoms, ["Cr"], initial_moments=m0, scf_kmesh=(5, 5, 3),
    kmesh=(5, 5, 3), Rcut=7.2, output_path="TB2J_results_soc")
```

Runnable example: `slakonet/examples/fe2o3_soc_dmi.py` (α-Fe2O3, full
tensor and DMI per shell, about 1–2 min).

The approach is one-shot, following the force theorem, as in TB2J's
split-SOC workflow:

1. The collinear model is converged (default +U model).
2. Its exchange field is rotated rigidly to a spin axis n.
3. On-site λ L·S is added on the p and d shells, in the interleaved
   spinor basis that TB2J expects.
4. TB2J `ExchangeNCL` runs with n along x, y and z.
5. The three runs are combined with TB2J's merger. A single orientation
   only gives the components transverse to the spins.

SOC constants (`SOC_ZETA_CM1`, overridable with `lambda_soc`):

- **3d:** free-ion M²⁺ ζ₃d derived from the measured ground-term fine
  structure in the NIST Atomic Spectra Database. A Landé-interval fit
  gives ζ = 2S|λ|: Ti 120, V 167, Cr 230, Fe 411, Co 534, Ni 650 and
  Cu 829 cm⁻¹, all within 3 % of Griffith's table. Mn²⁺ (⁶S, no fine
  structure) is interpolated to 312 cm⁻¹.
- **Halogens:** ζ_np = 2Δ/3 from the NIST ²P fine-structure splittings.

SOC is not self-consistent and has no two-centre terms.

Checks (tests in `tests/test_tb2j_ncl.py`):

- With SOC = 0 the collinear J is reproduced exactly, and J_ani = DMI = 0.
- In NiO the DMI vanishes, because the bonds have an inversion centre.
- In CrI3 the DMI is 0 on the centrosymmetric nearest-neighbour bond.
  On the second neighbours it is 0.008 meV, against 0.31 meV from INS.
  D/J2 is reasonable, but the tight-binding J2 itself is about 17 times
  too small, which limits DMIs mediated by long ligand paths.
- α-Fe2O3 (INS magnetic order; universal U = 2.97 eV, the default at the
  time; the O-class U of 2.74 eV, now the default, gives J3 +42 and J4 +29,
  see `examples/fe2o3_soc_dmi.py`):
  - J3 (+36.4 meV) and J4 (+27.4 meV) against INS +32.0 and +25.0.
  - D = 0 exactly on the centrosymmetric bonds, |D| ≈ 0.33 meV on the
    J3/J4 bonds, and D_ij = −D_ji.
  - The short face- and edge-sharing J1 and J2 come out AFM. The INS fit
    makes them FM, but all three DFT studies compared below also give
    them AFM.
  - Compared with the GGA+U SKKR calculation of Dannegger et al.
    (PRB 107, 184426 (2023)), the DMI components along c have the same
    shells and the same sign pattern (+ on J3, − on J4 from a given Fe).
    They are 3× (J3: 0.60 vs 0.20 meV) and 1.5× (J4: 0.35 vs 0.23 meV)
    larger, in their convention (D = 2 D_TB2J).

## Charge self-consistency (SCC)

`use_scc=True` adds the second-order SCC shift of `slakonet.scc`, which
lets charge flow between ions against an on-site Coulomb penalty. It uses:

- the Hubbard U of the highest **occupied** valence shell (O, F → p;
  3d metals → d);
- the periodic DFTB γ (Elstner's Slater-type form, with an Ewald sum for
  the long-range 1/r part);
- Anderson mixing of charges and moments.

`scc_legacy=True` (and `SimpleDftb(..., scc_legacy=True)`) restores the
original scheme, which had three problems:

- it took U from the highest-l shell of the basis, so U = 0 for O and F,
  whose basis carries an empty d shell;
- its Klopman–Ohno γ covered only the atoms inside the cell, so results
  depended on where the atoms sit in the cell;
- it used linear mixing.

It does not converge for FeF2 or MnF2 and gives unphysical charges for
Cr2O3.

## Symmetry of the self-consistency

The Stoner model has no Coulomb penalty for moving charge. Without SCC,
crystallographically equivalent sites can therefore drift into an
unphysical charge/spin disproportionation (rutile FeF2: 6 and 8 electrons
on the two Fe). `symmetrize=True` (the default) keeps |m| and Δq equal on
equivalent atoms (spglib), with the sign following the collinear order.

## Conventions

TB2J writes E = −Σ_{i≠j} J_ij e_i·e_j with unit spins and every pair
counted twice, so J > 0 is ferromagnetic. To compare with a convention
E = Σ_pairs J S² s_i·s_j (J > 0 AFM, each pair once), use J S² = −2 J_TB2J.

## Benchmark

`slakonet/examples/tb2j_benchmark.py` runs Fe, Co, Ni, NiO, MnO, MnF2,
FeF2, Cr2O3 and CrI3 in their experimental magnetic orders against an
exchange database (INS spin-wave fits and LSDA-LKAG references).
`slakonet_v1a`, default Stoner I except NiO (I_d = 0.12 Ha), no SCC,
nz = 60. J S² in meV, J S² > 0 antiferromagnetic.

| system | order | m (μB) | shell | slakonet | reference |
| --- | --- | --- | --- | --- | --- |
| bcc Fe | FM | 2.43 | J1 / J2 | −57.2 / −10.9 | −39.0 / −22.2 (LKAG) |
| hcp Co | FM | 1.32 | J1 / J2 | −21.8 / −3.3 | −27.7 / −1.1 (LKAG) |
| fcc Ni | FM | 0.31 | J1 | −1.4 | −5.6 (LKAG) |
| NiO | AFM-II | 1.61 | J1 / J2 | −1.15 / +18.2 | −1.37 / +19.0 (INS) |
| MnO | AFM-II | 4.66 | J1 / J2 | +5.7 / +11.7 | +4.8 / +5.6 (INS) |
| MnF2 | AFM | 4.75 | J1 / J2 | +3.4 / +6.8 | −0.42 / +1.89 (INS) |
| FeF2 | AFM | 3.72 | J1 / J2 | +2.2 / +4.1 | −0.02 / +1.81 (INS) |
| Cr2O3 | +−+− | 2.77 | J1 / J2 | +43.1 / +30.1 | +33.9 / +15.4 (INS) |
| CrI3 | FM | 2.90 | J1 | +3.6 | −4.5 (INS) |

The dominant couplings of the oxides come out with the right sign and
the right order of magnitude. In the fluorides the couplings are too
antiferromagnetic, which is the familiar overestimate of a model without
on-site Coulomb repulsion. CrI3 (90° Cr–I–Cr superexchange) comes out
with the wrong sign. Converged k-mesh / `nz` changes are below 1 %
(bcc Fe, 12³ → 20³).

### Default I vs. calibrated I vs. calibrated I + SCC

`--mode calibrated` calibrates I_d to the experimental moment of the
database. The target is the MAGNDATA moment capped at the spin-only 2S,
otherwise 2S. When a target lies above the saturation moment of the
model, the onset of the plateau is used. The 22 couplings with
|J S²| ≥ 1 meV (J S² in meV) give:

| mode | MAE | correct sign | median \|calc/ref\| |
| --- | --- | --- | --- |
| default I | 6.5 | 21/22 | 0.82 |
| calibrated I | 7.1 | 21/22 | 1.09 |
| calibrated I + SCC | 6.3 | 19/22 | 1.00 |

Matching the moment does not improve J on average. It fixes some cases:

- bcc Fe J1: −40.8 vs −39.0;
- CrI3 J1: now ferromagnetic, −5.2 vs −4.5;
- Cr2O3 J1 with SCC: 34.2 vs 33.9;
- NiO J2 with SCC: 18.6 vs 19.0.

It degrades others:

- hcp Co and fcc Ni J1 overestimated about 2×;
- MnO J1 and J2 collapse;
- FeF2 J2 changes sign.

(These calibrated runs were made before the all-bands fix above. With
large calibrated I they are affected, so treat them as indicative.) The
reason is that J is controlled by the exchange splitting Δ = I·m, not by
m alone. For metals J grows roughly as Δ²; for superexchange in
insulators J ∝ t²/Δ decreases. So raising I to reach a moment can move J
either way. SCC has no effect on the elemental metals, as expected.
