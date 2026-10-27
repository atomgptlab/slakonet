# Magnetism in SlaKoNet: spin-polarized and non-collinear tight binding with learned parameters

*Technical report, 2026-10-27 (final).*

This report covers four things:

- the spin-polarized (collinear) and non-collinear (spin–orbit) extensions of
  SlaKoNet;
- the interface to TB2J for exchange constants, Dzyaloshinskii–Moriya
  interactions (DMI) and magnons;
- the data and the training used to fit the magnetic part, and to try to
  retrain the Slater–Koster parameters for magnetism;
- the measured performance, the known limits, and examples of use.

The user guide [Magnetic exchange (TB2J)](guide/magnetic-exchange.md) gives
the short version. This page gives the equations, the routines and the data.

## 1. Summary

SlaKoNet is a Slater–Koster tight-binding model whose two-centre integrals
are learned by automatic differentiation. It covers 65 elements and was
trained on more than 20,000 JARVIS-DFT materials (TBmBJ functional). Before
this work it was non-magnetic.

What was added:

1. **A collinear spin-polarized model.** It takes the SlaKoNet H(k) and S(k),
   and adds:
   - an atomic Stoner exchange;
   - an orbital-resolved Dudarev +U on the 3d shell;
   - periodic self-consistent charges (SCC).

   No user parameter is needed: U comes from a table with one value per
   ligand class, fitted to 129 experimental exchange constants.
2. **Exact real-space H(R) and S(R) for both spins, handed to TB2J in memory.**
   TB2J then gives the isotropic J_ij by the magnetic force theorem (LKAG),
   and the magnon spectrum from them.
3. **A non-collinear model with spin–orbit coupling (SOC).** It is built as a
   spinor Hamiltonian: the collinear exchange field rigidly rotated to an
   axis n, plus on-site λ L·S, with ζ values derived from NIST atomic
   spectra. The spinor H goes to TB2J's non-collinear solver along x, y and
   z, and the three runs are merged into the full exchange tensor: J_iso,
   symmetric anisotropic exchange and DMI.
4. **A training framework for magnetism.** It fits bounded corrections of
   the learned Slater–Koster integrals to spin-polarized PBE+U bands and to
   DFT magnetic energy differences. The data are our own VASP calculations,
   about 3,000 runs. The framework works and improves the bands, but it did
   not improve J against experiment in any of the six training series,
   including training without U and a learning curve. The base SlaKoNet
   parameters therefore remain the default, and the trained corrections are
   an experimental option (section 7).

Performance in one line: on 129–143 experimental couplings (leave one
compound out), about 42–45 % are within a factor of 2 and 65–72 % have the
right sign, for a cost of minutes per compound on one CPU (section 8).

## 2. What was changed in SlaKoNet

### 2.1 Modified routines (existing files)

| file | change |
|---|---|
| `slakonet/scc.py` | New periodic DFTB γ (`build_gamma_matrix_periodic`: Elstner's Slater short-range part plus an Ewald 1/r sum, `_slater_short_range`, `_lattice_translations`). New `AndersonMixer` (Pulay mixing). `atom_U_from_skf(..., legacy=False)` now takes the Hubbard U of the highest **occupied** shell. |
| `slakonet/main.py` | `SimpleDftb(..., scc_legacy=False)`: the corrected SCC is the default; `scc_legacy=True` restores the original behaviour. The cell is passed to the SCC solver for periodic systems. |
| `setup.py` | New console script `slakonet2J`. |
| `mkdocs.yml` | Documentation pages. |

### 2.2 New modules

| module | content |
|---|---|
| `slakonet/tb2j_interface.py` | Exact H(R)/S(R) extraction (`extract_real_space_hs`), overlap check (`check_overlap`), `SlakonetSpinModel` (container passed to TB2J; also stores the frozen potentials), Stoner model (`build_spin_model`), Stoner calibration (`calibrate_stoner`), TB2J driver (`gen_exchange_slakonet`), magnons (`compute_magnon_bands`), command line (`main`). |
| `slakonet/hubbard_spin.py` | Default model: Stoner + Dudarev U + SCC (`build_hubbard_model`). U tables (`ANION_U_EV`, `UNIVERSAL_U_EV`). Frozen potentials for training (`_frozen_potentials`). |
| `slakonet/magnetic_sites.py` | Automatic site classification: environment (ionic/metallic), ligand class, formal oxidation state (charge balance, structural rules, Alexandria oxidation-state atlas with a bond-length tie-break), d count, expected spin moment (`site_details`, `site_anion_class`). |
| `slakonet/stoner_tables.py` | Stoner I table fitted to experiment (the alternative `magnetic_model="stoner"`). |
| `slakonet/tb2j_ncl.py` | SOC constants (`SOC_ZETA_CM1`), on-site λ L·S in TB2J's spinor order (`soc_onsite`), spinor Hamiltonian (`spinor_hamiltonian`), non-collinear TB2J driver (`gen_exchange_slakonet_ncl`), three-axis merge (`gen_exchange_slakonet_soc`). |
| `slakonet/spin_training.py` | Training on spin-polarized DFT: VASP readers, dataset, frozen potentials, trainable corrections (`SKCorrections`), band and energy losses (`TrainingStructure`, `EnergyGroup`), and use of the result (`apply_corrections`, `maybe_corrected`). |
| `examples/train_spin_bands.py` | Training driver: stages `dataset`, `precompute`, `train` (torch.distributed), `evaluate`. |
| `examples/fit_stoner_table.py` | Fit and leave-one-out validation of the U / Stoner tables against experimental J (stages `dataset`, `precompute`, `fit`). |
| `examples/tb2j_benchmark.py` | Benchmark of 9 reference magnets. |
| `examples/collect_dft_exchange.py`, `collect_round4.py`, `analyze_dft_uscan.py`, `analyze_round4.py`, `analyze_spin_training.py`, `fit_with_dft.py` | DFT energy mapping, U-scan analysis and training analysis. |
| `tests/test_tb2j_interface.py`, `test_scc_periodic.py`, `test_magnetic_sites_hubbard.py`, `test_tb2j_ncl.py` | Tests. |

TB2J itself is not modified. Two TB2J behaviours are worked around from
SlaKoNet (section 4.3).

## 3. Collinear spin-polarized model

### 3.1 Real-space matrices

SlaKoNet assembles the Bloch sums

$$
H_{\mu\nu}(\mathbf k)=\sum_{\mathbf R}H_{\mu\nu}(\mathbf R)\,e^{2\pi i\mathbf k\cdot\mathbf R},\qquad
S_{\mu\nu}(\mathbf k)=\sum_{\mathbf R}S_{\mu\nu}(\mathbf R)\,e^{2\pi i\mathbf k\cdot\mathbf R}.
$$

The two-centre integrals vanish beyond the Slater–Koster cutoff, so only
image cells with $|n_i|\le n_{\max}$ contribute. Sampling H(k) on a
Γ-centred grid of $2n_{\max}+1$ points per direction therefore inverts the
sum **exactly** (`extract_real_space_hs`). The result is symmetrized so that
$H(-\mathbf R)=H(\mathbf R)^{T}$; the symmetrization removed a spurious
symmetry breaking in FeF₂.

S(k) is then checked to be positive definite on the k-mesh
(`check_overlap`). The run stops with the list of the offending orbitals if
it is not. This happens for some heavy alkali and alkaline-earth atoms
(K, Cs, Sr, Ba) with the present parameters.

### 3.2 Hamiltonian

The default model (`magnetic_model="hubbard"`, `hubbard_spin.build_hubbard_model`)
is, in Hartree:

$$
H^{\sigma}(\mathbf R)=H^{0}(\mathbf R)
+\tfrac12\,S(\mathbf R)\circ\big(\Delta\varepsilon_\mu+\Delta\varepsilon_\nu\big)
-\sigma\,\delta_{\mathbf R,0}\,\mathrm{diag}\!\big(\tfrac12 I_{l}\,m_a\big)
+\delta_{\mathbf R,0}\sum_{a}U_a\big(\tfrac12\mathbb 1-n_a^{\sigma}\big),
$$

with σ = ±1. The terms are:

- **$H^0$:** the SlaKoNet Hamiltonian (learned integrals).
- **SCC:** $\Delta\varepsilon_\mu=\sum_b\gamma_{ab}\,\Delta q_b$ for an orbital μ
  on atom a. $\Delta q_b$ is the Mulliken charge excess, and γ is the
  periodic DFTB γ matrix (section 3.4).
- **Stoner:** $I_l$ is the atomic Stoner parameter of shell l
  (`slakonet.magnetism.DEFAULT_STONER_I`, about 0.024–0.037 Ha for 3d), and
  $m_a$ is the Mulliken moment of atom a. This term plays the role of the
  LSDA exchange and gives Hund's-rule high spin.
- **Dudarev +U:** acts on the full 5×5 d occupation matrix of each 3d site,

  $$
  n^{\sigma}_{a,mm'}=\sum_{\mathbf k}w_{\mathbf k}\sum_{b}f_{b}\;\mathrm{Re}\,\tfrac12\big(C^{*}_{mb}(SC)_{m'b}+(SC)^{*}_{mb}C_{m'b}\big)
  $$

  (`_occupation_matrices`), which makes it rotationally invariant. Occupied
  spin-orbitals move down by about U/2 and empty ones up by about U/2. The
  term can polarise a single hole (Cu d⁹), opens Mott-like gaps, and sets
  the occupied–empty splitting that controls superexchange, $J\sim t^2/U$.

The +U term is the DFT+U construction (LSDA + U), with the Stoner term in
the LSDA role.

**Choice of U.** U is chosen per ligand class of the site, which
`magnetic_sites.site_anion_class` finds from the structure (`u_table="anion"`,
the default):

| class | U (eV) |
|---|---|
| O | 2.737 |
| F | 9.189 |
| halide (Cl, Br, I) | 4.063 |
| chalcogenide (S, Se, Te) | 7.097 |
| metallic environment | 1.976 |
| single universal value (`u_table="universal"`) | 2.97 |

These values were fitted to experimental J (section 6.2). `hubbard_U`
accepts explicit values per element or per site.

### 3.3 Self-consistency

At every iteration the code:

1. diagonalizes $H^\sigma(\mathbf k)$ in the non-orthogonal basis (Cholesky
   of S);
2. finds a common Fermi level for both spins (Fermi smearing,
   kT = 0.025 eV);
3. updates $n^\sigma_a$, $m_a$ and $\Delta q_a$;
4. mixes all of them with Anderson (Pulay) mixing.

The SCF starts from the expected configuration of the formal ion
(`magnetic_sites.site_details`): high spin, or low spin when the geometry
says so. The signs of `initial_moments` set the collinear order. Occupation
eigenvalues are clipped to [0, 1]. For insulators the Fermi level is placed
at mid-gap.

`build_spin_model` gives the simpler Stoner-only model
(`magnetic_model="stoner"`). It uses a fitted per-site table of I
(`stoner_tables`), or I calibrated to a target moment (`calibrate_stoner`).
Equivalent atoms are symmetrized (spglib) to avoid spurious
charge/spin disproportionation.

### 3.4 Charge self-consistency (corrected SCC)

The original SlaKoNet SCC had three problems:

- it took U from the highest-l shell of the basis, which gives U = 0 for O
  and F, because their basis carries an empty d shell;
- its Klopman–Ohno γ covered only the atoms inside the cell;
- it used linear mixing.

It did not converge for FeF₂ or MnF₂, even without magnetism. The new
scheme (`scc.py`) fixes all three:

- the Hubbard U of the highest occupied shell (p for O and F, d for the
  3d metals);
- Elstner's γ,

  $$
  \gamma_{ab}(r)=\frac1r-S(r;\tau_a,\tau_b),\qquad \tau=\tfrac{16}{5}U,
  $$

  where the short-range Slater part S is summed in real space and 1/r by an
  Ewald sum over all images;
- Anderson mixing.

`scc_legacy=True` restores the old scheme.

### 3.5 Frozen potentials

`build_hubbard_model` also returns the converged potentials
(`SlakonetSpinModel.potentials`):

- the SCC shift per atom;
- the Stoner potential per orbital;
- the U blocks per spin.

With them, the Hamiltonian of a structure can be rebuilt for any $H^0$
without a new SCF. This is the basis of the training scheme (section 7).

## 4. Interface to TB2J

### 4.1 Collinear exchange

The two converged spin channels (H↑(R), H↓(R), S(R) in eV) become HamiltonIO
`LCAOHamiltonian` objects and run through TB2J's `ExchangeCL2` in memory, so
no files are written in between. TB2J computes

$$
J_{ij}=\frac{1}{4\pi}\,\mathrm{Im}\int_{-\infty}^{E_F}\!dE\;
\mathrm{Tr}\big[\Delta_i\,G^{\uparrow}_{ij}(E)\,\Delta_j\,G^{\downarrow}_{ji}(E)\big],
\qquad \Delta_i=H^{\uparrow}_{ii}(0)-H^{\downarrow}_{ii}(0),
$$

on a complex contour (LKAG, magnetic force theorem). In this model the
exchange splitting is

$$
\Delta_i = I_l m_i + U_i\,(n_i^{\downarrow}-n_i^{\uparrow}),
$$

on-site and well defined.

The output is the usual TB2J folder: `exchange.out`, `TB2J.pickle`, and the
Multibinit, Vampire and TomASD files.

### 4.2 Magnons

`compute_magnon_bands` gives the linear spin-wave spectrum from the TB2J
exchange. Our figures use a paraunitary (Colpa) diagonalization; see
section 4.3.

### 4.3 TB2J behaviours worked around

1. **Band cut.** TB2J's Green's function keeps only the bands below
   E_F + 5.1 eV. Superexchange needs the empty bands, so J then depended on
   where E_F sits in the gap (MnO J₁: −0.00, 0.16 or 0.84 meV). The
   interface keeps all bands (`keep_all_bands=True`; TB2J is restored on
   exit) and puts E_F at mid-gap. MnO J₁ is then 0.34 meV for any E_F.
2. **Magnon solver.** TB2J shifts H(k) when it is not positive definite,
   which opens a fake gap at the Goldstone mode. We use a paraunitary
   diagonalization instead.

### 4.4 Conventions

TB2J writes $E=-\sum_{i\ne j}J_{ij}\,\mathbf e_i\cdot\mathbf e_j$ with unit
spins and every pair counted twice, so J > 0 is ferromagnetic. The tables in
this report use $JS^2=-2J_{\rm TB2J}$ per bond, so that J > 0 is
antiferromagnetic, as in our experimental database.

## 5. Non-collinear model with spin–orbit coupling

### 5.1 Spinor Hamiltonian

From a converged collinear model, `spinor_hamiltonian` builds, in TB2J's
interleaved basis (orb₁↑, orb₁↓, orb₂↑, …):

$$
H(\mathbf R)=\bar H(\mathbf R)\otimes\mathbb 1_2+\Delta(\mathbf R)\otimes(\mathbf n\cdot\boldsymbol\sigma)
+\delta_{\mathbf R,0}\,H_{\rm SOC},
$$

$$
\bar H=\tfrac12(H^{\uparrow}+H^{\downarrow}),\qquad
\Delta=\tfrac12(H^{\uparrow}-H^{\downarrow}),\qquad
S(\mathbf R)\otimes\mathbb 1_2 .
$$

Here **n** is the spin axis: the collinear exchange field is rotated
rigidly to n. The SOC term is on-site,

$$
H_{\rm SOC}=\sum_{a,l}\zeta_{a,l}\;\mathbf L\cdot\mathbf S\quad(\mathbf S=\boldsymbol\sigma/2),
$$

on every p and d shell (`soc_onsite`, built from the `slakonet.soc` L·S
blocks). The Fermi level is recomputed after SOC is added.

### 5.2 SOC constants

The constants are `SOC_ZETA_CM1`, and `lambda_soc` overrides them.

- **3d metals:** free-ion M²⁺ ζ₃d derived from the measured fine structure
  of the ground term in the NIST Atomic Spectra Database. A Landé-interval
  fit $E(J)=E_0+\tfrac{\lambda}{2}J(J+1)$ gives $\zeta=2S|\lambda|$.

  | ion | ζ₃d (cm⁻¹) |
  |---|---|
  | Ti | 120 |
  | V | 167 |
  | Cr | 230 |
  | Mn | 312 (interpolated: the ⁶S ground term has no fine structure) |
  | Fe | 411 |
  | Co | 534 |
  | Ni | 650 |
  | Cu | 829 |

  All agree with Griffith's tables within 3 %.
- **Halogens:** $\zeta_{np}=2\Delta/3$ from the ²P₃/₂–²P₁/₂ splittings of the
  neutral atoms: F 269, Cl 588, Br 2457, I 5069 cm⁻¹.

### 5.3 Exchange tensor

TB2J's `ExchangeNCL` runs with **n** along x, y and z
(`gen_exchange_slakonet_soc`). TB2J's merger then combines the transverse
components into the full tensor:

$$
E=-\sum_{i\neq j}\Big[J^{\rm iso}_{ij}\,\mathbf e_i\cdot\mathbf e_j
+\mathbf e_i\cdot J^{\rm ani}_{ij}\,\mathbf e_j
+\mathbf D_{ij}\cdot(\mathbf e_i\times\mathbf e_j)\Big].
$$

The scheme is one-shot, following the force theorem, as in TB2J's split-SOC
workflow. SOC is not made self-consistent, and there are no two-centre SOC
integrals.

### 5.4 Checks

- With SOC = 0, the collinear J is reproduced exactly and J_ani = D = 0.
- NiO: D = 0 on every bond with an inversion centre.
- CrI₃: D = 0 on the centrosymmetric nearest-neighbour bond.
- α-Fe₂O₃, against Dannegger et al., PRB 107, 184426 (2023):
  - same DMI shells and sign pattern, D = 0 on centrosymmetric bonds and
    $\mathbf D_{ij}=-\mathbf D_{ji}$;
  - |D_z| is 0.60 against 0.20 meV for the J₃ bond, and 0.35 against
    0.23 meV for J₄;
  - J₃ = 36.4 meV against 32.0 (INS), and J₄ = 27.4 against 25.0.

## 6. Data

### 6.1 Experimental exchange constants

The experimental set is a curated magnetic database (ExpMagDB:
spin-wave fits of inelastic neutron scattering, with magnetic structures
from MAGNDATA). Each compound is computed in its experimental magnetic
order.

- **Validation set:** 70 compounds, of which 143 couplings have
  |JS²| ≥ 1 meV.
- **Earlier fits:** a 62-compound subset with 129 couplings.

Bonds are matched by distance and by the pair of magnetic sublattices.

### 6.2 Fitting U to experiment

`examples/fit_stoner_table.py` works in three stages:

1. **dataset:** compounds, orders and couplings.
2. **precompute:** J for every compound on a U grid of 0.54–5.71 eV (8
   points), about 700 TB2J calculations. Every result is cached.
3. **fit:** one U per group, by minimizing a symmetric log-ratio loss,
   $\sum[\ln(|J_{\rm calc}|/|J_{\rm exp}|)]^2$, with a sign penalty. The
   earlier relative-error loss pushed every J towards zero.

Validation is leave one compound out: the U used for a compound is fitted
without it. A coupling with no physical solution counts as J = 0.

### 6.3 Our DFT calculations (VASP, PBE+U)

All runs are collinear, spin-polarized static calculations. Settings:

- VASP 6.4, PAW PBE (PP 64);
- 520 eV cutoff, `PREC = Accurate`, `LASPH`;
- Gaussian smearing 0.05 eV;
- k-mesh from the pymatgen MP density;
- `EDIFF` 10⁻⁶ in round 1 and 10⁻⁵ later;
- Dudarev U (`LDAUTYPE 2`) on the 3d species.

| round | content | runs |
|---|---|---|
| 1 | 74 compounds, Materials Project U. 7–9 collinear configurations each: FM, the experimental ground state, and model states chosen to separate the exchange shells. | 2,141 (rounds 1 and 2 together) |
| 2 | Extra configurations (60 compounds), and a U-scan of 18 oxides and fluorides at U_eff = 0, 2.5, 5 and 7.5 eV plus the MP U. | (see round 1) |
| 4 | U-scan at U_eff = 0, 3, 6 and 9 eV of 33 compounds (halides, chalcogenides and thiophosphates, Co and Cu compounds, metals). 25 compounds reuse the round-1 configurations; 8 new supercells (La₂CuO₄, Li₂CuO₂, Sr₂CuO₂Cl₂, LaMnAsO, NaMnO₂, NiWO₄, CrSb, FeSn) have 8 configurations from a D-optimal design. | 853 (834 kept) |

A configuration is kept if:

- VASP converged;
- the collinear MAGMOM pattern survived;
- every magnetic |m| > 0.15 μB;
- |m| is within 0.5 μB of the median of its batch, which rejects
  spin-state changes.

DFT exchange constants come from energy mapping
(`collect_dft_exchange.py`, `collect_round4.py`). This uses one energy
offset per batch, and shells are added while the design matrix stays well
conditioned.

The OUTCAR files also hold the eigenvalues of both spins at the irreducible
k-points. These are the band data used in training.

### 6.4 What the DFT data showed

- **DFT itself:** PBE+U at the MP U, compared with experiment, gives 33/57
  couplings within ×2 (median ratio 1.29).
- **Oxides:** J(U) has the same slope in DFT and in the TB model
  (d ln|J|/d ln U is −0.65 for DFT and −0.55 for TB). TB at U ≈ 3 eV
  reproduces DFT at U = 5 eV.
- **Fluorides:** TB needs U ≈ 12.5 eV to reproduce DFT. This is a SlaKoNet
  F-parameter error (F 2p level or M–F hopping), not a U effect. It is why
  the default U depends on the ligand.
- **Round 4:** the DFT U that best reproduces experiment is ≈ 3 eV for O,
  halides and chalcogenides, and ≈ 6 eV for metals. Translated into TB U,
  it confirms the O value. For the other classes it does not beat the
  defaults.
- **Unreliable references:** high-spin Fe²⁺ and Co²⁺ (FeO, CoO, FeCl₂,
  FePS₃, CoPS₃) are not reliable DFT+U references, because J changes sign
  with U through orbital-state changes. Cuprates lose the Cu moment at
  U = 0.

## 7. Training the Slater–Koster parameters for magnetism

### 7.1 Goal and design

The base SlaKoNet integrals were learned on non-magnetic TBmBJ data. The
aim here was to correct them, mainly the metal–ligand pairs, so that the
spin-polarized model reproduces spin-polarized DFT. The cost of a full
retraining is avoided by fitting **bounded corrections** on top of the
learned integrals (`SKCorrections`). Each Hamiltonian integral (pair, shell
pair, bond type) gets a scale and a decay correction:

$$
h(r)=h_0(r)\,\exp\!\big[a-b\,(r-r_{\rm ref})\big],\qquad
a=0.4\tanh u,\quad b=0.3\tanh v\ \text{bohr}^{-1},\quad r_{\rm ref}=4\ \text{bohr},
$$

with the parameters tied over the two orientations of the pair.

- On-site levels move by $D\tanh w$, with D = 2 eV, or 0.5 eV in one
  ablation.
- The overlap is never changed, so S(k) stays positive definite.
- A small L2 penalty is applied to u, v and w.
- SlaKoNet's spline interpolator (scipy) does not carry gradients, so the
  corrections are applied to the interpolated values through a feed wrapper
  (`build_feeds`, `_ScaledInterp`).
- The trained model is baked into a normal SlaKoNet model with
  `apply_corrections`, or applied from the environment variable
  `SLAKONET_SK_CORRECTIONS` (`maybe_corrected`).

### 7.2 Frozen-potential scheme

A spin-polarized SCF inside every gradient step would be too expensive.
Instead:

1. **Precompute.** The +U model of each training structure is converged
   once per round, and its potentials are stored (section 3.5). A structure
   is usable if:
   - the SCF converged;
   - the TB moments have the DFT signs;
   - |m_TB − m_DFT| < 1 μB.
2. **Train.** With the potentials fixed, the bands are a differentiable
   function of the corrections:

   $$
   H^{\sigma}(\mathbf k;\theta)=H^{0}(\mathbf k;\theta)+\tfrac12 S(\mathbf k)\circ(s_i+s_j)\pm\mathrm{diag}(V)+U^{\sigma}\text{ blocks}.
   $$

   The generalized eigenproblem uses a fixed Cholesky factor of S(k), since
   S is not trained.
3. **Rounds.** A new round recomputes the potentials with the corrections of
   the previous one. Two or three rounds make the scheme self-consistent; in
   practice the corrections hardly change after the first round.

This is checked: the frozen bands reproduce the converged model to 0.35 meV.

### 7.3 Losses

**Band loss.** The RMSE of the matched band energies of both spins, after
one rigid shift common to both spins. It uses:

- the top valence bands within 8 eV of E_F;
- the empty bands up to E_F + 3 eV, with weight 0.5.

**Magnetic energy-difference loss** (`EnergyGroup`, weight `--w-energy`).
For each DFT batch, the potential of the reference (FM) configuration is
rotated rigidly onto the other configurations: the Stoner V sign is flipped
and the U up/down blocks are swapped on the reversed atoms. This is the
magnetic force theorem. The band energy at fixed electron number (Fermi
smearing, kT = 0.05 eV) then gives

$$
\Delta E_{\rm TB}^{(c)}=E_{\rm band}^{(c)}-E_{\rm band}^{\rm (FM)},
$$

which is compared with the DFT total-energy difference per magnetic atom.
This term carries the exchange information directly. Two options make it
more robust:

- a Huber loss (`--huber-delta`);
- excluding compounds with orbital-state changes (`--exclude-energy`).

The total loss is

$$
L=\frac{1}{N_s}\sum_s L^{\rm bands}_s+w_E\,\frac{1}{N_g}\sum_g L^{\Delta E}_g+\lambda\,\|\theta\|^2 .
$$

The model is selected on the held-out compounds, by holdout band RMSE plus
$w_E\cdot$ holdout ΔE.

### 7.4 Training data

| | |
|---|---|
| configurations | 238 |
| compounds | 61 |
| configurations per compound | up to 4: FM, the ground state and two more |
| DFT U | per compound, the batch with U_eff in [2.5, 5.5] eV |
| held out | 40 configurations of 11 whole compounds (BiFeO₃, FeO, KCoF₃, LaFeO₃, MnBi, MnPS₃, MnS, MnTe, NiO, NiWO₄, RbMnCl₃) |
| usable after the moment check | 167 configurations |

| ligand class | compounds |
|---|---|
| O | 34 |
| chalcogenide | 8 |
| halide | 7 |
| F | 7 |
| metallic | 5 |

### 7.5 Training runs (HPC cluster, CPU)

The jobs ran under Slurm on 64 cores per job, using torch.distributed (gloo)
with whole DFT batches assigned to each rank. Each epoch takes about 55 s,
and a round of 200 epochs about 3 h. Every series has three rounds plus a J
validation on the 70-compound experimental set (700 TB2J calculations).

| series | settings |
|---|---|
| r | bands only, all 3d-metal pairs, on-site ±2 eV, lr 0.02 |
| a | as r, with on-site ±0.5 eV and a band window of E_F ± 3 eV |
| e | bands + energy loss (w_E = 30) |
| f | bands + energy loss; only metal–ligand pairs trained (`--pair-mode ionic`); on-site shifts only on ligands; Huber 30 meV; energy-loss exclusions (FeO, CoO, FeCl₂, FePS₃, FePSe₃, CoPS₃, LaVO₃); lr 0.01 |

### 7.6 Results

Bands (mean RMSE, eV, the 156 structures usable in every round):

| | base | e2 | f2 |
|---|---|---|---|
| train | 0.514 | 0.311 | 0.366 |
| holdout | 0.664 | 0.497 | 0.623 |

Series r2 was measured on a slightly different set (167 structures):
0.52 → 0.24 eV in training and 0.68 → 0.35 eV on held-out compounds.

Magnetic energy differences against DFT (meV per magnetic atom, excluded
compounds removed): with series f the error drops from 57 to 20 meV on the
training compounds and from 89 to 62 meV on the held-out ones.

Exchange against experiment (143 couplings, leave one compound out, U
refitted per ligand class for each parameter set):

| parameters | within ×2 | sign | MAE (meV) | no solution | best U per compound (×2) |
|---|---|---|---|---|---|
| **base SlaKoNet** | **60** | **93** | 8.3 | 24 | 81 |
| r2 | 44 | 83 | 12.2 | 40 | 81 |
| a2 | 47 | 89 | 9.6 | 36 | 83 |
| e2 | 53 | 84 | 9.1 | 42 | 76 |
| f2 | 49 | 89 | **7.8** | 39 | 79 |

The same results by class (within ×2 / N):

| class | N | base | e2 | f2 |
|---|---|---|---|---|
| O | 61 | 26 | **33** | 23 |
| metallic | 40 | 14 | 6 | **15** |
| chalcogenide | 21 | **9** | 5 | 3 |
| halide | 10 | **4** | 3 | 3 |
| F | 11 | **7** | 6 | 5 |

### 7.7 What we learned

1. **Band fitting works.** The band error halves, also on held-out
   compounds.
2. **Better DFT bands do not mean better J against experiment.** The
   trained M–O hoppings grow by about 30 %, which matches the PBE+U band
   widths. Superexchange, $J\propto t^2/U$, then grows by about 1.8, and the
   refitted U of O falls to 1 eV.

   One reason is that the model is trained to reproduce DFT+U at
   U_DFT ≈ 3–5 eV, while it carries its own, different U. The mismatch is
   absorbed into the hoppings: U is counted twice. Another reason is that
   the DFT reference itself is limited (33/57 within ×2).
3. **The energy loss helps where the DFT data are clean.** The Fe³⁺
   ferrites, RbNiF₃ and MnTe improve, the U values become physical
   (F: 15 → 4.7 eV), and the oxides improve with e2 (26 → 33).
4. **Training only metal–ligand pairs protects the metals** (series f), but
   it gives up the oxide gain.
5. **Decision:** the base SlaKoNet parameters remain the default. The
   trained corrections are available as an option (`SLAKONET_SK_CORRECTIONS`).

### 7.8 Final experiments: training without U and learning curve

Two last experiments tested the two explanations of section 7.7.

- **Series u (no double counting).** Training on PBE bands at U = 0, with the
  TB model also without U (`--tb-u zero`); the U table is refitted
  afterwards. Data: 236 configurations of 60 compounds, the same held-out
  compounds. Without U, 86 configurations of Ni, Co and Fe²⁺ compounds lose
  their moment in the TB model (PBE keeps 1–2 μB) and are excluded
  automatically.
- **Learning curve.** Series f with 50 % and 75 % of the training compounds
  (`--train-frac`), compared with 100 % (f0). This tests whether more DFT data
  of the same kind would help.

Exchange against experiment (143 couplings, leave one compound out; all
models evaluated together, so the base row differs slightly from section 7.6):

| parameters | within ×2 | sign | MAE (meV) | refitted U_O (eV) |
|---|---|---|---|---|
| **base SlaKoNet** | **60** | **92** | 8.4 | 2.5 |
| u2 (U = 0 training) | 51 | 85 | 8.4 | 4.1 |
| p50 (50 % of data) | 48 | 83 | 8.4 | 2.1 |
| p75 (75 % of data) | 46 | 87 | 7.6 | 2.2 |
| f0 (100 % of data) | 52 | 90 | 8.2 | 1.1 |

Bands:

- series u, against PBE at U = 0 (123 structures): 0.62 → 0.42 eV in
  training and 0.79 → 0.59 eV on held-out compounds;
- learning curve, against DFT+U (167 structures): 0.40 / 0.40 / 0.38 eV in
  training and 0.63 / 0.64 / 0.61 eV held out, at 50 / 75 / 100 %.

Conclusions:

1. **The double counting of U is real.** Without U in the training, the
   refitted U of O no longer collapses (4.1 eV instead of 1.0 eV). But J does
   not improve: 51 against 60 within ×2.
2. **More DFT data of the same kind does not help J.** The bands saturate at
   50 % of the data, and J stays below the base model at every size.
3. **Fitting bands, or magnetic energy differences, does not transfer to the
   exchange constants.** J depends on details near the Fermi level and on the
   metal–ligand covalency, and a band RMSE over a wide window does not
   constrain them.

### 7.9 Status

The magnetism retraining is closed. The base SlaKoNet parameters with the
ligand-class U table are the supported model. The training code stays in
the package for reuse: `slakonet/spin_training.py` and
`examples/train_spin_bands.py`. The trained corrections are an experimental
option (`SLAKONET_SK_CORRECTIONS`).

The only route left open is to use J itself, or large-weight magnetic energy
differences, as the main training target. That would be a separate project.

## 8. Performance

### 8.1 Accuracy against experiment (default model)

Leave one compound out:

| model | within ×2 | sign | MAE |
|---|---|---|---|
| **Stoner + U per ligand class + SCC (default)** | 57/129 (60/143) | 88/129 (93/143) | 9.4 meV (8.3) |
| Stoner + universal U = 2.97 eV | 51/129 | 94/129 | 9.4 meV |
| U per element (free) | 39/131 | 84 | – |
| Materials Project U, no fit | 44/131 | 76 | – |
| Stoner I table fitted to J | 28/124 | 83/124 | 10.5 meV |
| atomic Stoner I | 20/124 | 45/124 | 11.7 meV |
| *reference: DFT (PBE+U, MP U)* | *33/57* | – | – |

With the best U chosen compound by compound, an upper bound, 69/131 (81/143)
are within ×2. That is the ceiling of this TB + U description.

Benchmark examples (JS² in meV, J > 0 AFM):

| system | coupling | SlaKoNet | experiment |
|---|---|---|---|
| NiO | J₂ | 21.9 | 19.0 |
| NiO | J₁ | −0.3 | −1.4 |
| α-Fe₂O₃ | J₃ | 36.4 | 32.0 |
| α-Fe₂O₃ | J₄ | 27.4 | 25.0 |
| hcp Co | J₁ | −23 to −29 | −27.5 (LKAG) |

### 8.2 Known limits

These come from the Slater–Koster parameters; no value of U fixes them:

- 90° ferromagnetic superexchange: Li₂CuO₂ J₁ ≈ 0 against −4.6 meV.
- CrX₃: ferromagnetic J₁ too weak.
- MnS: 180° J₂ too large (20–27 against 6.6 meV).
- bcc Fe: J₁ too large (−61 to −67 against −39 meV).
- fcc Ni: moment 0.3 μB against 0.6 μB.
- Fluorides: need a large U.
- No lanthanide parameters, and a non-positive-definite overlap for K, Cs,
  Sr and Ba.
- DMI along long ligand paths is too small (CrI₃ second neighbours:
  0.008 against 0.31 meV), because the TB J₂ itself is too small.

### 8.3 Cost

| task | cost |
|---|---|
| collinear J of a 10–30-atom cell (SCF + TB2J) | a few CPU-minutes |
| J validation (70 compounds × 10 U values, 700 calculations) | about 40–60 min on 64 cores |
| SOC tensor | three non-collinear TB2J runs with a doubled basis, about 3–8× the collinear cost |
| training forward + backward per structure | about 0.5 s (compared with 35–60 s with SlaKoNet's default trainer, which computes forces) |

## 9. How to use it

### 9.1 Installation

```bash
pip install -e ./slakonet
git clone https://github.com/mailhexu/TB2J.git
pip install --no-deps -e ./TB2J        # TB2J lists `pypao`, which is not on PyPI
pip install "HamiltonIO>=0.3.8" sympair supercellmap tomli tomli-w pathos \
            tqdm pyyaml "sisl>=0.16.2" progressbar2
```

### 9.2 Collinear exchange and magnons (NiO, AFM-II)

```python
from ase.io import read
from slakonet.optim import default_model
from slakonet.tb2j_interface import gen_exchange_slakonet, compute_magnon_bands

atoms = read("POSCAR")                      # NiO AFM-II cell, 2 Ni + 2 O
exchange, sm = gen_exchange_slakonet(
    atoms, magnetic_elements=["Ni"], model=default_model(),
    initial_moments=[2, -2, 0, 0],          # only the signs matter: they set the order
    scf_kmesh=(6, 6, 6), kmesh=(6, 6, 6), Rcut=6.0,
    output_path="TB2J_NiO")                 # default: Stoner + U(ligand class) + SCC
print(sm.moments, sm.efermi, sm.converged)  # always check the moments
compute_magnon_bands("TB2J_NiO", kpath="GXWKGL")
```

Command line:

```bash
slakonet2J POSCAR --elements Ni --magmoms 2 -2 0 0 \
           --scf-kmesh 6 6 6 --kmesh 6 6 6 --rcut 6 --magnon --kpath GXWKGL
```

### 9.3 Spin-polarized bands only

```python
from slakonet.hubbard_spin import build_hubbard_model

import numpy as np
from scipy.linalg import eigh

sm = build_hubbard_model(atoms, default_model(), initial_moments=[2, -2, 0, 0],
                         kmesh=(6, 6, 6))           # u_table="anion" by default
kpts = np.linspace([0, 0, 0], [0.5, 0.5, 0.5], 50)  # reduced coordinates
Hup, Hdn, S = sm.bloch(kpts)                        # eV, [Nk, N, N]
bands_up = np.array([eigh(Hup[i], S[i], eigvals_only=True) for i in range(len(kpts))])
bands_dn = np.array([eigh(Hdn[i], S[i], eigvals_only=True) for i in range(len(kpts))])
```

### 9.4 DMI and anisotropic exchange (SOC)

```python
from slakonet.tb2j_ncl import gen_exchange_slakonet_soc

merged, sm = gen_exchange_slakonet_soc(
    atoms, ["Fe"], initial_moments=m0,          # e.g. alpha-Fe2O3 in its INS order
    scf_kmesh=(5, 5, 5), kmesh=(5, 5, 5), Rcut=7.0,
    output_path="TB2J_Fe2O3_soc")               # axis_x, axis_y, axis_z + merged result

print(merged.dmi_ddict)                         # {(R, i, j): D vector (eV)}
```

A complete, runnable version (structure, magnetic order, table of J, J_ani
and D per shell, symmetry checks) is `slakonet/examples/fe2o3_soc_dmi.py`.

Checks and options:

- `soc_strength=0` must reproduce the collinear J with D = J_ani = 0.
- `lambda_soc={Z: {l: zeta_Ha}}` changes the SOC constants.

### 9.5 Other options

```python
gen_exchange_slakonet(atoms, ["Mn"], hubbard_U={"Mn": 4.0 / 27.211386})   # explicit U (Ha)
gen_exchange_slakonet(atoms, ["Ni"], magnetic_model="stoner",
                      target_moments={"Ni": 1.8})                         # Stoner, I calibrated
from slakonet.main import SimpleDftb
SimpleDftb(..., scc_legacy=True)                                          # original SCC
```

### 9.6 Trained Slater–Koster corrections (experimental)

```python
from slakonet.spin_training import apply_corrections, SKCorrections
model = apply_corrections(default_model(), SKCorrections.load("corr_f2.pt"))
```

Or, for the benchmark and fitting scripts:

```bash
export SLAKONET_SK_CORRECTIONS=/path/to/corr_f2.pt
```

### 9.7 Training on new DFT data

```bash
T=slakonet/examples/train_spin_bands.py
python $T dataset    --data DATA --sources "runs/*_U*" --holdout NiO MnO
python $T precompute --data DATA --round r0 --nproc 64          # --tb-u zero for U = 0 bands
python $T train      --data DATA --round r0 --nproc 64 --out DATA/corr_r0.pt \
                     --w-energy 30 --pair-mode ionic --onsite-elements O F Cl Br I S Se Te
python $T evaluate   --data DATA --round r0 --corrections DATA/corr_r0.pt
```

Each batch directory holds a `manifest.json` and one VASP run per magnetic
configuration (OUTCAR with both spins). On a cluster, run `precompute`
and `train` as Slurm jobs (`--nproc` ranks, torch.distributed with gloo).

### 9.8 Refitting the U table

```bash
F=slakonet/examples/fit_stoner_table.py
python $F dataset    --work W
python $F precompute --work W --nproc 8 --spin-model hubbard
python $F fit        --work W --spin-model hubbard --grouping anion   # LOO report
```

## 10. References

- SlaKoNet: K. Choudhary et al., learned Slater–Koster tight binding
  (atomgptlab/slakonet).
- TB2J: X. He, N. Helbig, M. J. Verstraete, E. Bousquet, Comput. Phys.
  Commun. 264, 107938 (2021).
- LKAG: A. I. Liechtenstein, M. I. Katsnelson, V. P. Antropov,
  V. A. Gubanov, J. Magn. Magn. Mater. 67, 65 (1987).
- DFTB SCC γ: M. Elstner et al., Phys. Rev. B 58, 7260 (1998).
- DFT+U: S. L. Dudarev et al., Phys. Rev. B 57, 1505 (1998).
- SOC constants: NIST Atomic Spectra Database; J. S. Griffith, *The Theory
  of Transition-Metal Ions* (1961).
- α-Fe₂O₃ DMI: T. Dannegger et al., Phys. Rev. B 107, 184426 (2023).
