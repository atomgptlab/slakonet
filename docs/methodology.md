# How it works

This page explains the method behind SlaKoNet — enough to understand
what the model is and why it behaves the way it does.

## Slater-Koster tight binding

Tight binding describes a material's electronic structure with a small
set of atomic-orbital basis functions. In the **Slater-Koster (SK)**
formalism, the Hamiltonian and overlap matrix elements between two atoms
are written as a compact set of *two-centre integrals* — `ssσ`, `spσ`,
`ppσ`, `ppπ`, `sdσ`, … — each a function only of the interatomic
distance, rotated into the lab frame by simple geometric factors.

Solving the resulting generalized eigenvalue problem

$$
H(\mathbf{k})\,c = E\,S(\mathbf{k})\,c
$$

at each **k**-point yields the band structure. This is orders of
magnitude cheaper than a full DFT calculation, and the basis is small
and physically interpretable.

The classic limitation is the **parameters**: the SK integral tables and
on-site energies are hard to obtain, transfer poorly between chemical
environments, and have traditionally been fitted by hand to limited
data.

## What SlaKoNet learns

SlaKoNet is a **parameter-optimization framework**. It represents the SK
integral tables and on-site energies as differentiable functions and
**optimizes them with automatic differentiation** so that the
tight-binding band structures reproduce reference electronic-structure
data.

- **Reference data**: density-functional-theory band structures from the
  JARVIS-DFT database, at the Tran-Blaha modified Becke-Johnson (TBmBJ)
  level — a high-fidelity description of band gaps. The training set
  spans more than 20,000 materials.
- **Optimization**: every step of the pipeline — building `H(k)` and
  `S(k)` from the SK parameters, diagonalizing, comparing to the
  reference bands — is implemented in PyTorch, so gradients of the loss
  with respect to *every* parameter are available via autograd. The
  parameters are tuned by gradient descent.
- **Coverage**: the optimization is carried out jointly across 65
  elements, producing a single transferable parameter set rather than a
  per-system fit.

The outcome is a tight-binding model with the **speed and
interpretability of SK theory** but **accuracy informed by
high-fidelity DFT** across the periodic table.

## The pipeline at a glance

```
structure ──► neighbour list ──► Slater-Koster integrals
          ──► H(k), S(k)  ──► generalized eigensolve
          ──► eigenvalues ──► band structure / DOS / gap / Fermi level
                          └─► occupations ──► total energy
                                          └─► autograd ──► forces, stress
```

Because the whole pipeline is differentiable:

- **forces** are `−∂E/∂x` and **stress** is `∂E/∂(strain)`, obtained by
  autograd rather than finite differences;
- the model is itself trainable end-to-end — the same machinery used to
  *use* SlaKoNet is what is used to *optimize* it.

## Accuracy

Against experimental band gaps, SlaKoNet reaches a mean absolute error
of **0.74 eV** — better than standard GGA functionals (1.14 eV) — while
retaining tight-binding cost. As with any tight-binding method:

- **band gaps** and **band shapes** are reliable;
- **absolute band positions** are less reliable than relative ones;
- accuracy is best for the chemistries well represented in the training
  data.

See [Performance](performance.md) for benchmarks and
[FAQ](faq.md) for guidance on interpreting results.

## Self-consistent charges (SCC)

For systems where charge transfer matters, SlaKoNet supports a
self-consistent-charge correction (`use_scc=True` on the calculator).
This iterates the on-site potentials to self-consistency with the
Mulliken charges, at additional cost. The default (`use_scc=False`) is a
single non-self-consistent solve and is appropriate for most band-gap
and screening work.
