# Formation energies

The **formation energy** measures how stable a compound is relative to
its constituent elements:

$$
E_\text{form} = \frac{1}{N}\left(E_\text{total} - \sum_i n_i\,\mu_i\right)
$$

where `E_total` is the compound's total energy, `n_i` the count of
element `i`, `μ_i` the per-atom chemical potential of element `i`, and
`N` the number of atoms.

## Bundled chemical potentials

SlaKoNet ships a set of per-element chemical potentials, available
through `default_mu()`:

```python
from slakonet.optim import default_mu

mu = default_mu()           # {"Si": ..., "O": ..., ...}
print(len(mu), "elements")
```

These are calibrated so that the elemental reference structures have a
formation energy of exactly zero, and that compound formation energies
are self-consistent with the model. `default_mu(full=True)` returns the
full record including calibration metadata.

## Computing a formation energy

```python
from ase.build import bulk
from slakonet.optim import default_model, default_mu
from slakonet.ase_calc import SlaKoNetCalculator
from jarvis.core.atoms import Atoms as JAtoms   # for composition

model = default_model().float()
# alpha=1.0 gives a physically meaningful total energy
calc = SlaKoNetCalculator(model, kpoints=(3, 3, 3), alpha=1.0,
                          compute_forces=False, compute_stress=False)
mu = default_mu()

atoms = bulk("GaAs", "zincblende", a=5.65)
atoms.calc = calc
e_total = atoms.get_potential_energy()

# count elements
from collections import Counter
comp = Counter(atoms.get_chemical_symbols())
n_atoms = len(atoms)

if all(el in mu for el in comp):
    ref = sum(n * mu[el] for el, n in comp.items())
    e_form = (e_total - ref) / n_atoms
    print(f"E_form = {e_form:.3f} eV/atom")
```

!!! note "Use `alpha=1.0`"
    The bundled chemical potentials are calibrated for `alpha=1.0`. Build
    the calculator with `alpha=1.0` when computing formation energies so
    energies are on the same footing. (`alpha` does not affect
    eigenvalues, so band gaps are unchanged.)

## Using your own chemical potentials

If you have calibrated chemical potentials for a different setup, supply
them directly — `default_mu()` is just a convenience. Any
`{element: mu_eV}` mapping works in the formula above. The screening
example accepts a user JSON via its `MU_JSON` setting.

## In a screening loop

The [`chipstb_bandgaps.py`](https://github.com/atomgptlab/slakonet/blob/main/slakonet/examples/chipstb_bandgaps.py)
example computes formation energies for every material it screens, in
the same pass as the band gaps — see
[Band-gap screening](bandgap-screening.md).

## Element coverage

`default_mu()` covers the elements SlaKoNet parameterizes. Compounds
containing elements outside that set cannot be evaluated — check
membership (`all(el in mu for el in comp)`) before computing, as in the
snippet above.
