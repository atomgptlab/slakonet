# Band structure

A band structure is the set of electronic eigenvalues plotted along a
path of high-symmetry **k**-points through the Brillouin zone. SlaKoNet
produces them directly from the trained tight-binding model.

## The quick way

```python
from ase.build import bulk
from slakonet.optim import default_model
from slakonet.ase_calc import SlaKoNetCalculator

calc = SlaKoNetCalculator(default_model().float())

si = bulk("Si", "diamond", a=5.43)
bs = calc.band_structure(si, path="GXWKGL", npoints=120,
                         savefig="si_bands.png")

print("band gap :", bs["gap"], "eV")
print("VBM / CBM:", bs["vbm"], bs["cbm"])
```

This writes `si_bands.png` and returns a dictionary.

## `band_structure()` arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `atoms` | — | the ASE `Atoms` object |
| `path` | `None` | high-symmetry path string, e.g. `"GXWKGL"`; `None` uses the ASE standard path for the cell |
| `npoints` | `80` | number of k-points along the path |
| `savefig` | `None` | if set, write a PNG to this filename |
| `emin`, `emax` | `-6`, `8` | y-axis window for the plot (eV) |

## The return value

`band_structure()` returns a dict:

| Key | Contents |
| --- | --- |
| `energies` | eigenvalues, shape `(n_k, n_band)`, eV, referenced to mid-gap |
| `kpts` | fractional k-points along the path |
| `labels` | high-symmetry point labels |
| `path` | the path string used |
| `gap`, `vbm`, `cbm` | band gap and band edges (eV) |

## Choosing the path

Leaving `path=None` lets ASE pick the standard path for the lattice —
convenient and always valid. To control it, pass a string of
high-symmetry point labels:

```python
# diamond / zinc-blende
calc.band_structure(si, path="GXWKGL", npoints=120)

# hexagonal
calc.band_structure(hexagonal_atoms, path="GMKGALH", npoints=150)
```

More k-points (`npoints`) give smoother curves; 20–40 is already enough
to locate the band gap, 100+ for a publication-quality figure.

## How many k-points do I need for the gap?

For the **band gap** specifically, the answer is "not many" — the
valence-band maximum and conduction-band minimum sit on the
high-symmetry lines, so even `npoints=20` brackets them accurately.
Larger `npoints` mainly improves the *look* of the bands between
symmetry points.

## Plotting it yourself

If you want full control of the figure, take `energies` and plot
directly:

```python
import matplotlib.pyplot as plt

bs = calc.band_structure(si, path="GXWKGL", npoints=120)
e = bs["energies"]                       # (n_k, n_band), eV
for band in range(e.shape[1]):
    plt.plot(e[:, band], lw=0.8)
plt.axhline(0.0, ls="--", c="k", lw=0.6) # mid-gap reference
plt.ylabel(r"$E - E_\mathrm{mid}$ (eV)")
plt.ylim(-6, 8)
plt.savefig("bands.png", dpi=200)
```

## A note on accuracy

Tight-binding band structures from SlaKoNet are accurate enough for
**screening and qualitative analysis** and reach a band-gap MAE of
0.74 eV against experiment. Like all tight-binding methods, absolute
band positions are less reliable than gaps and band *shapes*. For a
comparison of k-sampling schemes and their effect on the predicted gap,
see [Band-gap screening](bandgap-screening.md).
