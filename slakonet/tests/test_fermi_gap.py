"""Fermi level placement for gapped and metallic spectra.

Two regressions are locked down here, both of which silently corrupted
every NEGF transport run on a semiconductor:

1. ``fermi_search`` applied ``kT / H2E``, i.e. it assumed the eigenvalues
   were in Hartree.  ``SimpleDftb`` hands it eigenvalues in **eV**, so the
   smearing that actually reached the occupancy function was 27.2x too
   small (0.025 eV -> 0.92 meV).

2. At that smearing the Fermi-Dirac occupancies clamp to exactly 0 and 1
   across a gap, so the electron count ``n(mu)`` is a flat plateau and the
   bisection root is degenerate.  The tie-break walked ``mu`` to the top of
   the plateau, pinning E_F to the conduction band minimum.  Transport then
   reported the gap as lying entirely *below* E = 0, and an insulator like
   hBN came out conducting at E_F.

Everything here is synthetic: no model, no download, milliseconds.
"""

import torch

from slakonet.fermi import fermi_search

KT = 0.025  # eV, what SimpleDftb passes


def _spectrum(vbm_top, cbm_bot, n_val=8, n_con=8, width=6.0):
    """A gapped spectrum in eV: n_val valence bands below vbm_top,
    n_con conduction bands above cbm_bot.  One k-point."""
    val = torch.linspace(vbm_top - width, vbm_top, n_val, dtype=torch.float64)
    con = torch.linspace(cbm_bot, cbm_bot + width, n_con, dtype=torch.float64)
    return torch.cat([val, con]).reshape(1, 1, -1), 2.0 * n_val


def _mu(eig, n_elec, **kw):
    return float(fermi_search(eig, n_elec, kT=KT, **kw).flatten()[0])


def test_gapped_fermi_is_midgap_not_band_edge():
    for vbm, cbm in [(-2.0, -0.6), (-4.0, -0.2), (-1.0, 2.8), (-0.5, -0.4)]:
        eig, ne = _spectrum(vbm, cbm)
        gap = cbm - vbm
        mu = _mu(eig, ne)
        if gap > 0.05:
            assert abs(mu - 0.5 * (vbm + cbm)) < 1e-6, (
                f"gap {gap:.3f} eV: mu={mu:.4f} is not mid-gap "
                f"({0.5*(vbm+cbm):.4f})"
            )
            # the specific old failure: E_F glued to the conduction edge
            assert abs(mu - cbm) > 0.4 * gap, (
                f"gap {gap:.3f} eV: mu={mu:.4f} is pinned to the CBM "
                f"({cbm:.4f}) -- the plateau tie-break has regressed"
            )


def test_fermi_lies_strictly_inside_the_gap():
    eig, ne = _spectrum(-2.0, -0.6)
    mu = _mu(eig, ne)
    assert -2.0 < mu < -0.6


def _asymmetric_metal():
    """Sparse levels below, a dense manifold just above -- deliberately not
    particle-hole symmetric, so mu genuinely depends on the smearing."""
    levels = torch.cat(
        [
            torch.tensor([-5.0, -4.0, -3.0, -2.0, -1.0], dtype=torch.float64),
            torch.linspace(0.0, 0.6, 12, dtype=torch.float64),
        ]
    )
    return levels.reshape(1, 1, -1)


def _count(eig, mu, kT=KT):
    """Electron count with the occupancy evaluated in eV -- the unit the
    eigenvalues are actually in."""
    return float(2.0 * (1.0 / (torch.exp((eig - mu) / kT) + 1.0)).sum())


def test_kT_is_in_the_eigenvalue_unit():
    """mu must satisfy the electron count when occupancies use kT in eV.

    This is what the Hartree conversion broke: it placed mu using an
    effective smearing of kT/27.211, so the count evaluated at the eV
    smearing came out wrong (10.4995 instead of 10 for this spectrum).
    """
    eig = _asymmetric_metal()
    for ne in (9.0, 10.0, 11.0, 12.0):
        mu = _mu(eig, ne)
        assert abs(_count(eig, mu) - ne) < 1e-6, (
            f"ne={ne}: mu={mu:.4f} gives count {_count(eig, mu):.4f}"
        )


def test_odd_electron_count_is_metallic_not_midgap():
    """An odd count is a half-filled band.  Band-index gap detection must
    not fire on it -- rounding n/2 would invent a gap and return mid-gap,
    losing an electron."""
    eig = _asymmetric_metal()
    mu = _mu(eig, 9.0)
    assert abs(_count(eig, mu) - 9.0) < 1e-6


def test_gap_branch_rejected_when_it_breaks_the_count():
    """Overlapping bands can show a band-index 'gap' that does not hold at
    every k.  The count guard must reject it and fall through to bisection."""
    eig = _asymmetric_metal()
    for ne in (10.0, 12.0):
        mu = _mu(eig, ne)
        assert abs(_count(eig, mu) - ne) < 1e-6


def test_return_shape_is_unchanged():
    eig, ne = _spectrum(-2.0, -0.6)        # gapped branch
    assert tuple(fermi_search(eig, ne, kT=KT).shape) == (1, 1)
    val = torch.linspace(-8.0, 0.5, 8, dtype=torch.float64)
    con = torch.linspace(-0.5, 4.0, 8, dtype=torch.float64)
    eig_m = torch.cat([val, con]).reshape(1, 1, -1)   # metallic branch
    assert tuple(fermi_search(eig_m, 16.0, kT=KT).shape) == (1, 1)
