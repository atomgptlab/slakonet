"""Dielectric function from TB eigenvalues via Kubo-Greenwood.

We use the Peierls / gradient-of-H approximation for momentum matrix elements:

    p_alpha_{mn}(k) ~ (m_e / hbar) * <m,k| dH(k)/dk_alpha |n,k>

For dense k-grids (Monkhorst-Pack), dH/dk is evaluated by finite differences
using H(k) returned by slakonet's own machinery. For a precomputed k-path this
is still approximate but adequate for smoke-testing and relative trends.

The imaginary part of the dielectric tensor is (in Ha atomic units, prefactor
written explicitly so we can convert):

    eps_2^{ab}(omega) = (4 pi^2 e^2 / (omega^2 m^2 V)) * sum_{k,m,n} (f_m - f_n)
                         * p^a_{mn} * p^b_{nm} * delta(E_n - E_m - hbar*omega)

with standard smearing for the delta. Real part via Kramers-Kronig.

This is a smoke-test / qualitative calculator; for production, dipole-gauge
corrections and non-local commutator terms should be added.
"""
from __future__ import annotations

import math

import torch

from slakonet.atoms import Periodic
from slakonet.slaterkoster import hs_matrix
from slakonet.utils import eighb


def _diagonalize_HS(H, S):
    Nk = H.shape[-1]
    evals, evecs = [], []
    for ik in range(Nk):
        hk = H[..., ik].to(torch.complex128)
        sk = S[..., ik].to(torch.complex128)
        e, c = eighb(hk, sk, scheme="chol")
        evals.append(e); evecs.append(c)
    return torch.stack(evals, dim=-1), torch.stack(evecs, dim=-1)


def _build_H_at_k(calc, kfrac):
    """Evaluate H(k), S(k) at an arbitrary fractional k using Periodic.

    Uses klines with start==end and N=2, then returns the first of the two
    (identical) k-point matrices. Avoids the MP-grid interpretation of the
    kpoints kwarg.
    """
    geom = calc.geometry
    k = torch.as_tensor(kfrac, dtype=torch.float64).flatten()
    klines = torch.tensor(
        [[[float(k[0]), float(k[1]), float(k[2]),
           float(k[0]), float(k[1]), float(k[2]), 2]]],
        dtype=torch.float64,
    )
    per = Periodic(geom, geom.cell, cutoff=calc.cutoff, klines=klines)
    H = hs_matrix(per, calc.basis, calc.h_feed)
    S = hs_matrix(per, calc.basis, calc.s_feed)
    if H.ndim == 4:
        H = H[0]; S = S[0]
    return H[..., 0], S[..., 0]  # Norb x Norb complex


def momentum_matrix_elements(calc, kfrac, dk=1e-3):
    """Finite-difference dH/dk_alpha at a fractional k-point.

    Returns (p_alpha, E_n, C_n), where:
        p_alpha: 3-list of [Nband, Nband] complex  (in Hartree/(2pi/a_lattice))
        E_n   : [Nband] real eigenvalues (Hartree)
        C_n   : [Norb, Nband] complex eigenvectors solving Hc = Esc at k
    The alpha index runs over fractional crystal directions.
    """
    H0, S0 = _build_H_at_k(calc, kfrac)
    e0, c0 = eighb(
        H0.to(torch.complex128), S0.to(torch.complex128), scheme="chol"
    )

    p = []
    kfrac = torch.as_tensor(kfrac, dtype=torch.float64).flatten()
    for a in range(3):
        kp = kfrac.clone(); kp[a] += dk
        km = kfrac.clone(); km[a] -= dk
        Hp, _ = _build_H_at_k(calc, kp)
        Hm, _ = _build_H_at_k(calc, km)
        dH = (Hp - Hm) / (2.0 * dk)
        # transform to eigen basis: p_{mn} = c_m^dag (dH) c_n
        dHe = c0.conj().T @ dH.to(torch.complex128) @ c0
        p.append(dHe)
    return p, e0, c0


def compute_dielectric(
    calc,
    kgrid=(4, 4, 4),
    omega_range_eV=(0.0, 10.0),
    n_omega=500,
    smearing_eV=0.1,
    occupations_threshold=1e-6,
    dk=1e-3,
):
    """Compute eps_2(omega) and eps_1(omega) (isotropic average of diagonal).

    Parameters
    ----------
    calc : SimpleDftb
        Need calc.h_feed, calc.s_feed, calc.basis, calc.geometry, calc.cutoff.
        Does NOT need calc.calculate() to have run.
    kgrid : (nx, ny, nz)
        Uniform Monkhorst-Pack grid.
    omega_range_eV : (w_min, w_max)
    n_omega : int
    smearing_eV : float
        Gaussian smearing of the delta(E-hw).
    """
    H2E = getattr(calc, "H2E", 27.211)
    nx, ny, nz = kgrid
    # MP grid, fractional
    kx = (torch.arange(nx, dtype=torch.float64) + 0.5) / nx - 0.5
    ky = (torch.arange(ny, dtype=torch.float64) + 0.5) / ny - 0.5
    kz = (torch.arange(nz, dtype=torch.float64) + 0.5) / nz - 0.5
    grid = torch.stack(
        torch.meshgrid(kx, ky, kz, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    Nk = grid.shape[0]
    w = 1.0 / Nk  # uniform weight

    # frequency grid (eV and Hartree)
    w_min, w_max = omega_range_eV
    omega_eV = torch.linspace(w_min, w_max, n_omega, dtype=torch.float64)
    omega_Ha = omega_eV / H2E
    sigma_Ha = smearing_eV / H2E

    # accumulate eps_2 tensor 3x3
    eps2 = torch.zeros(3, 3, n_omega, dtype=torch.float64)

    # determine occupancy: use current model's nelectron
    nelec = float(calc.nelectron.flatten()[0].item())
    # infer: integer-filled band count = nelec/2 (spin degeneracy)
    nfull = int(round(nelec / 2.0))

    cell = calc.geometry.cell
    if cell.ndim == 3:
        cell = cell[0]
    volume = float(torch.abs(torch.det(cell.to(torch.float64))).item())

    for ik in range(Nk):
        p, e, _c = momentum_matrix_elements(calc, grid[ik], dk=dk)
        Nb = e.shape[0]
        # occupations: 1 for first nfull, 0 after (sharp; good enough for insulators
        # and for metals gives qualitatively reasonable results).
        f = torch.zeros(Nb, dtype=torch.float64)
        f[:nfull] = 1.0

        # valence -> conduction
        for m in range(Nb):
            for n in range(Nb):
                df = f[m] - f[n]
                if abs(df) < occupations_threshold:
                    continue
                dE = (e[n] - e[m]).real.to(torch.float64)
                if dE <= 0:
                    continue
                delta = torch.exp(-0.5 * ((omega_Ha - dE) / sigma_Ha) ** 2) / (
                    sigma_Ha * math.sqrt(2.0 * math.pi)
                )
                for a in range(3):
                    for b in range(3):
                        val = (p[a][m, n] * p[b][n, m]).real
                        eps2[a, b] += (
                            w * df.item() * float(val.item()) / max(dE.item() ** 2, 1e-12)
                        ) * delta

    # prefactor: 4 pi^2 / V  (in Hartree atomic units, e=m=hbar=1)
    pref = 4.0 * math.pi ** 2 / volume
    eps2 = pref * eps2
    # isotropic average
    eps2_iso = (eps2[0, 0] + eps2[1, 1] + eps2[2, 2]) / 3.0

    # Kramers-Kronig: eps_1(w) = 1 + (2/pi) P int_0^inf w' eps_2(w')/(w'^2 - w^2) dw'
    dw = float(omega_Ha[1] - omega_Ha[0])
    eps1_iso = torch.ones_like(omega_Ha)
    for i, w_i in enumerate(omega_Ha):
        denom = omega_Ha ** 2 - w_i ** 2
        # avoid singular point
        mask = torch.abs(denom) > 1e-10
        integrand = torch.zeros_like(omega_Ha)
        integrand[mask] = omega_Ha[mask] * eps2_iso[mask] / denom[mask]
        eps1_iso[i] = 1.0 + (2.0 / math.pi) * torch.sum(integrand) * dw

    return {
        "omega_eV": omega_eV,
        "eps2": eps2,
        "eps2_iso": eps2_iso,
        "eps1_iso": eps1_iso,
        "volume_bohr3": volume,
        "kgrid": kgrid,
    }


def plot_dielectric(result, filename="dielectric.png"):
    import matplotlib.pyplot as plt

    w = result["omega_eV"].detach().cpu().numpy()
    e1 = result["eps1_iso"].detach().cpu().numpy()
    e2 = result["eps2_iso"].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(w, e1, label=r"$\varepsilon_1$")
    ax.plot(w, e2, label=r"$\varepsilon_2$")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$\varepsilon(\omega)$")
    ax.legend()
    ax.axhline(0, color="k", lw=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
