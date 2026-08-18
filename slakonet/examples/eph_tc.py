"""Electron-phonon coupling and superconducting Tc from SlaKoNet.

Computes the isotropic Eliashberg function alpha^2 F(omega), the coupling
constant lambda, the logarithmic frequency omega_log, and the
Allen-Dynes/McMillan critical temperature.

Method -- frozen phonons in a supercell
---------------------------------------
Phonon wavevectors commensurate with a supercell fold to its Gamma
point, so *both* halves of the problem live in one cell and no q/k
bookkeeping is needed:

* The supercell force-constant matrix (phonopy, from SlaKoNet forces)
  diagonalises directly into 3*N_sc modes. Those modes ARE the union
  over all commensurate q of every phonon branch.
* Electron states of the supercell at a supercell k already contain the
  primitive k and k+q folded together, so a matrix element between two
  supercell states at the same k is exactly the k -> k+q coupling.

For each mode nu the Hamiltonian and overlap derivatives come from a
central difference along the mass-weighted mode coordinate, and the
non-orthogonal vertex is

    g_mn = <m| dH/dQ - eps_bar dS/dQ |n> * sqrt(hbar / 2 omega_nu)

with eps_bar = (eps_m + eps_n)/2. The dS/dQ term is not optional: the
basis moves with the atoms, and dropping it leaves a spurious coupling
proportional to the band energy.

Then

    a2F(w) = 1/N(Ef) sum_nu d(w-w_nu) sum_knm W_k |g|^2 d(e_m-Ef) d(e_n-Ef)
    lambda = 2 int a2F(w)/w dw,     Tc = Allen-Dynes(lambda, w_log, mu*)

with the mode sum divided by the number of primitive cells so lambda is
the per-cell, size-consistent quantity.

    python slakonet/examples/eph_tc.py --element Al --supercell 3
    python slakonet/examples/eph_tc.py --element Nb --mu-star 0.13

Validation (Al, fcc, 3x3x3 supercell = 27 q-points, 4^3 k-mesh)
--------------------------------------------------------------
    quantity      SlaKoNet    reference
    N(Ef)         0.27        ~0.4 /eV/cell/spin
    lambda        0.33        0.43
    omega_log     196 K       ~270 K
    Tc (mu*=0.1)  0.22 K      1.18 K (experiment)

lambda and omega_log land within ~25% of the accepted values, which for
a tight-binding model with no self-consistency is a reasonable place to
be. Tc is not: it depends exponentially on both, so a 25% shortfall in
each compounds into a factor of five. Read lambda and omega_log as the
meaningful outputs and Tc as a strongly amplified derived quantity.

Choosing the force model (--force-calc)
---------------------------------------
SlaKoNet's own forces leave most bcc transition metals dynamically
unstable -- Nb 20 imaginary modes of 81, Ta 32, Mo 12, V 8 -- while
alignn matpes_pbe gives zero for all of them, converged across 3^3,
4^3 and 5^3 supercells. So the hybrid is what makes anything past Al
and Cu computable at all.

It is not uniformly better, though. On Al, where both force models are
stable, swapping ONLY the phonon source (N(Ef) identical to four
digits) gives:

                    SlaKoNet ph.   matpes_pbe ph.   reference
    lambda              0.330          0.254           0.43
    omega_log           196 K          253 K          ~270 K
    max frequency      9.35 THz       7.35 THz        ~9.7 THz

lambda moves 23% -- the same order as the N(Ef) error, not a footnote.
And the two disagree in spectral *shape*, not scale: SlaKoNet has the
higher ceiling but more weight at low frequency, which raises lambda
(~1/omega^2) while lowering omega_log. No single stiffness factor
reconciles them.

Practical rule: use SlaKoNet's forces where they are stable (better
lambda there, and self-consistent); use the hybrid where they are not;
and carry ~25% uncertainty on lambda from the phonon source whenever
the hybrid is used.

What this does and does not include
-----------------------------------
The SlaKoNet Hamiltonian is fitted to DFT band structures, so dH/du
carries the screened deformation potential rather than a bare one --
which is what the vertex should be. But the calculation is non-SCC:
there is no self-consistent charge response to the displacement, so
polar and charge-transfer contributions are missing. Expect this to
work best for simple metals and to under-describe strongly polar or
strongly correlated superconductors. Treat absolute Tc with caution and
convergence (supercell size, k-mesh, smearing) as something to check
rather than assume -- ``--converge`` reports the smearing sensitivity.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import scipy.constants as const
import torch
from ase import Atoms as AseAtoms
from ase.build import bulk

from slakonet.ase_calc import SlaKoNetCalculator
from slakonet.optim import default_model

# Keep BLAS from oversubscribing against torch's pool (see solve_bands).
torch.set_num_threads(max(1, min(8, (os.cpu_count() or 8) // 2)))

HARTREE_EV = 27.211386245988
# 1 eV*s^2 expressed in amu*Angstrom^2 -- converts hbar/(2 omega) into the
# mass-weighted displacement units the mode coordinate uses.
EV_S2_TO_AMU_A2 = const.e / (const.atomic_mass * 1e-20)
THZ_TO_RAD_S = 2.0 * np.pi * 1e12
KELVIN_PER_EV = const.e / const.k

# np.trapezoid is NumPy >= 2.0; np.trapz is the older spelling and is
# removed in the newest. Bind whichever this environment actually has.
_trapz = getattr(np, "trapezoid", None) or np.trapz
# hbar*omega in eV for a frequency given in THz. a2F is built on an
# ENERGY axis: A_nu = sum|g|^2 dd / N(Ef) carries eV, so lambda =
# 2 sum A_nu / (hbar omega_nu) is only dimensionless if the denominator
# is an energy. Smearing on a THz axis instead silently divides by a
# number 242x too large.
THZ_TO_EV = 2.0 * np.pi * 1e12 * const.hbar / const.e


def build_phonons(prim, calc, supercell, distance=0.02):
    """Force constants of `prim` x `supercell` from SlaKoNet forces."""
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    ph_atoms = PhonopyAtoms(
        symbols=prim.get_chemical_symbols(),
        cell=prim.get_cell(),
        scaled_positions=prim.get_scaled_positions(),
    )
    phonon = Phonopy(ph_atoms, supercell_matrix=np.diag([supercell] * 3))
    phonon.generate_displacements(distance=distance)
    sups = phonon.supercells_with_displacements
    forces = []
    for i, sc in enumerate(sups, 1):
        a = AseAtoms(
            symbols=sc.symbols,
            cell=sc.cell,
            scaled_positions=sc.scaled_positions,
            pbc=True,
        )
        a.calc = calc
        t = time.time()
        forces.append(a.get_forces())
        print(
            f"    displacement {i}/{len(sups)}  {time.time() - t:.0f}s",
            flush=True,
        )
    phonon.forces = np.array(forces)
    phonon.produce_force_constants()
    return phonon


def supercell_gamma_modes(phonon):
    """Diagonalise the supercell force constants -> (freq_THz, evec, masses).

    These Gamma modes of the supercell are every branch at every q
    commensurate with it, which is exactly the mode sum a2F needs.
    """
    fc = phonon.force_constants  # (N, N, 3, 3) eV/Ang^2
    sc = phonon.supercell
    masses = np.array(sc.masses)
    n = len(masses)
    phi = np.transpose(fc, (0, 2, 1, 3)).reshape(3 * n, 3 * n)
    minv = 1.0 / np.sqrt(np.repeat(masses, 3))
    dyn = phi * minv[:, None] * minv[None, :]
    dyn = 0.5 * (dyn + dyn.T)
    w2, evec = np.linalg.eigh(dyn)  # eV/(Ang^2 amu)

    # eV/(Ang^2 amu) -> rad/s: E = amu Ang^2 / s^2 => 1 eV = EV_S2_TO_AMU_A2
    omega_rad = np.sqrt(np.abs(w2) * EV_S2_TO_AMU_A2) * np.sign(w2)
    return omega_rad / THZ_TO_RAD_S, evec, masses


def solve_bands(H, S, device="cpu"):
    """Generalised eigenproblem for every k at once -> (eps_eV, coeffs).

    Batched through torch rather than looping scipy per k-point. The
    per-k loop is not merely slower: torch's thread pool and LAPACK's
    fight over the same cores, and a 64-k solve that should take ~10 s
    spun for half an hour at 750% CPU. Batching sidesteps that entirely.

    S is positive definite (it is an overlap matrix), so reduce the
    generalised problem with a Cholesky factor instead of inverting:
    S = L L^H, A = L^-1 H L^-H, then c = L^-H y.
    """
    Ht = torch.as_tensor(np.asarray(H), dtype=torch.complex128, device=device)
    St = torch.as_tensor(np.asarray(S), dtype=torch.complex128, device=device)
    L = torch.linalg.cholesky(St)
    Linv_H = torch.linalg.solve_triangular(L, Ht, upper=False)
    A = torch.linalg.solve_triangular(L, Linv_H.mH, upper=False).mH
    A = 0.5 * (A + A.mH)
    w, y = torch.linalg.eigh(A)
    c = torch.linalg.solve_triangular(L.mH, y, upper=True)
    return (
        (w.real * HARTREE_EV).cpu().numpy(),
        c.cpu().numpy(),
    )


def gaussian(x, sigma):
    return np.exp(-((x / sigma) ** 2) / 2.0) / (sigma * np.sqrt(2 * np.pi))


def compute_a2f(
    calc,
    sc_atoms,
    freqs_thz,
    evecs,
    masses,
    n_prim_cells,
    kmesh,
    amplitude=0.01,
    sigma_el=0.20,
    sigma_ph=0.30,
    solver_device="cpu",
    omega_max_thz=None,
    n_omega=400,
    freq_floor_thz=0.15,
    progress=True,
):
    """Eliashberg function on a frequency grid, plus lambda and w_log."""
    H0, S0 = calc.get_HS(sc_atoms, kpoints=kmesh)
    eps0, c0 = solve_bands(H0, S0, device=solver_device)
    nk = eps0.shape[0]
    wk = np.full(nk, 1.0 / nk)
    ef = calc.get_fermi_level()

    # N(Ef) per primitive cell per spin, from the same smeared delta the
    # double-delta sum uses -- consistency matters more than the absolute
    # value here, since N(Ef) cancels between a2F and the sum.
    dl = gaussian(eps0 - ef, sigma_el)
    n_ef = float((dl * wk[:, None]).sum()) / n_prim_cells

    # Energy axis (eV). sigma_ph is given in THz for readability and
    # converted here so the smearing Gaussian integrates to 1 in eV.
    omega_max = (omega_max_thz or (freqs_thz.max() * 1.15)) * THZ_TO_EV
    sigma_ph_ev = sigma_ph * THZ_TO_EV
    grid = np.linspace(0.0, omega_max, n_omega)
    a2f = np.zeros_like(grid)

    pos0 = sc_atoms.get_positions()
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(masses, 3))
    n_modes = len(freqs_thz)
    used = 0
    for nu in range(n_modes):
        f_thz = freqs_thz[nu]
        if f_thz < freq_floor_thz:
            continue  # acoustic zero modes and any residual imaginary ones
        disp = (evecs[:, nu] * inv_sqrt_m).reshape(-1, 3)

        a = sc_atoms.copy()
        a.set_positions(pos0 + amplitude * disp)
        Hp, Sp = calc.get_HS(a, kpoints=kmesh)
        a.set_positions(pos0 - amplitude * disp)
        Hm, Sm = calc.get_HS(a, kpoints=kmesh)
        dH = (Hp - Hm) / (2 * amplitude) * HARTREE_EV
        dS = (Sp - Sm) / (2 * amplitude)

        omega_rad = f_thz * THZ_TO_RAD_S
        # <Q^2>_zp in mass-weighted units (amu^1/2 Ang)
        amp_zp = np.sqrt(
            const.hbar / const.e / (2 * omega_rad) * EV_S2_TO_AMU_A2
        )

        acc = 0.0
        for ik in range(nk):
            c = c0[ik]
            e = eps0[ik]
            d = gaussian(e - ef, sigma_el)
            # keep only states with weight at Ef; the double delta kills
            # everything else and the matrix product is O(n^3)
            sel = d > 1e-6 * d.max() if d.max() > 0 else np.zeros_like(d, bool)
            if sel.sum() == 0:
                continue
            cs = c[:, sel]
            ds = d[sel]
            es = e[sel]
            M = cs.conj().T @ dH[ik] @ cs
            Ms = cs.conj().T @ dS[ik] @ cs
            ebar = 0.5 * (es[:, None] + es[None, :])
            g = (M - ebar * Ms) * amp_zp
            acc += wk[ik] * float(
                (np.abs(g) ** 2 * ds[:, None] * ds[None, :]).sum()
            )
        a2f += acc / n_ef * gaussian(grid - f_thz * THZ_TO_EV, sigma_ph_ev)
        used += 1
        if progress and (used % 10 == 0 or nu == n_modes - 1):
            print(f"    mode {nu + 1}/{n_modes} (used {used})", flush=True)

    a2f /= n_prim_cells
    return grid, a2f, n_ef, used


def lambda_wlog(grid_ev, a2f):
    """(lambda, omega_log in eV) from a2F on an energy grid."""
    m = grid_ev > 1e-12
    w, f = grid_ev[m], a2f[m]
    lam = 2.0 * _trapz(f / w, w)
    if lam <= 0:
        return 0.0, 0.0
    wlog_ev = np.exp((2.0 / lam) * _trapz(f / w * np.log(w), w))
    return lam, wlog_ev


def allen_dynes_tc(lam, wlog_ev, mu_star=0.10):
    """McMillan/Allen-Dynes Tc in kelvin from lambda and omega_log in eV."""
    if lam <= 0:
        return 0.0
    denom = lam - mu_star * (1 + 0.62 * lam)
    if denom <= 0:
        return 0.0  # coupling too weak against the Coulomb pseudopotential
    wlog_K = wlog_ev * KELVIN_PER_EV
    return wlog_K / 1.2 * np.exp(-1.04 * (1 + lam) / denom)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--element", default="Al")
    ap.add_argument(
        "--jid",
        default=None,
        help="JARVIS-DFT id; overrides --element/--structure/--a and uses "
        "the database structure directly (needed for compounds).",
    )
    ap.add_argument("--structure", default="fcc")
    ap.add_argument("--a", type=float, default=None)
    ap.add_argument("--model", default="slakonet_v1a_full")
    ap.add_argument("--supercell", type=int, default=3)
    ap.add_argument("--kmesh", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mu-star", type=float, default=0.10)
    ap.add_argument(
        "--force-calc",
        default="slakonet",
        choices=["slakonet", "alignn"],
        help="which model supplies the FORCES for the force constants. "
        "The electronic vertex always comes from SlaKoNet. SlaKoNet's own "
        "forces leave most bcc transition metals dynamically unstable "
        "(Nb: 20 imaginary modes of 81), while alignn matpes_pbe gives "
        "zero at every supercell size -- so 'alignn' is what makes "
        "anything past Al and Cu computable.",
    )
    ap.add_argument("--ff-model", default="matpes_pbe")
    ap.add_argument("--disp", type=float, default=0.03)
    ap.add_argument("--sigma-el", type=float, default=0.20)
    ap.add_argument("--sigma-ph", type=float, default=0.30)
    ap.add_argument("--amplitude", type=float, default=0.01)
    ap.add_argument(
        "--converge",
        action="store_true",
        help="also report lambda at several electronic smearings",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.jid:
        from jarvis.core.atoms import Atoms as JAtoms
        from jarvis.db.figshare import data as jdata

        entry = {r["jid"]: r for r in jdata("dft_3d")}[args.jid]
        prim = JAtoms.from_dict(entry["atoms"]).ase_converter()
        label = f"{args.jid} {prim.get_chemical_formula()}"
    else:
        lattice = {"Al": 4.05, "Nb": 3.30, "Pb": 4.95, "Cu": 3.61}
        a0 = args.a or lattice.get(args.element, 4.0)
        prim = bulk(args.element, args.structure, a=a0, cubic=False)
        label = f"{args.element} {args.structure} a={a0} A"

    t0 = time.time()
    model = default_model(model_name=args.model).float()
    calc = SlaKoNetCalculator(
        model,
        kspacing=0.25,
        device=args.device,
        compute_forces=True,
        compute_stress=False,
    )
    print(
        f"[*] {label}, {len(prim)} atoms/cell, "
        f"{args.supercell}^3 supercell"
    )

    if args.force_calc == "alignn":
        from alignn.ff.ff import (
            AlignnAtomwiseCalculator,
            get_figshare_model_ff,
        )

        force_calc = AlignnAtomwiseCalculator(
            path=get_figshare_model_ff(model_name=args.ff_model)
        )
        print(f"[*] forces for phonons: alignn {args.ff_model}")
    else:
        force_calc = calc
        print("[*] forces for phonons: slakonet")

    print("[*] phonons")
    phonon = build_phonons(prim, force_calc, args.supercell, args.disp)
    freqs, evecs, masses = supercell_gamma_modes(phonon)
    n_at = len(masses)
    n_prim = n_at // len(prim)
    n_imag = int((freqs < -0.05).sum())
    print(
        f"    {n_at} atoms, {n_prim} primitive cells, "
        f"{len(freqs)} modes, {freqs.max():.2f} THz max, "
        f"{n_imag} imaginary"
    )

    sc = phonon.supercell
    sc_atoms = AseAtoms(
        symbols=sc.symbols,
        cell=sc.cell,
        scaled_positions=sc.scaled_positions,
        pbc=True,
    )
    sc_atoms.calc = calc
    sc_atoms.get_potential_energy()  # populates the Fermi level

    print(f"[*] electron-phonon on a {args.kmesh}^3 k-mesh")
    grid, a2f, n_ef, used = compute_a2f(
        calc,
        sc_atoms,
        freqs,
        evecs,
        masses,
        n_prim,
        [args.kmesh] * 3,
        amplitude=args.amplitude,
        sigma_el=args.sigma_el,
        sigma_ph=args.sigma_ph,
        solver_device=args.device,
    )
    if args.out:  # checkpoint before any post-processing can fail
        with open(args.out + ".a2f", "w") as f:
            json.dump(
                {
                    "omega_eV": grid.tolist(),
                    "a2F": a2f.tolist(),
                    "n_ef": n_ef,
                    "modes_used": used,
                },
                f,
            )
    lam, wlog_ev = lambda_wlog(grid, a2f)
    tc = allen_dynes_tc(lam, wlog_ev, args.mu_star)
    wlog_K = wlog_ev * KELVIN_PER_EV
    wlog = wlog_ev / THZ_TO_EV

    print(
        f"\n=== {label} ===\n"
        f"  modes used            : {used}\n"
        f"  N(Ef) [/eV/cell/spin] : {n_ef:.4f}\n"
        f"  lambda                : {lam:.3f}\n"
        f"  omega_log             : {wlog:.3f} THz = {wlog_K:.1f} K\n"
        f"  Tc (mu*={args.mu_star}) : {tc:.2f} K\n"
        f"  wall time             : {(time.time() - t0) / 60:.1f} min"
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "system": label,
                    "lambda": lam,
                    "omega_log_THz": wlog,
                    "omega_log_K": wlog_K,
                    "Tc_K": tc,
                    "mu_star": args.mu_star,
                    "n_ef": n_ef,
                    "omega_eV": grid.tolist(),
                    "a2F": a2f.tolist(),
                },
                f,
                indent=2,
            )
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
