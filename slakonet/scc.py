"""SCC-DFTB: self-consistent-charge extension for slakonet.

Implements the standard second-order SCC correction on top of slakonet's
non-SCC DFTB1 Hamiltonian. Given H0, S, Hubbard U per atom, and reference
atomic occupations, this module runs a charge-mixing SCF loop to find the
Mulliken charges that satisfy

    H_SCC = H_0 + V_SCC,   V_SCC[mu, nu] = (1/2) S[mu, nu] (Delta eps_A + Delta eps_B)
    Delta eps_A = sum_B gamma_AB * Delta q_B
    Delta q_A   = q_A - q_A_ref                               (Mulliken)

with E_SCC = (1/2) Sum_{AB} gamma_AB Delta q_A Delta q_B added to the total.

The gamma function is the Klopman-Ohno short-range Coulomb

    gamma_AB(r) = 1 / sqrt(r^2 + 1/Ubar^2),   Ubar = (U_A + U_B)/2

with gamma_AA = U_A. This is a simpler variant than DFTB+'s exponential-Slater
form but is adequate for capturing the on-site charge-fluctuation penalty
that stabilises E(V) curves.

All units internal to this module are atomic (Hartree, Bohr, electrons).
"""
from __future__ import annotations

import torch

from slakonet.utils import eighb
from slakonet.slaterkoster import fermi


# ---------------------------------------------------------------------------
# Gamma matrix
# ---------------------------------------------------------------------------
def build_gamma_matrix(positions_bohr, U_per_atom):
    """Klopman-Ohno gamma_AB matrix.

    Parameters
    ----------
    positions_bohr : Tensor [Natom, 3]
    U_per_atom     : Tensor [Natom]  Hubbard U in Hartree

    Returns
    -------
    gamma : Tensor [Natom, Natom]  (Hartree)
    """
    pos = positions_bohr.to(torch.float64)
    U = U_per_atom.to(torch.float64)
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)           # [N, N, 3]
    r2 = (diff * diff).sum(dim=-1)                       # [N, N]
    Ubar = 0.5 * (U.unsqueeze(0) + U.unsqueeze(1))       # [N, N]
    inv_U2 = 1.0 / (Ubar * Ubar)
    gamma = 1.0 / torch.sqrt(r2 + inv_U2)
    # On-site: gamma_AA = U_A (stronger than Klopman-Ohno limit 1/(1/U) = U,
    # which holds automatically here since r2=0 so gamma_AA = U_A)
    return gamma


# ---------------------------------------------------------------------------
# Atom-level helpers
# ---------------------------------------------------------------------------
def atom_U_from_skf(updated_skfs, Z, shell_dict):
    """Pick a single Hubbard U per atom: the U of the highest-l shell present
    in the atom's basis. Returns Hartree."""
    from jarvis.core.specie import atomic_numbers_to_symbols
    sym = atomic_numbers_to_symbols([Z])[0]
    pair = f"{sym}-{sym}"
    if pair not in updated_skfs:
        return None
    skf = updated_skfs[pair]
    d = skf.to_dict()
    ad = d.get("atomic_data", {})
    us = ad.get("hubbard_us") if ad else None
    if us is None:
        return None
    shells = shell_dict.get(int(Z), [0])
    # U is stored in SKF in order of shells available; pick the one matching
    # the highest-l shell the basis uses.
    l_max = max(shells)
    if l_max < len(us):
        return float(us[l_max])
    return float(us[-1])


def reference_charges(atomic_numbers, updated_skfs, shell_dict):
    """q_A_ref = total valence electrons on atom A (sum of occupations for the
    shells actually in the basis)."""
    from jarvis.core.specie import atomic_numbers_to_symbols
    q_ref = []
    Zs = atomic_numbers.flatten().tolist()
    for Z in Zs:
        if Z <= 0:
            q_ref.append(0.0)
            continue
        sym = atomic_numbers_to_symbols([int(Z)])[0]
        pair = f"{sym}-{sym}"
        if pair not in updated_skfs:
            q_ref.append(0.0)
            continue
        d = updated_skfs[pair].to_dict()
        occ = d.get("atomic_data", {}).get("occupations")
        if occ is None:
            q_ref.append(0.0)
            continue
        shells = shell_dict.get(int(Z), list(range(len(occ))))
        # Sum occupation of shells present in the basis
        total = 0.0
        for l in shells:
            if l < len(occ):
                total += float(occ[l])
        q_ref.append(total)
    return torch.tensor(q_ref, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Mulliken-per-atom from eigenvectors, overlaps, and occupations
# ---------------------------------------------------------------------------
def mulliken_atom_charges(C, S_k, occ, k_weights, on_atoms, Natom):
    """Standard Mulliken population analysis aggregated per atom.

    C         : [Norb, Nband, Nk] complex
    S_k       : [Norb, Norb, Nk] complex
    occ       : [Nband, Nk]       real
    k_weights : [Nk]              real, sums to 1
    on_atoms  : [Norb]            LongTensor, atom index per orbital
    Natom     : int
    Returns q_atom [Natom] (electrons on each atom).
    """
    device = C.device
    q = torch.zeros(Natom, dtype=torch.float64, device=device)
    atom_idx = on_atoms.clamp_min(0).to(device)
    Nk = C.shape[-1]
    for ik in range(Nk):
        Sk = S_k[..., ik].to(torch.complex128)
        Ck = C[..., ik].to(torch.complex128)
        SC = Sk @ Ck                                      # [Norb, Nband]
        pop = (Ck.conj() * SC).real                       # [Norb, Nband]
        f = occ[..., ik].to(torch.float64).reshape(-1)    # [Nband]
        pop_orb = (pop * f.unsqueeze(0)).sum(dim=1)       # [Norb]
        pop_orb = pop_orb * float(k_weights[ik])
        q.scatter_add_(0, atom_idx, pop_orb)
    return q


# ---------------------------------------------------------------------------
# SCC loop
# ---------------------------------------------------------------------------
def scc_solve(
    H0, S, basis, positions_bohr, U_per_atom, q_ref, nelectron,
    k_weights, kT_Ha=0.001, max_iter=60, mixing=0.2, tol=1e-5,
    verbose=False,
):
    """Iterate delta_q to self-consistency.

    H0, S : [Norb, Norb, Nk] complex tensors (non-SCC hamiltonian / overlap)
    basis : slakonet Basis (supplies on_atoms)
    positions_bohr : [Natom, 3]
    U_per_atom     : [Natom]
    q_ref          : [Natom]
    nelectron      : scalar (total valence electrons)
    k_weights      : [Nk]
    kT_Ha          : Fermi smearing in Hartree for the Fermi search
    Returns dict with keys:
        eigenvalues [Nband, Nk] (Ha), eigenvectors, occupations [Nband, Nk],
        delta_q [Natom], E_scc (Ha), converged (bool)
    """
    Norb, _, Nk = H0.shape
    device = H0.device
    on_atoms = basis.on_atoms
    if on_atoms.ndim == 2:
        on_atoms = on_atoms[0]
    Natom = q_ref.shape[0]

    # orbital -> atom index lookup
    orb_to_atom = on_atoms.to(device).long()

    gamma = build_gamma_matrix(positions_bohr, U_per_atom).to(device)

    delta_q = torch.zeros(Natom, dtype=torch.float64, device=device)

    for it in range(max_iter):
        # on-site shift per atom:  Delta_eps_A = sum_B gamma_AB Delta_q_B
        deps_atom = gamma @ delta_q                         # [Natom]
        # per-orbital: deps_orb[mu in A] = deps_atom[A]
        deps_orb = deps_atom[orb_to_atom]                   # [Norb]

        # V_SCC[mu, nu] = 1/2 S[mu, nu] (deps[mu] + deps[nu])
        # build for each k
        evals_k, occ_k, C_k = [], [], []
        for ik in range(Nk):
            Sk = S[..., ik].to(torch.complex128)
            Hk = H0[..., ik].to(torch.complex128)
            shift = 0.5 * (deps_orb.unsqueeze(1) + deps_orb.unsqueeze(0))
            Vk = Sk * shift.to(torch.complex128)
            H_scc = Hk + Vk
            e, c = eighb(H_scc, Sk, scheme="chol")
            evals_k.append(e); C_k.append(c)
        evals = torch.stack(evals_k, dim=-1)                # [Nband, Nk]

        # Fermi occupation at shared mu across all k. Spin-restricted:
        # occupations in [0, 2] with n_electrons set by nelectron total.
        mu = _solve_mu(evals, k_weights, float(nelectron), kT=kT_Ha)
        fu = 2.0 / (1.0 + torch.exp((evals.real - mu) / kT_Ha))
        occ = fu                                             # [Nband, Nk]
        C = torch.stack(C_k, dim=-1)                         # [Norb, Nband, Nk]

        q = mulliken_atom_charges(C, S, occ, k_weights, on_atoms, Natom)
        delta_q_new = q - q_ref.to(device)
        diff = (delta_q_new - delta_q).abs().max().item()
        if verbose:
            print(f"[SCC] iter {it:3d}  |dq|_inf={diff:.3e}  "
                  f"delta_q={delta_q_new.tolist()}")
        delta_q = (1.0 - mixing) * delta_q + mixing * delta_q_new
        if diff < tol:
            converged = True
            break
    else:
        converged = False

    # E_SCC = 1/2 Sum_AB gamma_AB dq_A dq_B
    E_scc = 0.5 * torch.dot(delta_q, gamma @ delta_q)

    return {
        "eigenvalues": evals,
        "eigenvectors": C,
        "occupations": occ,
        "delta_q": delta_q,
        "E_scc": E_scc,
        "mu_Ha": mu,
        "converged": converged,
        "n_iter": it + 1,
    }


def _solve_mu(evals, k_weights, n_target, kT):
    """Bisection for shared chemical potential in a spin-restricted setup
    (occupations in [0, 2])."""
    lo = float(evals.real.min().item()) - 1.0
    hi = float(evals.real.max().item()) + 1.0
    kw = k_weights.to(torch.float64)
    for _ in range(100):
        mu = 0.5 * (lo + hi)
        f = 2.0 / (1.0 + torch.exp((evals.real - mu) / kT))
        n = float(((f * kw.view(1, -1)).sum()).item())
        if n < n_target:
            lo = mu
        else:
            hi = mu
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


__all__ = [
    "build_gamma_matrix",
    "atom_U_from_skf",
    "reference_charges",
    "mulliken_atom_charges",
    "scc_solve",
]
