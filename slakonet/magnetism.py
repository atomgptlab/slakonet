"""Collinear spin-polarized bandstructure via Stoner-like on-site exchange.

Post-processor for an already-built non-magnetic SimpleDftb calculator. For
each k we add a diagonal (in orbital space) shell-resolved shift

    V_{mu mu}^{sigma} = -(sigma/2) * I_l(Z) * m_a

where sigma = +1 for up, -1 for down, m_a is the atom's magnetic moment, and
I_l(Z) is a shell-resolved Stoner parameter. Two spin channels are solved
independently. Optional SCF iterates m_a from the Mulliken spin density.

This treats the exchange splitting rigidly at the Stoner level - it is not
a full spin-polarized SCC-DFTB implementation. It reproduces the qualitative
band-splitting of Fe/Ni/Co/Cr/Mn-containing systems when reasonable I values
are supplied.
"""
from __future__ import annotations

import torch

# Shell-resolved Stoner parameters (Hartree). Values from published DFTB
# parameter sets (mio, 3ob, matsci) and atomic calculations; override via
# stoner_I= kwarg if you need different numbers.
DEFAULT_STONER_I = {
    1:  {0: 0.072},
    3:  {0: 0.000},
    6:  {0: 0.000, 1: 0.000},
    7:  {0: 0.000, 1: 0.000},
    8:  {0: 0.000, 1: 0.000},
    22: {0: 0.000, 1: 0.000, 2: 0.024},  # Ti
    23: {0: 0.000, 1: 0.000, 2: 0.026},  # V
    24: {0: 0.000, 1: 0.000, 2: 0.028},  # Cr
    25: {0: 0.000, 1: 0.000, 2: 0.031},  # Mn
    26: {0: 0.000, 1: 0.000, 2: 0.035},  # Fe
    27: {0: 0.000, 1: 0.000, 2: 0.034},  # Co
    28: {0: 0.000, 1: 0.000, 2: 0.037},  # Ni
    29: {0: 0.000, 1: 0.000, 2: 0.028},  # Cu
}


def _orbital_to_shell_l_and_atom(basis):
    """Return two 1-D LongTensors of length Norb: atom index per orbital and
    angular momentum l per orbital. Works for the unbatched case."""
    on_atoms = basis.on_atoms
    on_shells = basis.on_shells
    shell_ls = basis.shell_ls
    if on_atoms.ndim == 2:
        on_atoms = on_atoms[0]
        on_shells = on_shells[0]
        shell_ls = shell_ls[0] if shell_ls.ndim == 2 else shell_ls

    atom_idx = on_atoms.long()
    # on_shells is a global shell index (concatenated orbs_per_shell counter)
    global_shell_idx = on_shells.long()
    ls = shell_ls[global_shell_idx]
    return atom_idx, ls


def _build_stoner_shift_per_orbital(basis, moments, stoner_I):
    """Compute per-orbital shift tensor V_mu for spin +1 channel (spin -1
    is just -V). Shape: [Norb]. Hartree."""
    atomic_numbers = basis.atomic_numbers
    if atomic_numbers.ndim == 2:
        atomic_numbers = atomic_numbers[0]
    atom_idx, ls = _orbital_to_shell_l_and_atom(basis)
    device = atomic_numbers.device
    V = torch.zeros(atom_idx.shape[0], device=device, dtype=torch.float64)
    for orb, (ai, l) in enumerate(zip(atom_idx.tolist(), ls.tolist())):
        if ai < 0:
            continue
        Z = int(atomic_numbers[ai].item())
        I = stoner_I.get(Z, {}).get(int(l), 0.0)
        V[orb] = -0.5 * I * float(moments[ai].item())
    return V


def _mulliken_shell_spin_moments(
    eigenvecs_up, eigenvecs_dn, occ_up, occ_dn, S_k, k_weights, basis
):
    """Compute atom-resolved spin moments m_a = n_up - n_dn via Mulliken.

    Inputs (unbatched):
        eigenvecs_*: [Norb, Nband, Nk] complex
        occ_*:      [Nband, Nk]
        S_k:        [Norb, Norb, Nk] complex
        k_weights:  [Nk]
    Returns: moments [Natom]
    """
    atom_idx, _ = _orbital_to_shell_l_and_atom(basis)
    Natom = int(atom_idx.max().item()) + 1
    device = eigenvecs_up.device
    m = torch.zeros(Natom, device=device, dtype=torch.float64)
    # scatter target: squeeze any negative (pad) indices
    atom_idx_pos = atom_idx.clone().clamp_min(0).to(device)

    Nk = eigenvecs_up.shape[-1]
    for ik in range(Nk):
        Sk = S_k[..., ik].to(torch.complex128)
        for sigma, (C, f) in enumerate(
            [(eigenvecs_up[..., ik], occ_up[..., ik]),
             (eigenvecs_dn[..., ik], occ_dn[..., ik])]
        ):
            # Squeeze any batch dim that leaked in from fermi()
            C = C.to(torch.complex128)
            if C.ndim == 3:
                C = C.squeeze(0)
            f_1d = f.to(torch.float64).reshape(-1)        # [Nband]
            SC = Sk @ C                                   # [Norb, Nband]
            pop = (C.conj() * SC).real * f_1d.unsqueeze(0)
            pop_per_orb = pop.sum(dim=1)                  # [Norb]
            pop_per_orb = pop_per_orb * float(k_weights[ik])
            sign = 1.0 if sigma == 0 else -1.0
            # scatter per atom
            m.scatter_add_(0, atom_idx_pos, sign * pop_per_orb)
    return m


def _diagonalize_spin_channel(H_k, S_k, V_orb):
    """Diagonalize H + diag(V_orb) at each k. Returns eigenvalues (Hartree)
    and eigenvectors. No occupation filling."""
    from slakonet.utils import eighb

    Nk = H_k.shape[-1]
    evals, evecs = [], []
    diagV = torch.diag(V_orb.to(torch.complex128))
    for ik in range(Nk):
        h = H_k[..., ik].to(torch.complex128) + diagV
        s = S_k[..., ik].to(torch.complex128)
        e, c = eighb(h, s, scheme="chol")
        evals.append(e)
        evecs.append(c)
    return torch.stack(evals, dim=-1), torch.stack(evecs, dim=-1)


def _shared_fermi_and_occ(ev_up, ev_dn, k_weights, nelec_total, kT=0.025):
    """Find a single chemical potential mu that gives nelec_total electrons
    across the two spin channels, using Fermi-Dirac smearing. Occupations are
    in [0,1] (no spin-doubling). Bisection in mu.

    ev_* : [Nband, Nk] real  (Hartree)
    k_weights: [Nk] real, sums to 1
    Returns: mu (scalar), occ_up [Nband, Nk], occ_dn [Nband, Nk]
    """
    all_e = torch.cat([ev_up.flatten(), ev_dn.flatten()])
    lo = all_e.min().item() - 1.0
    hi = all_e.max().item() + 1.0

    def occ_at(mu):
        # Use Fermi-Dirac at temperature kT (Hartree)
        fu = 1.0 / (1.0 + torch.exp((ev_up - mu) / kT))
        fd = 1.0 / (1.0 + torch.exp((ev_dn - mu) / kT))
        return fu, fd

    def n_at(mu):
        fu, fd = occ_at(mu)
        kw = k_weights.view(1, -1)
        return float(((fu + fd) * kw).sum().item())

    # bisection
    n_target = float(nelec_total)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if n_at(mid) < n_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    mu = 0.5 * (lo + hi)
    fu, fd = occ_at(mu)
    return mu, fu, fd


def compute_spin_polarized_bands(
    calc,
    stoner_I=None,
    initial_moments=None,
    scf=True,
    max_iter=30,
    mixing=0.3,
    tol=1e-4,
    verbose=False,
):
    """Run collinear spin-polarized bands on top of an evaluated SimpleDftb.

    Parameters
    ----------
    calc : SimpleDftb
        Must have been instantiated with with_eigenvectors=True, include_HS=True
        and had calc.calculate() already called.
    stoner_I : dict {Z: {l: I_in_Hartree}}
        Shell-resolved Stoner parameters; defaults filled in for common elements.
    initial_moments : tensor or list [Natom]
        Initial atomic moments in e (up-down). Defaults to +2 on d-block atoms.
    scf : bool
        If False, perform a single shot with initial_moments.
    """
    H2E = getattr(calc, "H2E", 27.211)

    results = calc._results
    if results is None:
        raise RuntimeError("Run calc.calculate() before spin-polarizing it.")
    if "hamiltonian" not in results or "overlap" not in results:
        raise RuntimeError("calc must have include_HS=True.")

    H = results["hamiltonian"]
    S = results["overlap"]
    # unbatched view
    if H.ndim == 4:
        H = H[0]
        S = S[0]
    basis = calc.basis
    k_weights = calc.k_weights.flatten().to(H.device).to(torch.float64)
    nelec_total = calc.nelectron.flatten()[0].to(torch.float64)

    stoner_full = dict(DEFAULT_STONER_I)
    if stoner_I is not None:
        for k, v in stoner_I.items():
            stoner_full[k] = {**stoner_full.get(k, {}), **v}

    atomic_numbers = basis.atomic_numbers
    if atomic_numbers.ndim == 2:
        atomic_numbers = atomic_numbers[0]
    Natom = atomic_numbers.shape[0]
    device = H.device

    if initial_moments is None:
        m = torch.zeros(Natom, device=device, dtype=torch.float64)
        for i, Z in enumerate(atomic_numbers.tolist()):
            if Z in (24, 25, 26, 27, 28):
                m[i] = 2.0
            elif Z in (22, 23, 29):
                m[i] = 0.5
    else:
        m = torch.as_tensor(
            initial_moments, device=device, dtype=torch.float64
        ).flatten()
        if m.shape[0] != Natom:
            raise ValueError(
                f"initial_moments length {m.shape[0]} != Natom {Natom}"
            )

    # kT: calc.kT is in eV, convert to Hartree for shared-mu solver
    kT_Ha = float(calc.kT) / H2E

    converged = False
    for it in range(max_iter if scf else 1):
        V_up = _build_stoner_shift_per_orbital(basis, m, stoner_full).to(device)
        V_dn = -V_up

        ev_up, vc_up = _diagonalize_spin_channel(H, S, V_up)
        ev_dn, vc_dn = _diagonalize_spin_channel(H, S, V_dn)

        # Shared Fermi level across both spin channels
        mu, oc_up, oc_dn = _shared_fermi_and_occ(
            ev_up.real, ev_dn.real, k_weights, nelec_total, kT=kT_Ha
        )

        if not scf:
            break

        m_new = _mulliken_shell_spin_moments(
            vc_up, vc_dn, oc_up, oc_dn, S, k_weights, basis
        )
        dm = (m_new - m).abs().max().item()
        if verbose:
            print(
                f"[spin-SCF] iter {it}  |dm|_inf={dm:.4e}  "
                f"M_tot={float(m_new.sum().item()):.3f}  mu={mu:.4f} Ha"
            )
        m = (1.0 - mixing) * m + mixing * m_new
        if dm < tol:
            converged = True
            break

    return {
        "eigenvalues_up": ev_up * H2E,
        "eigenvalues_dn": ev_dn * H2E,
        "eigenvectors_up": vc_up,
        "eigenvectors_dn": vc_dn,
        "occupations_up": oc_up,
        "occupations_dn": oc_dn,
        "moments": m,
        "converged": converged,
        "total_moment": float(m.sum().item()),
        "fermi_eV": float(mu * H2E),
    }


def plot_spin_bands(result, fermi_shift_eV=0.0, filename="bands_spin.png"):
    import matplotlib.pyplot as plt

    eu = result["eigenvalues_up"].detach().cpu().numpy()
    ed = result["eigenvalues_dn"].detach().cpu().numpy()
    # shape: [Nband, Nk]
    plt.figure(figsize=(8, 6))
    for b in range(eu.shape[0]):
        plt.plot(eu[b] - fermi_shift_eV, color="tab:red", lw=0.8)
    for b in range(ed.shape[0]):
        plt.plot(ed[b] - fermi_shift_eV, color="tab:blue", lw=0.8, ls="--")
    plt.axhline(0, ls="-.", color="k")
    plt.xlabel("k-point")
    plt.ylabel("E (eV)")
    plt.title(f"Spin bands (M = {result['total_moment']:.3f})")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
