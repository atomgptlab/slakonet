"""Collinear spin-polarized slakonet model with an orbital-resolved Hubbard
correction (Dudarev form) on the d shell of transition-metal sites.

    H_sigma(R) = H0(R) + 1/2 S(R) o (deps_mu + deps_nu)            (SCC)
                 - sigma delta_{R,0} diag(1/2 I_l m_a)               (Stoner)
                 + delta_{R,0} sum_a U_a (1/2 1 - n_a^sigma)         (+U)

As in DFT+U = LSDA + U, the Stoner term (atomic I, the LSDA-like
exchange that gives Hund's high-spin states, e.g. Mn2+ d5) is kept and the
+U correction is added on top of it.

n_a^sigma is the 5x5 Mulliken occupation matrix of the d shell of site a
(rotationally invariant: the full matrix, not only its diagonal). Unlike the
rigid Stoner field of ``slakonet.magnetism`` (one shift per shell,
proportional to the atomic moment), the +U potential acts orbital by orbital
and spin by spin: occupied spin-orbitals move down by ~U/2 and empty ones up
by ~U/2. It can polarise a single hole (Cu d9 x2-y2), open Mott-like gaps,
and it sets the occupied-empty splitting that controls superexchange
J ~ t^2 / U.

The correction is on-site (R = 0), so TB2J's exchange splitting
Delta = H_up(0) - H_dn(0) = U (n_dn - n_up) is well defined.
"""

from __future__ import annotations

import numpy as np
import torch

from slakonet.magnetism import (
    DEFAULT_STONER_I,
    _build_stoner_shift_per_orbital,
)
from slakonet.tb2j_interface import (
    SlakonetSpinModel,
    _bloch,
    _default_initial_moments,
    _mp_grid,
    _orbital_populations,
    _shared_fermi_and_occ,
    _solve_generalized,
    check_overlap,
    extract_real_space_hs,
    orbital_labels,
)

HARTREE_EV = 27.211386

# Universal U_eff (eV) for all 3d metals, fitted on 2026-09-30 to 129
# experimental exchange constants of 62 compounds (ExpMagDB) with
# examples/fit_stoner_table.py (--spin-model hubbard --grouping global,
# symmetric log-ratio loss). Leave-one-compound-out: 51/129 couplings within
# a factor of 2, 94/129 with the right sign, median |J_calc/J_ref| = 0.88,
# MAE 9.4 meV (Stoner table: 28/124, 83/124, 0.27, 10.5 meV). A single U
# beat per-element/per-class U tables in cross-validation (they overfit).
UNIVERSAL_U_EV = 2.97
TRANSITION_3D = ("Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu")
# Default (2026-10-02): U_eff (eV) by ligand class of the site
# (``slakonet.magnetic_sites.site_anion_class``), fitted the same way
# (grouping "anion"). Leave-one-compound-out: 57/129 couplings within a
# factor of 2 (universal U: 51), median |J_calc/J_ref| 0.82, 88/129 signs
# (universal: 94). Independent support: in DFT+U U-scans (PBE+U, 18
# oxides/fluorides) the slakonet U that reproduces DFT at U = 5 eV is ~3 eV
# for oxides but ~12.5 eV for fluorides, i.e. the TB error depends on the
# ligand (F 2p level / M-F hopping), not on the metal. Classes without a
# fitted value use UNIVERSAL_U_EV.
ANION_U_EV = {
    "O": 2.737,
    "F": 9.189,
    "halide": 4.063,
    "chalcogenide": 7.097,
    "metallic": 1.976,
}
# Materials Project GGA+U values (eV), for reference only.
MP_HUBBARD_U_EV = {
    "V": 3.25,
    "Cr": 3.7,
    "Mn": 3.9,
    "Fe": 5.3,
    "Co": 3.32,
    "Ni": 6.2,
}


def _occupation_matrices(C, SC, occ, k_weights, blocks):
    """Mulliken occupation matrices n[m, m'] of the orbital blocks.

    C, SC : [Nk, Norb, Nb] complex; occ : [Nk, Nb]; blocks : {site: indices}.
    n = sum_k w_k sum_b f_b 1/2 (C*_mb (SC)_m'b + (SC)*_mb C_m'b), real part.
    """
    out = {}
    for a, idx in blocks.items():
        c, sc = C[:, idx, :], SC[:, idx, :]
        wf = occ * k_weights[:, None]  # [Nk, Nb]
        A = torch.einsum("kmb,knb,kb->mn", c.conj(), sc, wf.to(c.dtype))
        n = 0.5 * (A + A.conj().T)
        out[a] = n.real
    return out


def build_hubbard_model(
    atoms,
    model=None,
    hubbard_U=None,
    u_table="anion",
    stoner_I=None,
    initial_moments=None,
    magnetic_elements=None,
    kmesh=(8, 8, 8),
    use_scc=True,
    kT=0.025,
    max_iter=300,
    mixing=0.3,
    tol=1e-5,
    cutoff=10.0,
    device="cpu",
    verbose=True,
):
    """Converge the collinear slakonet + U (+ SCC) model of ``atoms``.

    Parameters
    ----------
    hubbard_U : dict {symbol: U_eff in Hartree} or {atom index: U}, optional
        d-shell U per element (or per site); takes precedence over the table.
    u_table : {"anion", "universal"}
        Default U of the 3d sites: by ligand class (``ANION_U_EV``, default)
        or the single ``UNIVERSAL_U_EV``, both fitted to experimental
        exchange constants.
    initial_moments : sequence [Natom], optional
        Signed starting moments; set the collinear order.
    stoner_I : dict {Z: {l: I_Ha}}, optional
        LSDA-like Stoner term; defaults to the atomic values
        (``slakonet.magnetism.DEFAULT_STONER_I``). Pass zero values to switch
        it off and keep +U alone.
    use_scc : bool
        Converge the SCC charges as well (periodic DFTB gamma); recommended,
        since +U alone has no penalty for moving charge between sites.
    Other parameters as in ``slakonet.tb2j_interface.build_spin_model``.
    """
    from slakonet.atoms import Geometry
    from slakonet.main import SimpleDftb
    from slakonet.scc import AndersonMixer

    if model is None:
        from slakonet.optim import default_model

        model = default_model()

    calc = SimpleDftb(
        Geometry.from_ase_atoms([atoms]),
        model,
        cutoff=cutoff,
        kpoints=torch.tensor([1, 1, 1]),
        device=device,
        compute_forces=False,
        include_dos_data=False,
        repulsive=False,
        kT=kT,
    )
    H2E = calc.H2E
    Rlist_np, HR0_np, SR_np = extract_real_space_hs(calc)
    labels, orb_atom_np, orb_l_np = orbital_labels(calc, atoms)
    Rlist = torch.as_tensor(Rlist_np, dtype=torch.float64)
    HR0, SR = torch.as_tensor(HR0_np), torch.as_tensor(SR_np)
    iR0 = int(np.where(~Rlist_np.any(axis=1))[0][0])
    orb_atom = torch.as_tensor(orb_atom_np)
    natom = len(atoms)
    symbols = atoms.get_chemical_symbols()
    Zs = atoms.get_atomic_numbers()

    kpts = torch.as_tensor(_mp_grid(kmesh), dtype=torch.float64)
    k_weights = torch.full((len(kpts),), 1.0 / len(kpts), dtype=torch.float64)
    check_overlap(
        Rlist_np, SR_np, np.vstack([np.zeros((1, 3)), kpts.numpy()]), labels
    )
    Sk = _bloch(Rlist, SR, kpts)
    Lk_inv = torch.linalg.inv(torch.linalg.cholesky(Sk))

    # correlated sites and their U
    if u_table not in ("anion", "universal"):
        raise ValueError(
            f"u_table must be 'anion' or 'universal', got {u_table!r}"
        )
    explicit = hubbard_U or {}
    site_U = {}
    for a, s in enumerate(symbols):
        u = explicit.get(a, explicit.get(s))
        if u is None and s in TRANSITION_3D:
            if u_table == "anion":
                from slakonet.magnetic_sites import site_anion_class

                u = (
                    ANION_U_EV.get(site_anion_class(atoms, a), UNIVERSAL_U_EV)
                    / HARTREE_EV
                )
            else:
                u = UNIVERSAL_U_EV / HARTREE_EV
        if u and ((orb_atom_np == a) & (orb_l_np == 2)).sum() == 5:
            site_U[a] = float(u)
    blocks = {
        a: np.where((orb_atom_np == a) & (orb_l_np == 2))[0] for a in site_U
    }
    stoner_full = dict(DEFAULT_STONER_I)
    for Z, shells in (stoner_I or {}).items():
        stoner_full[Z] = {**stoner_full.get(Z, {}), **shells}

    if initial_moments is None:
        mag = set(magnetic_elements or [])
        mag_idx = [i for i, s in enumerate(symbols) if s in mag]
        if not mag_idx:
            raise ValueError("Give initial_moments or magnetic_elements.")
        initial_moments = _default_initial_moments(Zs, mag_idx)
    m0 = np.asarray(initial_moments, dtype=float).flatten()
    if len(m0) != natom:
        raise ValueError(
            f"initial_moments has {len(m0)} entries, Natom={natom}"
        )

    nelec = float(calc.nelectron.flatten()[0])
    kT_Ha = kT / H2E

    # starting occupation matrices: non-magnetic d populations, polarised by m0
    e0, C0 = _solve_generalized(_bloch(Rlist, HR0, kpts), Lk_inv)
    _, f0, _ = _shared_fermi_and_occ(e0.T, e0.T, k_weights, nelec, kT=kT_Ha)
    SC0 = Sk @ C0
    n0 = _occupation_matrices(
        C0, SC0, f0.T.to(torch.float64), k_weights, blocks
    )

    def clip(n):  # keep occupation eigenvalues in [0, 1]
        w, v = torch.linalg.eigh(n)
        return (v * w.clamp(0.0, 1.0)) @ v.T

    # start from the expected configuration of the formal ion (high spin,
    # or low spin when the geometry says so); +U amplifies the starting
    # polarisation, so a weak start would end low spin
    from slakonet.magnetic_sites import site_details

    n_up, n_dn = {}, {}
    for a, n in n0.items():
        m_exp = site_details(atoms, a)["m_expected"]  # HS, or LS from geometry
        m_start = np.sign(m0[a]) * (m_exp if m_exp else abs(m0[a]))
        m0[a] = m_start
        shift = torch.eye(5, dtype=torch.float64) * (m_start / 10.0)
        n_up[a], n_dn[a] = clip(n + shift), clip(n - shift)

    m = torch.as_tensor(m0, dtype=torch.float64)
    dq = torch.zeros(natom, dtype=torch.float64)
    if use_scc:
        from slakonet.scc import (
            atom_U_from_skf,
            build_gamma_matrix_periodic,
            reference_charges,
        )

        U_scc = torch.tensor(
            [
                atom_U_from_skf(calc.updated_skfs, int(Z), calc.shell_dict)
                for Z in Zs.tolist()
            ],
            dtype=torch.float64,
        )
        gamma = build_gamma_matrix_periodic(
            calc.geometry.positions[0].to(torch.float64).cpu(),
            calc.geometry.cell[0],
            U_scc,
        )
        q_ref = reference_charges(
            torch.as_tensor(Zs), calc.updated_skfs, calc.shell_dict
        ).to(torch.float64)

    def pack(n_up, n_dn, m, dq):
        parts = [n_up[a].flatten() for a in blocks] + [
            n_dn[a].flatten() for a in blocks
        ]
        return torch.cat(parts + [m] + ([dq] if use_scc else []))

    def unpack(x):
        k = 0
        nu, nd = {}, {}
        for a in blocks:
            nu[a] = x[k : k + 25].reshape(5, 5)
            k += 25
        for a in blocks:
            nd[a] = x[k : k + 25].reshape(5, 5)
            k += 25
        m_x = x[k : k + natom]
        k += natom
        return nu, nd, m_x, (x[k:] if use_scc else dq)

    def hamiltonians(n_up, n_dn, m, dq):
        HR = HR0.clone()
        if use_scc:
            deps = (gamma @ dq)[orb_atom]
            HR = HR + 0.5 * SR * (deps[:, None] + deps[None, :])
        V = _build_stoner_shift_per_orbital(calc.basis, m, stoner_full).cpu()
        HR_up, HR_dn = HR.clone(), HR.clone()
        HR_up[iR0] += torch.diag(V)
        HR_dn[iR0] -= torch.diag(V)
        half = 0.5 * torch.eye(5, dtype=torch.float64)
        for a, idx in blocks.items():
            ix = np.ix_(idx, idx)
            HR_up[iR0][ix] += site_U[a] * (half - n_up[a])
            HR_dn[iR0][ix] += site_U[a] * (half - n_dn[a])
        return HR_up, HR_dn

    mixer = AndersonMixer(alpha=mixing)
    history, converged = [], False
    for it in range(max_iter):
        HR_up, HR_dn = hamiltonians(n_up, n_dn, m, dq)
        e_up, C_up = _solve_generalized(_bloch(Rlist, HR_up, kpts), Lk_inv)
        e_dn, C_dn = _solve_generalized(_bloch(Rlist, HR_dn, kpts), Lk_inv)
        mu, f_up, f_dn = _shared_fermi_and_occ(
            e_up.T, e_dn.T, k_weights, nelec, kT=kT_Ha
        )
        fu, fd = f_up.T.to(torch.float64), f_dn.T.to(torch.float64)
        nu_new = _occupation_matrices(C_up, Sk @ C_up, fu, k_weights, blocks)
        nd_new = _occupation_matrices(C_dn, Sk @ C_dn, fd, k_weights, blocks)
        p_up = _orbital_populations(C_up, Sk, fu, k_weights)
        p_dn = _orbital_populations(C_dn, Sk, fd, k_weights)
        m_new = torch.zeros(natom, dtype=torch.float64).scatter_add_(
            0, orb_atom, p_up - p_dn
        )
        q = torch.zeros(natom, dtype=torch.float64).scatter_add_(
            0, orb_atom, p_up + p_dn
        )
        dq_new = q - q_ref if use_scc else dq
        x_in, x_out = pack(n_up, n_dn, m, dq), pack(
            nu_new, nd_new, m_new, dq_new
        )
        res = (x_out - x_in).abs().max().item() if len(x_in) else 0.0
        history.append(
            {
                "iter": it,
                "residual": res,
                "M_tot": float(m_new.sum()),
                "mu_eV": mu * H2E,
            }
        )
        if verbose:
            print(
                f"[slakonet+U] iter {it:3d}  residual={res:.3e}  "
                f"M_tot={float(m_new.sum()):+.4f}  mu={mu * H2E:.4f} eV"
            )
        if res < tol:
            converged = True
            break
        n_up, n_dn, m, dq = unpack(mixer(x_in, x_out))

    if not converged:
        print(
            f"WARNING: slakonet+U SCF not converged after {max_iter} iterations "
            f"(residual {res:.2e}); the model of the last iteration is used."
        )

    # insulator: Fermi level at mid-gap (see build_spin_model)
    e_all = torch.cat([e_up.T.flatten(), e_dn.T.flatten()])
    f_all = torch.cat([f_up.flatten(), f_dn.flatten()])
    occupied, empty = e_all[f_all > 0.5], e_all[f_all <= 0.5]
    if (
        len(occupied)
        and len(empty)
        and empty.min() - occupied.max() > 4 * kT_Ha
    ):
        mu = 0.5 * float(occupied.max() + empty.min())

    return SlakonetSpinModel(
        atoms=atoms.copy(),
        Rlist=Rlist_np,
        HR_up=HR_up.numpy() * H2E,
        HR_dn=HR_dn.numpy() * H2E,
        SR=SR_np,
        labels=labels,
        efermi=float(mu * H2E),
        moments=m_new.numpy(),
        charges=q.numpy(),
        nelectron=nelec,
        converged=converged,
        n_iter=it + 1,
        history=history,
        hubbard_U={int(a): u for a, u in site_U.items()},
        potentials=_frozen_potentials(
            calc.basis,
            m,
            dq,
            n_up,
            n_dn,
            site_U,
            blocks,
            stoner_full,
            gamma if use_scc else None,
            H2E,
        ),
    )


def _frozen_potentials(
    basis, m, dq, n_up, n_dn, site_U, blocks, stoner_full, gamma, H2E
):
    """Spin and charge potentials of the last iteration, in eV, so that
    H_sigma(R) = H0(R) + 0.5 S(R) (s_i + s_j) + diag(+-V) + U (1/2 - n_sigma)
    can be rebuilt with other Slater-Koster parameters (frozen-potential
    training, see slakonet.spin_training)."""
    half = 0.5 * torch.eye(5, dtype=torch.float64)
    V = (
        _build_stoner_shift_per_orbital(basis, m, stoner_full)
        .cpu()
        .to(torch.float64)
    )
    return {
        "scc_shift": (
            ((gamma @ dq) * H2E).numpy()
            if gamma is not None
            else np.zeros(len(m))
        ),
        "stoner_V": (V * H2E).numpy(),
        "U_up": {
            int(a): (blocks[a], (site_U[a] * (half - n_up[a]) * H2E).numpy())
            for a in blocks
        },
        "U_dn": {
            int(a): (blocks[a], (site_U[a] * (half - n_dn[a]) * H2E).numpy())
            for a in blocks
        },
    }
