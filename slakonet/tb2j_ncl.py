"""Spin-orbit coupling in the slakonet -> TB2J workflow (non-collinear
exchange: isotropic J, symmetric anisotropic exchange, DMI).

One-shot (force-theorem) approach, as in TB2J's split-SOC workflow:

1. converge the collinear model (``gen_exchange_slakonet`` defaults: atomic
   Stoner + anion-dependent U + SCC, or the Stoner model);
2. build the spinor Hamiltonian in real space, in TB2J's interleaved order
   (orb1 up, orb1 down, orb2 up, ...):

       H(R) = Hbar(R) x 1 + Delta(R) x (n . sigma) + delta_{R,0} H_SOC
       Hbar = (H_up + H_dn) / 2,  Delta = (H_up - H_dn) / 2 (on-site only),

   where n is the spin quantisation axis (the collinear exchange field is
   rigidly rotated to n) and H_SOC = lambda_l L.S on each p/d shell
   (``slakonet.soc`` blocks, H = lambda L.S with S = sigma/2, so lambda is
   the atomic zeta_nl);
3. run TB2J ``ExchangeNCL`` -> J_iso, J_ani, DMI.

SOC is not made self-consistent; on-site SOC only (no two-centre SOC
integrals). The full exchange tensor needs runs with n along x, y and z,
combined with ``TB2J_merge.py`` (standard TB2J practice).
"""

from __future__ import annotations

import numpy as np

HARTREE_EV = 27.211386
CM1_TO_HA = 1.0 / 219474.63

# zeta_nl in cm^-1 with sources. Values are atomic/ionic; they can be
# overridden with ``lambda_soc={Z: {l: value_in_Hartree}}``.
#  - 3d: free-ion M2+ zeta_3d derived (2026-09-30) from the measured
#    fine structure of the ground term in the NIST Atomic Spectra Database
#    (Ti III 3F, V III 4F, Cr III 5D, Fe III 5D, Co III 4F, Ni III 3F,
#    Cu III 2D): least-squares Lande-interval fit E(J) = E0 + (lambda/2)
#    J(J+1), zeta = 2S|lambda|. They agree within 3 % with Griffith's
#    tabulated values. Mn2+ (6S ground term, no fine structure) is
#    interpolated along the M2+ series (cubic fit, +-15 cm^-1).
#  - halogens: zeta_np = 2 Delta / 3 from the 2P3/2 - 2P1/2 fine-structure
#    splitting Delta of the neutral atoms (NIST ASD):
#    F 404.14, Cl 882.35, Br 3685.24, I 7603.15 cm^-1.
SOC_ZETA_CM1 = {
    "Ti": {2: 120.0},
    "V": {2: 166.6},
    "Cr": {2: 230.3},
    "Mn": {2: 311.7},
    "Fe": {2: 410.5},
    "Co": {2: 534.2},
    "Ni": {2: 650.2},
    "Cu": {2: 828.7},
    "F": {1: 2 * 404.14 / 3},
    "Cl": {1: 2 * 882.35 / 3},
    "Br": {1: 2 * 3685.24 / 3},
    "I": {1: 2 * 7603.15 / 3},
}

_L_OF_ORBITAL = {
    "s": 0,
    "py": 1,
    "pz": 1,
    "px": 1,
    "dxy": 2,
    "dyz": 2,
    "dz2": 2,
    "dxz": 2,
    "dx2-y2": 2,
}


def default_lambda_soc():
    """{Z: {l: zeta in Hartree}} from ``SOC_ZETA_CM1``."""
    from ase.data import atomic_numbers

    return {
        atomic_numbers[el]: {l: z * CM1_TO_HA for l, z in shells.items()}
        for el, shells in SOC_ZETA_CM1.items()
    }


def _shells(labels):
    """[(atom label, l, [orbital indices])] of the collinear basis."""
    out, cur = [], None
    for i, lab in enumerate(labels):
        atom, orb = lab.split("|")[:2]
        l = _L_OF_ORBITAL[orb]
        if cur and cur[0] == atom and cur[1] == l and len(cur[2]) < 2 * l + 1:
            cur[2].append(i)
        else:
            cur = (atom, l, [i])
            out.append(cur)
    return out


def soc_onsite(labels, atoms, lambda_soc=None):
    """On-site lambda L.S (eV) in the interleaved spinor basis, [2N, 2N]."""
    from slakonet.soc import _onsite_LS_block

    lam = default_lambda_soc() if lambda_soc is None else lambda_soc
    symbol_of = {}
    counts = {}
    for s in atoms.get_chemical_symbols():
        counts[s] = counts.get(s, 0) + 1
        symbol_of[f"{s}{counts[s]}"] = s
    from ase.data import atomic_numbers

    N = len(labels)
    H = np.zeros((2 * N, 2 * N), dtype=complex)
    for atom, l, idx in _shells(labels):
        z = lam.get(atomic_numbers[symbol_of[atom]], {}).get(l, 0.0)
        if l == 0 or z == 0.0:
            continue
        uu, ud, du, dd = (
            b.numpy() * HARTREE_EV for b in _onsite_LS_block(l, z)
        )
        ii = np.array(idx)
        for (su, sv), blk in (
            ((0, 0), uu),
            ((0, 1), ud),
            ((1, 0), du),
            ((1, 1), dd),
        ):
            H[np.ix_(2 * ii + su, 2 * ii + sv)] = blk
    return H


def spinor_hamiltonian(
    spin_model, axis=(0.0, 0.0, 1.0), lambda_soc=None, soc_strength=1.0
):
    """Real-space spinor H(R), S(R) (eV) of a converged collinear model.

    Returns (Rlist, HR [nR, 2N, 2N], SR [nR, 2N, 2N], labels) in TB2J's
    interleaved order.
    """
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    sigma_n = (
        n[0] * np.array([[0, 1], [1, 0]])
        + n[1] * np.array([[0, -1j], [1j, 0]])
        + n[2] * np.array([[1, 0], [0, -1]])
    )
    Hbar = 0.5 * (spin_model.HR_up + spin_model.HR_dn)
    Delta = 0.5 * (spin_model.HR_up - spin_model.HR_dn)
    HR = np.array(
        [
            np.kron(Hbar[r], np.eye(2)) + np.kron(Delta[r], sigma_n)
            for r in range(len(Hbar))
        ]
    )
    iR0 = int(np.where(~np.asarray(spin_model.Rlist).any(axis=1))[0][0])
    if soc_strength:
        HR[iR0] += soc_strength * soc_onsite(
            spin_model.labels, spin_model.atoms, lambda_soc
        )
    SR = np.array([np.kron(S, np.eye(2)) for S in spin_model.SR])
    labels = [
        f"{lab}|{spin}" for lab in spin_model.labels for spin in ("up", "down")
    ]
    return spin_model.Rlist, HR, SR, labels


def _fermi_level(Rlist, HR, SR, nelectron, kmesh, kT=0.025):
    """Fermi level (eV) of the spinor model on a Monkhorst-Pack mesh; mid-gap
    for insulators."""
    from scipy.linalg import eigh

    axes = [(np.arange(k) + 0.5) / k for k in kmesh]
    kpts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    phase = np.exp(2j * np.pi * kpts @ np.asarray(Rlist).T)
    e = np.concatenate(
        [
            eigh(
                np.einsum("r,rab->ab", p, HR),
                np.einsum("r,rab->ab", p, SR),
                eigvals_only=True,
            )
            for p in phase
        ]
    )
    e = np.sort(e)
    n_occ = int(round(nelectron)) * len(kpts)  # one electron per spinor state
    vbm, cbm = e[n_occ - 1], e[n_occ]
    if cbm - vbm > 4 * kT:
        return 0.5 * (vbm + cbm)
    lo, hi = e[0] - 1, e[-1] + 1
    for _ in range(100):
        mu = 0.5 * (lo + hi)
        n = np.sum(
            1.0 / (1.0 + np.exp(np.clip((e - mu) / kT, -60, 60)))
        ) / len(kpts)
        lo, hi = (mu, hi) if n < nelectron else (lo, mu)
    return 0.5 * (lo + hi)


def gen_exchange_slakonet_ncl(
    atoms,
    magnetic_elements,
    spin_model=None,
    axis=(0.0, 0.0, 1.0),
    lambda_soc=None,
    soc_strength=1.0,
    kmesh=(7, 7, 7),
    emin=-14.0,
    emax=0.0,
    nz=100,
    Rcut=None,
    output_path="TB2J_results_ncl",
    efermi=None,
    keep_all_bands=True,
    description=None,
    verbose=True,
    **collinear_kwargs,
):
    """slakonet (+ on-site SOC) -> TB2J non-collinear exchange (ExchangeNCL).

    ``spin_model``: a converged collinear ``SlakonetSpinModel``; built with
    ``gen_exchange_slakonet``'s model (``collinear_kwargs``: model,
    magnetic_model, initial_moments, scf_kmesh, ...) when not given.
    ``axis``: spin quantisation axis. ``lambda_soc``: {Z: {l: zeta_Ha}},
    default ``SOC_ZETA_CM1``. ``soc_strength``: 0 switches SOC off (the
    result must then reproduce the collinear J_iso, with DMI = J_ani = 0).
    Returns (exchange, spin_model).
    """
    import contextlib

    from HamiltonIO.lcao_hamiltonian import LCAOHamiltonian
    from TB2J.exchange import ExchangeNCL

    from slakonet.tb2j_interface import _green_with_all_bands

    if spin_model is None:
        spin_model = _collinear_model(
            atoms, magnetic_elements, verbose=verbose, **collinear_kwargs
        )
    Rlist, HR, SR, labels = spinor_hamiltonian(
        spin_model, axis, lambda_soc, soc_strength
    )
    if efermi is None:  # SOC shifts the bands: recompute on the TB2J mesh
        efermi = _fermi_level(Rlist, HR, SR, spin_model.nelectron, kmesh)
    tb = LCAOHamiltonian(
        HR=HR,
        SR=SR,
        Rlist=Rlist,
        nbasis=len(labels),
        atoms=spin_model.atoms,
        nspin=1,
    )
    tb._name = "slakonet"
    tb.basis = labels
    tb.norb = len(labels) // 2
    tb.nspin = 2
    tb.nbasis = len(labels)

    if description is None:
        description = (
            "Input from slakonet with on-site SOC (one-shot, force theorem).\n"
            f"Spin axis: {list(np.round(axis, 4))}; SOC strength: {soc_strength}\n"
            f"Collinear moments (mu_B): {np.round(spin_model.moments, 3).tolist()}\n"
        )
    context = (
        _green_with_all_bands() if keep_all_bands else contextlib.nullcontext()
    )
    with context:
        exchange = ExchangeNCL(
            tbmodels=tb,
            atoms=spin_model.atoms,
            basis=labels,
            efermi=efermi,
            magnetic_elements=list(magnetic_elements),
            kmesh=list(kmesh),
            emin=emin,
            emax=emax,
            nz=nz,
            Rcut=Rcut,
            output_path=output_path,
            description=description,
        )
        exchange.run(path=output_path)
    return exchange, spin_model


def _collinear_model(
    atoms,
    magnetic_elements,
    model=None,
    magnetic_model="hubbard",
    initial_moments=None,
    scf_kmesh=(8, 8, 8),
    verbose=True,
    **kw,
):
    if magnetic_model == "hubbard":
        from slakonet.hubbard_spin import build_hubbard_model

        return build_hubbard_model(
            atoms,
            model,
            initial_moments=initial_moments,
            magnetic_elements=magnetic_elements,
            kmesh=scf_kmesh,
            verbose=verbose,
            **kw,
        )
    from slakonet.tb2j_interface import build_spin_model

    return build_spin_model(
        atoms,
        model,
        initial_moments=initial_moments,
        magnetic_elements=magnetic_elements,
        kmesh=scf_kmesh,
        verbose=verbose,
        **kw,
    )


def gen_exchange_slakonet_soc(
    atoms,
    magnetic_elements,
    spin_model=None,
    output_path="TB2J_results_soc",
    axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    **kwargs,
):
    """Full exchange tensor (J_iso, J_ani, DMI) with on-site SOC.

    Runs ``gen_exchange_slakonet_ncl`` with the spin axis along x, y and z
    (sub-folders ``axis_x``, ``axis_y``, ``axis_z``) and combines them with
    TB2J's merger (``TB2J.io_merge.merge``). A single orientation only gives
    the components transverse to the spins; the longitudinal ones are
    unphysical. Returns (merged SpinIO, spin_model); the merged result is
    also written to ``output_path`` (TB2J.pickle, exchange.out, ...).
    """
    import os

    from TB2J.io_exchange import SpinIO
    from TB2J.io_merge import merge

    if spin_model is None:
        keys = (
            "model",
            "magnetic_model",
            "initial_moments",
            "scf_kmesh",
            "verbose",
        )
        spin_model = _collinear_model(
            atoms,
            magnetic_elements,
            **{k: kwargs.pop(k) for k in keys if k in kwargs},
        )
    paths = []
    for name, axis in zip("xyz", axes):
        path = os.path.join(output_path, f"axis_{name}")
        gen_exchange_slakonet_ncl(
            atoms,
            magnetic_elements,
            spin_model=spin_model,
            axis=axis,
            output_path=path,
            **kwargs,
        )
        paths.append(path)
    # TB2J's merge() returns the per-axis inputs; the merged tensor is the
    # one it writes to write_path
    merge(*paths, write_path=output_path)
    return SpinIO.load_pickle(output_path), spin_model
