"""
NEGF quantum transport for SlakoNet
===================================

Landauer-Buttiker transmission through a two-probe geometry, built from
SlakoNet Slater-Koster Hamiltonians in the same way TranSIESTA/TBtrans
build theirs from a SIESTA density matrix:

* the *electrode* is a bulk principal layer (PL), periodic along the
  transport axis.  Its real-space blocks ``H00``/``H01`` are obtained by
  inverse-Fourier-transforming ``H(k)`` over a uniform k-line along the
  transport axis.  The transform is exact as long as the PL is longer
  than the Slater-Koster cutoff, which is checked (``R = +-2`` blocks
  must vanish).
* the *device* supercell contains an integer number of those PLs.  Its
  ``R = 0`` block is the scattering-region Hamiltonian with the periodic
  wrap-around removed.
* semi-infinite leads enter through Lopez-Sancho/Lopez-Sancho surface
  Green functions, giving the self-energies on the first and last PL.

The transverse Brillouin zone is sampled explicitly, so the transmission
is ``T(E) = sum_k w_k Tr[Gamma_L G Gamma_R G^dag]`` exactly as TBtrans
reports it.

This is a *non-self-consistent* (equilibrium, zero-bias) transport
calculator: it is the SlakoNet analogue of TBtrans run on a converged
Hamiltonian, not of the TranSIESTA NEGF-SCF cycle.

Author: Kamal Choudhary (NIST/JHU)
"""

from __future__ import annotations

import numpy as np
import torch

from slakonet.atoms import Geometry
from slakonet.main import SimpleDftb
from slakonet.slaterkoster import hs_matrix

HARTREE_TO_EV = 27.211
# 2e^2/h in siemens (spin degeneracy included)
G0_SIEMENS = 7.748091729e-5
KB_EV_PER_K = 8.617333262e-5


# --------------------------------------------------------------------------
# Slater-Koster matrices at arbitrary k
# --------------------------------------------------------------------------
def _make_calc(atoms, model, cutoff=10.0, device=None):
    """A SimpleDftb whose only purpose is to own the basis/feeds/Periodic."""
    geom = Geometry.from_ase_atoms([atoms])
    return SimpleDftb(
        geom,
        model=model,
        cutoff=cutoff,
        kpoints=torch.tensor([[1, 1, 1]]),
        compute_forces=False,
        include_dos_data=False,
        device=device,
    )


def hs_at_kpoints(calc, kpts_frac):
    """H(k), S(k) for an explicit list of fractional k-points.

    Returns two ``[n_orb, n_orb, n_k]`` complex tensors; H is in eV.
    """
    per = calc.periodic
    kp = torch.as_tensor(
        np.asarray(kpts_frac, dtype=float), dtype=per.cellvec.dtype
    ).unsqueeze(0)
    nk = kp.shape[1]
    per.kpoints = kp
    per.n_kpoints = torch.tensor([nk])
    per.k_weights = torch.ones(1, nk, dtype=kp.dtype) / nk

    H = hs_matrix(per, calc.basis, calc.h_feed)[0] * HARTREE_TO_EV
    S = hs_matrix(per, calc.basis, calc.s_feed)[0]
    return H.detach(), S.detach()


def real_space_blocks(calc, axis, kt, n_r=5):
    """Blocks ``H(R a_axis)``, ``S(R a_axis)`` at fixed transverse k.

    ``kt`` are the two transverse fractional k components, in the order of
    the remaining axes.  The inverse DFT over ``n_r`` Gamma-centred points
    along ``axis`` is exact when the interaction range is below
    ``n_r // 2`` cells.
    """
    other = [i for i in range(3) if i != axis]
    ks = []
    for n in range(n_r):
        k = [0.0, 0.0, 0.0]
        k[axis] = n / n_r
        k[other[0]], k[other[1]] = kt
        ks.append(k)

    Hk, Sk = hs_at_kpoints(calc, ks)
    nR = torch.arange(n_r, dtype=torch.float64)
    blocks = {}
    for R in range(-(n_r // 2), n_r // 2 + 1):
        ph = torch.exp(-2j * np.pi * nR * R / n_r).to(Hk.dtype) / n_r
        blocks[R] = (
            torch.einsum("ijk,k->ij", Hk, ph),
            torch.einsum("ijk,k->ij", Sk, ph),
        )
    return blocks


# --------------------------------------------------------------------------
# surface Green function
# --------------------------------------------------------------------------
def surface_green_function(
    z, h00, s00, h01, s01, side="right", tol=1e-12, max_iter=100
):
    """Lopez-Sancho decimation for a semi-infinite lead.

    ``z`` is a batch of complex energies ``[B]``; ``h00 ... s01`` are
    ``[n, n]``.  ``side='right'`` returns the surface Green function of the
    lead that extends towards **+R** (used for the right electrode);
    ``side='left'`` the one extending towards **-R**.

    Returns ``g_s`` with shape ``[B, n, n]``.
    """
    z = z.reshape(-1, 1, 1)
    # d = z S00 - H00 ;  u = z S01 - H01 (layer n -> n+1)
    d = z * s00 - h00
    u = z * s01 - h01
    # note: (n+1 -> n) coupling is  z S01^dag - H01^dag, *not* u^dag,
    # because z is complex.
    ubar = z * s01.conj().T - h01.conj().T

    if side == "right":
        a, b = u, ubar  # eps_s couples "forward"
    elif side == "left":
        a, b = ubar, u
    else:
        raise ValueError("side must be 'left' or 'right'")

    eps_s = d.clone()
    eps = d.clone()
    alpha = a.expand_as(d).clone()
    beta = b.expand_as(d).clone()

    ident = torch.eye(d.shape[-1], dtype=d.dtype, device=d.device)
    for _ in range(max_iter):
        g = torch.linalg.solve(eps, ident.expand_as(eps))
        agb = alpha @ g @ beta
        bga = beta @ g @ alpha
        eps_s = eps_s - agb
        eps = eps - agb - bga
        alpha = alpha @ g @ alpha
        beta = beta @ g @ beta
        if torch.max(torch.abs(alpha)) < tol:
            break

    return torch.linalg.solve(eps_s, ident.expand_as(eps_s))


# --------------------------------------------------------------------------
# main calculator
# --------------------------------------------------------------------------
class SlakoNetNEGF:
    """Zero-bias NEGF transmission for a SlakoNet two-probe geometry.

    Parameters
    ----------
    model
        A ``MultiElementSkfParameterOptimizer`` (e.g. from
        ``slakonet.optim.default_model``).
    device_atoms, elec_atoms : ase.Atoms
        Scattering-region supercell and the bulk electrode principal
        layer.  The device must begin and end with a copy of the
        electrode PL, in the same atom order.
    axis, elec_axis : int
        Transport lattice-vector index for the device and the electrode.
    cutoff : float
        Slater-Koster cutoff handed to SlakoNet (Bohr).
    eta : float
        Retarded-Green-function broadening (eV).
    torch_device : str
        ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        model,
        device_atoms,
        elec_atoms,
        axis=2,
        elec_axis=None,
        cutoff=10.0,
        eta=1e-4,
        n_r=5,
        torch_device=None,
        verbose=True,
    ):
        self.model = model
        self.axis = axis
        self.elec_axis = axis if elec_axis is None else elec_axis
        self.cutoff = cutoff
        self.eta = eta
        self.n_r = n_r
        self.verbose = verbose
        self.torch_device = torch_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device_atoms = device_atoms
        self.elec_atoms = elec_atoms

        self._calc_dev = _make_calc(device_atoms, model, cutoff)
        self._calc_elec = _make_calc(elec_atoms, model, cutoff)

        self.n_orb_dev = int(self._calc_dev.basis.n_orbitals.sum())
        self.n_orb_elec = int(self._calc_elec.basis.n_orbitals.sum())
        self.n_pl = self.n_orb_dev // self.n_orb_elec

        if self.verbose:
            print(
                f"  device: {len(device_atoms)} atoms, "
                f"{self.n_orb_dev} orbitals"
            )
            print(
                f"  electrode PL: {len(elec_atoms)} atoms, "
                f"{self.n_orb_elec} orbitals "
                f"({self.n_pl} PLs in device)"
            )

        self.fermi_energy = None

    # ------------------------------------------------------------------
    def compute_fermi_energy(self, kgrid=(1, 1, 1), kT=0.025):
        """Electrode Fermi level (eV), from a bulk SlakoNet calculation.

        This is the reference that TBtrans calls ``E = 0``.
        """
        geom = Geometry.from_ase_atoms([self.elec_atoms])
        calc = SimpleDftb(
            geom,
            model=self.model,
            cutoff=self.cutoff,
            kpoints=torch.tensor([list(kgrid)]),
            compute_forces=False,
            include_dos_data=False,
            kT=kT,
        )
        calc.calculate()
        self.fermi_energy = float(calc.fermi_energy.item())
        if self.verbose:
            print(
                f"  electrode E_F = {self.fermi_energy:.4f} eV "
                f"(k-grid {tuple(kgrid)})"
            )
        return self.fermi_energy

    # ------------------------------------------------------------------
    def _blocks(self, kt):
        """Electrode and device real-space blocks at transverse k ``kt``."""
        eb = real_space_blocks(self._calc_elec, self.elec_axis, kt, self.n_r)
        db = real_space_blocks(self._calc_dev, self.axis, kt, self.n_r)
        return eb, db

    def check_setup(self, kt=(0.0, 0.0)):
        """Diagnostics: PL locality and device/electrode consistency."""
        eb, db = self._blocks(kt)
        out = {}
        far = max(abs(R) for R in eb)
        out["elec_H_far"] = float(torch.abs(eb[far][0]).max())
        out["elec_H01"] = float(torch.abs(eb[1][0]).max())
        out["dev_H_far"] = float(torch.abs(db[far][0]).max())
        n = self.n_orb_elec
        # first diagonal PL block of the device vs bulk electrode H00
        out["H00_mismatch"] = float(
            torch.abs(db[0][0][:n, :n] - eb[0][0]).max()
        )
        out["H01_mismatch"] = float(
            torch.abs(db[0][0][:n, n : 2 * n] - eb[1][0]).max()
        )
        return out

    # ------------------------------------------------------------------
    def _transmission_single_k(self, energies_ev, kt, chunk=128):
        """T(E) at one transverse k-point."""
        eb, db = self._blocks(kt)
        dev = self.torch_device
        cdt = torch.complex128

        h00, s00 = (t.to(dev, cdt) for t in eb[0])
        h01, s01 = (t.to(dev, cdt) for t in eb[1])
        hD, sD = (t.to(dev, cdt) for t in db[0])

        n = self.n_orb_elec
        nD = self.n_orb_dev
        ident_n = torch.eye(n, dtype=cdt, device=dev)

        # columns of the identity picking the last PL (for the corner block)
        P = torch.zeros(nD, n, dtype=cdt, device=dev)
        P[nD - n :, :] = ident_n

        T = np.empty(len(energies_ev))
        for i0 in range(0, len(energies_ev), chunk):
            e = torch.as_tensor(
                energies_ev[i0 : i0 + chunk], dtype=torch.float64, device=dev
            )
            z = (e + 1j * self.eta).to(cdt)
            B = z.numel()

            gL = surface_green_function(z, h00, s00, h01, s01, side="left")
            gR = surface_green_function(z, h00, s00, h01, s01, side="right")

            zz = z.reshape(-1, 1, 1)
            u = zz * s01 - h01
            ubar = zz * s01.conj().T - h01.conj().T

            sig_L = ubar @ gL @ u  # on the first PL
            sig_R = u @ gR @ ubar  # on the last  PL

            gam_L = 1j * (sig_L - sig_L.conj().transpose(-1, -2))
            gam_R = 1j * (sig_R - sig_R.conj().transpose(-1, -2))

            A = zz * sD - hD
            A[:, :n, :n] -= sig_L
            A[:, nD - n :, nD - n :] -= sig_R

            # G[:, last block] only: solve A X = P
            X = torch.linalg.solve(A, P.expand(B, nD, n))
            G1N = X[:, :n, :]  # [B, n_L, n_R]

            t = torch.einsum(
                "bij,bjk,bkl,bil->b",
                gam_L,
                G1N,
                gam_R,
                G1N.conj(),
            )
            T[i0 : i0 + chunk] = t.real.cpu().numpy()

        return T

    # ------------------------------------------------------------------
    def transmission(
        self,
        energies_ev,
        kt_points=None,
        kt_weights=None,
        relative_to_fermi=True,
    ):
        """Transmission ``T(E)`` averaged over the transverse BZ.

        ``energies_ev`` are relative to the electrode Fermi level when
        ``relative_to_fermi`` (the TBtrans convention).
        ``kt_points`` is a list of ``(k_a, k_b)`` transverse fractional
        coordinates, ordered as the two non-transport axes.
        """
        energies_ev = np.asarray(energies_ev, dtype=float)
        if relative_to_fermi:
            if self.fermi_energy is None:
                raise RuntimeError(
                    "call compute_fermi_energy() first, or pass "
                    "relative_to_fermi=False"
                )
            abs_e = energies_ev + self.fermi_energy
        else:
            abs_e = energies_ev

        if kt_points is None:
            kt_points = [(0.0, 0.0)]
        if kt_weights is None:
            kt_weights = np.ones(len(kt_points)) / len(kt_points)
        kt_weights = np.asarray(kt_weights, dtype=float)

        T = np.zeros_like(abs_e)
        for i, (kt, w) in enumerate(zip(kt_points, kt_weights)):
            T += w * self._transmission_single_k(abs_e, kt)
            if self.verbose and (i + 1) % 10 == 0:
                print(f"    transverse k {i + 1}/{len(kt_points)}")
        return T

    # ------------------------------------------------------------------
    @staticmethod
    def conductance(T_at_ef):
        """Zero-bias conductance in S from ``T(E_F)``."""
        return G0_SIEMENS * T_at_ef

    @staticmethod
    def current(energies_ev, T, voltages, temperature=300.0):
        """Landauer current ``I(V)`` in amperes.

        ``I(V) = (2e^2/h) * int T(E) [f(E - eV/2) - f(E + eV/2)] dE`` with a
        symmetric bias drop.  ``T`` is the *zero-bias* transmission, so this
        is the coherent low-bias estimate; it does not include the
        rearrangement of the potential that a TranSIESTA NEGF-SCF cycle
        would produce at finite bias.
        """
        kT = KB_EV_PER_K * temperature

        def fermi(e, mu):
            return 1.0 / (1.0 + np.exp(np.clip((e - mu) / kT, -60, 60)))

        energies_ev = np.asarray(energies_ev, dtype=float)
        T = np.asarray(T, dtype=float)
        out = []
        for v in np.atleast_1d(voltages):
            window = fermi(energies_ev, +v / 2) - fermi(energies_ev, -v / 2)
            # G0 already carries e^2/h; the remaining factor of V comes from
            # integrating over the eV-wide bias window in eV units.
            trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            out.append(G0_SIEMENS * trapz(T * window, energies_ev))
        return np.array(out)


# --------------------------------------------------------------------------
def transverse_kpoints(n, axis, periodic_axis):
    """Gamma-centred transverse k-points for a transport axis.

    ``axis`` is the transport lattice vector, ``periodic_axis`` the one
    periodic transverse direction to sample (the third is vacuum and is
    held at Gamma).  Returns ``(kt_points, weights)`` with each ``kt``
    ordered as the two non-transport lattice vectors, i.e. exactly what
    :func:`real_space_blocks` expects.
    """
    other = [i for i in range(3) if i != axis]
    if periodic_axis not in other:
        raise ValueError("periodic_axis must differ from the transport axis")
    slot = other.index(periodic_axis)
    pts = []
    for i in range(n):
        pair = [0.0, 0.0]
        pair[slot] = i / n
        pts.append(tuple(pair))
    return pts, np.ones(n) / n
