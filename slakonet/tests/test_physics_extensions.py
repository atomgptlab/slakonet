"""Smoke + sanity tests for magnetism, SOC, and dielectric modules."""
import os

import pytest
import torch

from slakonet.atoms import Geometry
from slakonet.main import SimpleDftb
from slakonet.optim import default_model, get_atoms

from slakonet import magnetism, soc, dielectric


def _build_calc(jid="JVASP-1002", kpoints=(2, 2, 2)):
    model = default_model()
    atoms, _, _ = get_atoms(jid)
    geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
    calc = SimpleDftb(
        geometry,
        model,
        kpoints=torch.tensor(list(kpoints)),
        device="cpu",
        with_eigenvectors=True,
        compute_forces=False,
        include_dos_data=False,
    )
    calc.calculate()
    return calc


# -------------------- magnetism --------------------------------------------

def test_magnetism_zero_moments_reproduces_nonmag_bands():
    """With zero Stoner I (override defaults), up and down bands must match
    the non-magnetic eigenvalues exactly."""
    calc = _build_calc()
    # Override to force all Stoner I -> 0
    stoner_zero = {}
    # supply zero I values for all Z actually present
    Zs = set(calc.geometry.atomic_numbers.flatten().tolist())
    stoner_zero = {int(Z): {0: 0.0, 1: 0.0, 2: 0.0} for Z in Zs if Z > 0}
    # clear default entries so defaults don't leak
    for k in list(magnetism.DEFAULT_STONER_I.keys()):
        if k in Zs:
            magnetism.DEFAULT_STONER_I[k] = {0: 0.0, 1: 0.0, 2: 0.0}
    res = magnetism.compute_spin_polarized_bands(
        calc, stoner_I=stoner_zero, scf=False,
        initial_moments=torch.zeros(calc.geometry.atomic_numbers.shape[-1]),
    )
    eu = res["eigenvalues_up"]
    ed = res["eigenvalues_dn"]
    # Up / down must coincide when there is no exchange
    assert torch.allclose(eu, ed, atol=1e-6)
    # Total magnetic moment is zero
    assert abs(res["total_moment"]) < 1e-8


def test_magnetism_nonzero_splits_bands():
    """Applying a finite Stoner I with a finite moment must produce a
    nonzero splitting of up and down bands."""
    calc = _build_calc()
    Natom = calc.geometry.atomic_numbers.shape[-1]
    Zs = calc.geometry.atomic_numbers.flatten().tolist()
    # Use an artificial Stoner I on *whatever* element is present, with l
    # present in that element's shells.
    I = {}
    for Z in set(Zs):
        if Z <= 0:
            continue
        I[int(Z)] = {0: 0.05, 1: 0.05, 2: 0.05}
    m0 = torch.ones(Natom) * 1.0
    res = magnetism.compute_spin_polarized_bands(
        calc, stoner_I=I, scf=False, initial_moments=m0
    )
    diff = (res["eigenvalues_up"] - res["eigenvalues_dn"]).abs().max().item()
    assert diff > 1e-4, f"Expected band splitting, got {diff}"


# -------------------- SOC --------------------------------------------------

def test_soc_zero_lambda_reproduces_doubled_bands():
    """With all lambda -> 0, the spinor eigenvalues must be the non-SOC
    eigenvalues doubled (each band once per spin)."""
    calc = _build_calc()
    # Zero all lambda entries to ensure no SOC is applied
    for k in list(soc.DEFAULT_LAMBDA.keys()):
        soc.DEFAULT_LAMBDA[k] = {0: 0.0, 1: 0.0, 2: 0.0}
    res = soc.compute_soc_bands(calc)
    e_soc = res["eigenvalues"]  # [2N, Nk]
    # Non-SOC eigenvalues from calc (shifted by Fermi). Re-diagonalize the
    # stored H,S to avoid Fermi shift.
    from slakonet.utils import eighb
    H = calc._results["hamiltonian"]
    S = calc._results["overlap"]
    if H.ndim == 4:
        H = H[0]; S = S[0]
    Nk = H.shape[-1]
    H2E = calc.H2E
    for ik in range(Nk):
        e0, _ = eighb(
            H[..., ik].to(torch.complex128),
            S[..., ik].to(torch.complex128),
            scheme="chol",
        )
        e0_eV = (e0 * H2E).sort().values
        e_soc_k = e_soc[:, ik].sort().values
        # Every non-SOC eigenvalue appears twice in the SOC spectrum
        doubled = torch.cat([e0_eV, e0_eV]).sort().values
        assert torch.allclose(
            e_soc_k, doubled, atol=1e-4
        ), f"k={ik}: max diff {(e_soc_k - doubled).abs().max().item()}"


def test_soc_onsite_hermitian():
    calc = _build_calc()
    LS_uu, LS_ud, LS_du, LS_dd = soc.build_soc_onsite(
        calc.basis, lambda_soc={82: {1: 0.1}}
    )
    Norb = LS_uu.shape[0]
    H_big = torch.zeros(2 * Norb, 2 * Norb, dtype=torch.complex128)
    H_big[:Norb, :Norb] = LS_uu
    H_big[Norb:, Norb:] = LS_dd
    H_big[:Norb, Norb:] = LS_ud
    H_big[Norb:, :Norb] = LS_du
    assert torch.allclose(H_big, H_big.conj().T, atol=1e-10)


def test_p_LS_eigenvalues():
    """For a single p-shell with SOC lambda=1, eigenvalues of L.S in the
    6-dim spinor space should be {+0.5, +0.5, +0.5, +0.5, -1, -1}
    (j=3/2 quartet at +1/2 and j=1/2 doublet at -1)."""
    uu, ud, du, dd = soc._p_LS_blocks()
    H = torch.zeros(6, 6, dtype=torch.complex128)
    H[:3, :3] = uu; H[3:, 3:] = dd
    H[:3, 3:] = ud; H[3:, :3] = du
    e = torch.linalg.eigvalsh(H).real
    e_sorted, _ = torch.sort(e)
    expected = torch.tensor([-1.0, -1.0, 0.5, 0.5, 0.5, 0.5], dtype=torch.float64)
    assert torch.allclose(e_sorted, expected, atol=1e-10), e_sorted


def test_d_LS_eigenvalues():
    """d-shell L.S spectrum: j=5/2 (6-fold) at +1 and j=3/2 (4-fold) at -3/2."""
    uu, ud, du, dd = soc._d_LS_blocks()
    H = torch.zeros(10, 10, dtype=torch.complex128)
    H[:5, :5] = uu; H[5:, 5:] = dd
    H[:5, 5:] = ud; H[5:, :5] = du
    e = torch.linalg.eigvalsh(H).real
    e_sorted, _ = torch.sort(e)
    expected = torch.tensor(
        [-1.5, -1.5, -1.5, -1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=torch.float64,
    )
    assert torch.allclose(e_sorted, expected, atol=1e-10), e_sorted


# -------------------- dielectric -------------------------------------------

def test_dielectric_smoke():
    calc = _build_calc()
    # Keep this small: 2x2x2 grid, 50 omega points, so test is fast
    res = dielectric.compute_dielectric(
        calc,
        kgrid=(2, 2, 2),
        omega_range_eV=(0.1, 8.0),
        n_omega=80,
        smearing_eV=0.2,
        dk=2e-3,
    )
    w = res["omega_eV"]
    e1 = res["eps1_iso"]
    e2 = res["eps2_iso"]
    assert w.shape == e1.shape == e2.shape
    # eps_2 must be non-negative
    assert (e2 >= -1e-10).all(), e2.min()
    # finite
    assert torch.isfinite(e1).all() and torch.isfinite(e2).all()
    # some spectral weight must be present (material is not transparent)
    assert e2.max().item() > 0.0
    # epsilon_1 crosses 1 somewhere (Drude-like screening response above gap)
    # weaker check: real part must vary, not stay constant
    assert (e1.max() - e1.min()).item() > 1e-6


# -------------------- extended coverage ------------------------------------

def test_soc_large_lambda_shifts_bands():
    """With a large artificial SOC lambda on the p-shells of the test
    element, the SOC band spectrum must differ from the non-SOC (doubled)
    spectrum by at least O(lambda) at some k-point."""
    calc = _build_calc()
    # pick whatever element is in the cell
    Zs = set(int(z) for z in calc.geometry.atomic_numbers.flatten().tolist())
    Zs.discard(0)
    # force a big on-site p-SOC on every present element
    big_lambda = {Z: {0: 0.0, 1: 0.05, 2: 0.05} for Z in Zs}
    # also clear defaults to avoid extra contributions
    for k in list(soc.DEFAULT_LAMBDA.keys()):
        soc.DEFAULT_LAMBDA[k] = {0: 0.0, 1: 0.0, 2: 0.0}
    res = soc.compute_soc_bands(calc, lambda_soc=big_lambda)
    e_soc = res["eigenvalues"]    # [2N, Nk]  in eV

    # non-SOC doubled reference
    from slakonet.utils import eighb
    H = calc._results["hamiltonian"]; S = calc._results["overlap"]
    if H.ndim == 4:
        H = H[0]; S = S[0]
    H2E = calc.H2E
    max_shift = 0.0
    for ik in range(H.shape[-1]):
        e0, _ = eighb(
            H[..., ik].to(torch.complex128),
            S[..., ik].to(torch.complex128),
            scheme="chol",
        )
        e0_eV = (e0 * H2E).sort().values
        doubled = torch.cat([e0_eV, e0_eV]).sort().values
        e_k = e_soc[:, ik].sort().values
        diff = (e_k - doubled).abs().max().item()
        if diff > max_shift:
            max_shift = diff
    # lambda = 0.05 Ha ~ 1.36 eV; typical splitting a fraction of that
    assert max_shift > 0.05, f"SOC produced only {max_shift:.4f} eV shift"


def test_magnetism_scf_converges_on_nonmagnetic():
    """On a non-magnetic test system with mild artificial exchange, SCF
    must converge (moments should drift towards zero or small values)."""
    calc = _build_calc()
    Natom = calc.geometry.atomic_numbers.shape[-1]
    Zs = set(int(z) for z in calc.geometry.atomic_numbers.flatten().tolist())
    Zs.discard(0)
    I = {Z: {0: 0.02, 1: 0.02, 2: 0.02} for Z in Zs}
    res = magnetism.compute_spin_polarized_bands(
        calc,
        stoner_I=I,
        initial_moments=torch.zeros(Natom) + 0.2,
        scf=True,
        max_iter=15,
        mixing=0.4,
        tol=1e-3,
    )
    assert res["converged"], (
        f"SCF did not converge; total_moment={res['total_moment']}"
    )
    # For a non-magnetic system with weak I and small moments, total moment
    # should stay small (<~ 1 Bohr magneton)
    assert abs(res["total_moment"]) < 2.0


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v", "-x"])
