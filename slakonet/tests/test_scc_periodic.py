"""Tests for the periodic SCC (Ewald DFTB gamma, occupied-shell U, Anderson
mixing) and its legacy mode."""

import numpy as np
import pytest
import torch
from ase.build import bulk
from ase.spacegroup import crystal

from slakonet.atoms import Geometry
from slakonet.main import SimpleDftb
from slakonet.optim import default_model
from slakonet.scc import (
    AndersonMixer,
    _slater_short_range,
    atom_U_from_skf,
    build_gamma_matrix_periodic,
)

BOHR = 1.0 / 0.529177210903


@pytest.fixture(scope="module")
def model():
    return default_model()


def test_ewald_madelung_nacl():
    nacl = bulk("NaCl", "rocksalt", a=5.64)
    U = torch.tensor([50.0, 50.0])  # short-range part negligible
    g = build_gamma_matrix_periodic(
        torch.tensor(nacl.positions * BOHR),
        torch.tensor(nacl.cell.array * BOHR),
        U,
    ).numpy()
    q = np.array([1.0, -1.0])
    energy = 0.5 * q @ (g - np.diag(U.numpy())) @ q
    r0 = 5.64 / 2 * BOHR
    assert abs(-energy * r0 - 1.747565) < 1e-5


def test_slater_gamma_onsite_limit():
    U = 0.4
    tau = 16 / 5 * U
    r = np.array([1e-6])
    assert abs(1 / r[0] - _slater_short_range(r, tau, tau)[0] - U) < 1e-6


def test_occupied_shell_U(model):
    skfs = model.get_updated_skfs()
    shells = {8: [0, 1, 2], 28: [0, 1, 2]}
    U_O = skfs["O-O"].to_dict()["atomic_data"]["hubbard_us"]
    assert atom_U_from_skf(skfs, 8, shells) == pytest.approx(U_O[1])  # p
    assert atom_U_from_skf(skfs, 8, shells, legacy=True) == pytest.approx(
        U_O[2]
    )  # d
    U_Ni = skfs["Ni-Ni"].to_dict()["atomic_data"]["hubbard_us"]
    assert atom_U_from_skf(skfs, 28, shells) == pytest.approx(U_Ni[2])


def test_anderson_mixer_linear_fixed_point():
    rng = np.random.default_rng(0)
    A = torch.tensor(rng.normal(size=(6, 6)) * 0.3)
    b = torch.tensor(rng.normal(size=6))
    exact = torch.linalg.solve(torch.eye(6, dtype=A.dtype) - A, b)
    mixer, x = AndersonMixer(alpha=0.5), torch.zeros(6, dtype=torch.float64)
    for _ in range(40):
        x = mixer(x, A @ x + b)
    assert torch.allclose(x, exact, atol=1e-8)


def _scc_calc(atoms, model, legacy, kpts=(3, 3, 4)):
    calc = SimpleDftb(
        Geometry.from_ase_atoms([atoms]),
        model,
        kpoints=torch.tensor(kpts),
        device="cpu",
        compute_forces=False,
        include_dos_data=False,
        repulsive=False,
        use_scc=True,
        scc_max_iter=300,
        scc_legacy=legacy,
    )
    calc.calculate()
    return calc._scc_info


def test_periodic_scc_converges_rutile_fef2(model):
    fef2 = crystal(
        ["Fe", "F"],
        basis=[(0, 0, 0), (0.3016, 0.3016, 0)],
        spacegroup=136,
        cellpar=[4.697, 4.697, 3.309, 90, 90, 90],
    )
    info = _scc_calc(fef2, model, legacy=False)
    assert info["converged"] and info["n_iter"] < 40
    dq = info["delta_q"].numpy()
    fe = [i for i, s in enumerate(fef2.get_chemical_symbols()) if s == "Fe"]
    f = [i for i, s in enumerate(fef2.get_chemical_symbols()) if s == "F"]
    assert np.ptp(dq[fe]) < 1e-4 and np.ptp(dq[f]) < 1e-4  # equivalent sites
    assert dq[fe].mean() < 0 < dq[f].mean()  # Fe -> F transfer
    assert abs(dq.sum()) < 1e-3  # Fermi-level bisection tolerance


def test_legacy_scc_unchanged(model):
    """Legacy mode reproduces the original NiO result (Klopman-Ohno gamma,
    U from the highest-l shell, linear mixing), obtained with the unmodified
    scc.py on this structure and mesh."""
    # O at fractional (1/2, 1/2, 1/2): the legacy cluster gamma (no periodic
    # images) depends on where the atoms sit inside the cell.
    nio = bulk("NiO", "rocksalt", a=4.1863375874)
    nio.set_scaled_positions([[0, 0, 0], [0.5, 0.5, 0.5]])
    info = _scc_calc(nio, model, legacy=True, kpts=(5, 5, 5))
    assert info["converged"]
    assert info["delta_q"].numpy() == pytest.approx(
        [-0.06657, 0.06656], abs=2e-4
    )
