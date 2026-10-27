"""Tests for the slakonet -> TB2J interface (slakonet.tb2j_interface)."""

import numpy as np
import pytest
import torch
from ase.build import bulk
from scipy.linalg import eigh

from slakonet.atoms import Geometry
from slakonet.main import SimpleDftb
from slakonet.optim import default_model
from slakonet import magnetism
from slakonet.tb2j_interface import (
    build_spin_model,
    extract_real_space_hs,
    orbital_labels,
    parse_stoner,
)

NIO_STONER = {28: {2: 0.12}}


@pytest.fixture(scope="module")
def model():
    return default_model()


@pytest.fixture(scope="module")
def nio():
    return bulk("NiO", "rocksalt", a=4.17)


def _calc(atoms, model, **kw):
    return SimpleDftb(
        Geometry.from_ase_atoms([atoms]),
        model,
        device="cpu",
        compute_forces=False,
        include_dos_data=False,
        repulsive=False,
        **kw,
    )


def test_real_space_hs_reproduces_bloch_matrices(nio, model):
    calc = _calc(nio, model, kpoints=torch.tensor([3, 3, 3]))
    calc.calculate()
    Rlist, HR, SR = extract_real_space_hs(calc)
    k = calc.kpoints[0].numpy()
    phase = np.exp(2j * np.pi * k @ Rlist.T)
    Hk = np.einsum("kr,rab->abk", phase, HR)
    Sk = np.einsum("kr,rab->abk", phase, SR)
    assert np.abs(Hk - calc._results["hamiltonian"][0].numpy()).max() < 1e-5
    assert np.abs(Sk - calc._results["overlap"][0].numpy()).max() < 1e-5
    # H(-R) = H(R)^T and S(R=0) has a unit diagonal
    index = {tuple(R): i for i, R in enumerate(Rlist)}
    for i, R in enumerate(Rlist):
        assert np.allclose(HR[index[tuple(-R)]], HR[i].T)
    assert np.allclose(np.diag(SR[index[(0, 0, 0)]]), 1.0, atol=1e-5)
    # the k-points of calc.periodic are restored
    assert calc.periodic.kpoints.shape[1] == 27


def test_orbital_labels(nio, model):
    calc = _calc(nio, model, kpoints=torch.tensor([1, 1, 1]))
    labels, atom_idx, ls = orbital_labels(calc, nio)
    assert len(labels) == int(calc.basis.orbs_per_atom.sum())
    assert (
        labels[0] == "Ni1|s" and labels[4] == "Ni1|dxy" and labels[9] == "O1|s"
    )
    assert atom_idx.tolist() == [0] * 9 + [1] * 9


def test_spin_model_matches_magnetism_module(nio, model):
    sm = build_spin_model(
        nio,
        model,
        stoner_I=NIO_STONER,
        magnetic_elements=["Ni"],
        kmesh=(4, 4, 4),
        verbose=False,
    )
    assert sm.converged
    assert 1.4 < sm.moments[0] < 2.0
    assert abs(sm.charges.sum() - sm.nelectron) < 1e-6

    calc = _calc(
        nio, model, kpoints=torch.tensor([3, 3, 3]), with_eigenvectors=True
    )
    calc.calculate()
    res = magnetism.compute_spin_polarized_bands(
        calc,
        stoner_I=NIO_STONER,
        initial_moments=torch.tensor(sm.moments),
        scf=False,
    )
    k = calc.kpoints[0].numpy()
    Hu, Hd, S = sm.bloch(k)
    for H, ref in ((Hu, res["eigenvalues_up"]), (Hd, res["eigenvalues_dn"])):
        e = np.array(
            [eigh(H[i], S[i], eigvals_only=True) for i in range(len(k))]
        )
        assert np.abs(e - ref.real.numpy().T).max() < 1e-3  # eV


def test_exchange_splitting_is_onsite_only(nio, model):
    sm = build_spin_model(
        nio,
        model,
        stoner_I=NIO_STONER,
        magnetic_elements=["Ni"],
        kmesh=(3, 3, 3),
        verbose=False,
    )
    diff = sm.HR_up - sm.HR_dn
    iR0 = int(np.where(~sm.Rlist.any(axis=1))[0][0])
    off = np.delete(diff, iR0, axis=0)
    assert np.abs(off).max() < 1e-12
    D = diff[iR0]
    assert np.abs(D - np.diag(np.diag(D))).max() < 1e-12
    # Delta = I_d * m on the Ni d-orbitals only (eV)
    ni_d = np.diag(D)[4:9]
    assert np.allclose(ni_d, ni_d[0]) and ni_d[0] < 0  # V_up = -I m / 2 < 0


def test_zero_stoner_is_non_magnetic(nio, model):
    zero = {28: {0: 0.0, 1: 0.0, 2: 0.0}, 8: {0: 0.0, 1: 0.0, 2: 0.0}}
    sm = build_spin_model(
        nio,
        model,
        stoner_I=zero,
        initial_moments=[2.0, 0.0],
        kmesh=(3, 3, 3),
        verbose=False,
    )
    assert np.abs(sm.moments).max() < 1e-4
    assert np.allclose(sm.HR_up, sm.HR_dn, atol=1e-6)


def test_symmetrize_keeps_equivalent_sites_equal(model):
    """Rutile FeF2 AFM: without symmetrization the Stoner SCF drifts into a
    charge/spin disproportionation of the two equivalent Fe sites."""
    from ase.spacegroup import crystal

    fef2 = crystal(
        ["Fe", "F"],
        basis=[(0, 0, 0), (0.3016, 0.3016, 0)],
        spacegroup=136,
        cellpar=[4.697, 4.697, 3.309, 90, 90, 90],
    )
    m0 = [
        2.0 if s == "Fe" and np.allclose(p, 0) else -2.0 if s == "Fe" else 0.0
        for s, p in zip(
            fef2.get_chemical_symbols(), fef2.get_scaled_positions()
        )
    ]
    fe = [i for i, s in enumerate(fef2.get_chemical_symbols()) if s == "Fe"]
    sm = build_spin_model(
        fef2,
        model,
        initial_moments=m0,
        kmesh=(3, 3, 4),
        symmetrize=True,
        verbose=False,
    )
    assert sm.converged
    assert abs(sm.moments[fe[0]] + sm.moments[fe[1]]) < 1e-3
    assert abs(sm.charges[fe[0]] - sm.charges[fe[1]]) < 1e-3


@pytest.mark.parametrize("legacy", [False, True])
def test_spin_scc_matches_slakonet_scc_modes(nio, model, legacy):
    calc = _calc(
        nio,
        model,
        kpoints=torch.tensor([3, 3, 3]),
        use_scc=True,
        scc_tol=1e-8,
        scc_max_iter=300,
        scc_legacy=legacy,
    )
    calc.calculate()
    zero = {28: {0: 0.0, 1: 0.0, 2: 0.0}, 8: {0: 0.0, 1: 0.0, 2: 0.0}}
    sm = build_spin_model(
        nio,
        model,
        stoner_I=zero,
        initial_moments=[0.0, 0.0],
        kmesh=(3, 3, 3),
        use_scc=True,
        scc_legacy=legacy,
        tol=1e-8,
        max_iter=300,
        verbose=False,
    )
    from slakonet.scc import reference_charges

    q_ref = reference_charges(
        torch.tensor(nio.numbers), calc.updated_skfs, calc.shell_dict
    ).numpy()
    assert (
        np.abs((sm.charges - q_ref) - calc._scc_info["delta_q"].numpy()).max()
        < 1e-4
    )


def test_calibrate_stoner_reaches_target(nio, model):
    from slakonet.tb2j_interface import calibrate_stoner

    stoner, sm = calibrate_stoner(
        nio, {"Ni": 1.8}, model, kmesh=(4, 4, 4), verbose=False
    )
    assert sm.converged
    assert abs(sm.moments[0] - 1.8) < 0.02
    assert 0.037 < stoner[28][2] < 0.3
    assert sm.stoner_I[28][2] == pytest.approx(stoner[28][2])


def test_insulator_J_independent_of_fermi_level_in_gap(model, tmp_path):
    """TB2J cuts bands above E_F + 5.1 eV; with a large exchange splitting
    that removes the empty minority d band and J then depends on where E_F
    sits in the gap. With all bands kept, J must not depend on it."""
    pytest.importorskip("TB2J")
    from ase.build import make_supercell
    from scipy.linalg import eigh
    from slakonet.tb2j_interface import gen_exchange_slakonet

    prim = bulk("MnO", "rocksalt", a=4.45)
    mno = make_supercell(
        prim, np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    )  # AFM-II
    frac = np.linalg.solve(prim.cell.T, mno.positions.T).T
    parity = np.rint(frac).astype(int).sum(axis=1) % 2
    m0 = [
        (2.0 if p == 0 else -2.0) if s == "Mn" else 0.0
        for s, p in zip(mno.get_chemical_symbols(), parity)
    ]
    sm = build_spin_model(
        mno,
        model,
        stoner_I={25: {2: 0.12}},
        initial_moments=m0,
        kmesh=(4, 4, 4),
        verbose=False,
    )
    from TB2J.kpoints import monkhorst_pack

    kpts = monkhorst_pack(
        [5, 5, 5], gamma_center=True
    )  # TB2J's mesh for kmesh=4
    Hu, Hd, S = sm.bloch(kpts)
    n = int(round(sm.nelectron))
    e = np.array(
        [
            np.sort(
                np.r_[
                    eigh(Hu[k], S[k], eigvals_only=True),
                    eigh(Hd[k], S[k], eigvals_only=True),
                ]
            )
            for k in range(len(kpts))
        ]
    )
    vbm, cbm = (
        e[:, n - 1].max(),
        e[:, n].min(),
    )  # band edges over the TB2J mesh
    assert cbm - vbm > 2.0

    def J_values(efermi):
        sm.efermi = efermi
        ex, _ = gen_exchange_slakonet(
            mno,
            ["Mn"],
            spin_model=sm,
            kmesh=(4, 4, 4),
            nz=40,
            Rcut=4.6,
            output_path=str(tmp_path / f"{efermi:.2f}"),
            verbose=False,
        )
        return np.array(
            [ex.exchange_Jdict[k] for k in sorted(ex.exchange_Jdict)]
        )

    low, high = J_values(vbm + 0.5), J_values(cbm - 0.5)
    assert np.abs(high).max() > 1e-5
    assert np.abs(low - high).max() < 0.02 * np.abs(high).max()


def test_fitted_stoner_table_per_site(model):
    """MnO -> Mn(ionic, 2+) entry; an explicit stoner_I wins over the table;
    stoner_table='atomic' restores the atomic values."""
    from slakonet.magnetic_sites import site_class
    from slakonet.stoner_tables import FITTED_STONER_I_D, fitted_site_stoner_I

    mno = bulk("MnO", "rocksalt", a=4.45)
    assert site_class(mno, 0) == ("Mn", "ionic", 2)
    assert fitted_site_stoner_I(mno) == {
        0: FITTED_STONER_I_D[("Mn", "ionic", 2)]
    }
    sm = build_spin_model(
        mno, model, initial_moments=[2.0, 0.0], kmesh=(3, 3, 3), verbose=False
    )
    assert sm.site_stoner_I == {0: FITTED_STONER_I_D[("Mn", "ionic", 2)]}
    sm = build_spin_model(
        mno,
        model,
        stoner_I={25: {2: 0.05}},
        initial_moments=[2.0, 0.0],
        kmesh=(3, 3, 3),
        verbose=False,
    )
    assert sm.site_stoner_I == {} and sm.stoner_I[25][2] == 0.05
    sm = build_spin_model(
        mno,
        model,
        stoner_table="atomic",
        initial_moments=[2.0, 0.0],
        kmesh=(3, 3, 3),
        verbose=False,
    )
    assert sm.site_stoner_I == {}


def test_parse_stoner():
    assert parse_stoner("Ni:d=0.12, O:p=0.0") == {28: {2: 0.12}, 8: {1: 0.0}}
    assert parse_stoner("") == {}
    with pytest.raises(ValueError):
        parse_stoner("Ni=0.1")


def test_tb2j_end_to_end_nio(nio, model, tmp_path):
    pytest.importorskip("TB2J")
    from slakonet.tb2j_interface import gen_exchange_slakonet

    out = tmp_path / "TB2J_results"
    ex, sm = gen_exchange_slakonet(
        nio,
        ["Ni"],
        model=model,
        magnetic_model="stoner",
        stoner_I=NIO_STONER,
        scf_kmesh=(4, 4, 4),
        kmesh=(4, 4, 4),
        nz=30,
        Rcut=4.3,
        output_path=str(out),
        verbose=False,
    )
    assert (out / "exchange.out").exists()
    J = {}
    for (R, i, j), val in ex.exchange_Jdict.items():
        d = round(float(np.linalg.norm(np.array(R) @ nio.cell.array)), 2)
        J.setdefault(d, []).append(val)
    d1, d2 = sorted(J)[:2]
    # NiO: J1 ferromagnetic, J2 (180 deg superexchange) antiferromagnetic
    # and dominant; all bonds of a shell equal by symmetry.
    assert np.mean(J[d1]) > 0 and np.mean(J[d2]) < 0
    assert abs(np.mean(J[d2])) > abs(np.mean(J[d1]))
    assert np.ptp(J[d1]) < 1e-3 * abs(np.mean(J[d1]))
