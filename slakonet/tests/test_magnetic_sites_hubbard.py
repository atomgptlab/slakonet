"""Tests for the automatic site classification and the +U spin model."""

import numpy as np
import pytest
from ase.build import bulk
from ase.spacegroup import crystal

from slakonet.magnetic_sites import (
    formal_oxidation,
    site_class,
    site_environment,
    spin_only_moment,
)

HARTREE_EV = 27.211386


def test_rocksalt_oxide():
    mno = bulk("MnO", "rocksalt", a=4.45)
    assert site_class(mno, 0) == ("Mn", "ionic", 2)
    assert spin_only_moment("Mn", 2) == 5.0
    assert spin_only_moment("Cu", 2) == 1.0
    assert spin_only_moment("Fe", 3) == 5.0


def test_persulfide_dimer_rule():
    """Pyrite-type MnS2 is Mn2+ with S2(2-) dimers, not Mn4+."""
    mns2 = crystal(
        ["Mn", "S"],
        basis=[(0, 0, 0), (0.401, 0.401, 0.401)],
        spacegroup=205,
        cellpar=[6.10, 6.10, 6.10, 90, 90, 90],
    )
    assert formal_oxidation(mns2, "Mn") == 2


def test_metal_environment():
    fe = bulk("Fe", "bcc", a=2.87)
    assert site_environment(fe, 0) == "metallic"
    assert formal_oxidation(fe, "Fe") is None


def test_site_details_geometry_and_spin():
    from slakonet.magnetic_sites import _atlas_physics, site_details

    if _atlas_physics() is None:
        pytest.skip("oxidation-state atlas not available")
    nio = bulk("NiO", "rocksalt", a=4.17)
    info = site_details(nio, 0)
    assert (info["ox"], info["cn"], info["geometry"]) == (2, 6, "octahedral")
    assert info["m_expected"] == 2.0
    # low-spin Fe2+ read off the short Fe-O bond; long bond -> high spin
    feo_ls = bulk("FeO", "rocksalt", a=2 * 1.98)
    feo_hs = bulk("FeO", "rocksalt", a=2 * 2.16)
    assert site_details(feo_ls, 0)["spin"] == "LS"
    assert site_details(feo_ls, 0)["m_expected"] == 0.0
    assert site_details(feo_hs, 0)["spin"] == "HS"
    assert site_details(feo_hs, 0)["m_expected"] == 4.0


@pytest.fixture(scope="module")
def model():
    from slakonet.optim import default_model

    return default_model()


def test_hubbard_model_nio(model):
    from slakonet.hubbard_spin import build_hubbard_model

    nio = bulk("NiO", "rocksalt", a=4.17)
    sm = build_hubbard_model(
        nio,
        model,
        hubbard_U={"Ni": 6.0 / HARTREE_EV},
        initial_moments=[2.0, 0.0],
        kmesh=(3, 3, 3),
        verbose=False,
    )
    assert sm.converged
    assert 0.8 < sm.moments[0] < 2.0
    assert abs(sm.charges.sum() - sm.nelectron) < 1e-6
    assert sm.hubbard_U == {0: pytest.approx(6.0 / HARTREE_EV)}
    # the spin splitting is on-site only, and inside the Ni d block
    diff = sm.HR_up - sm.HR_dn
    iR0 = int(np.where(~sm.Rlist.any(axis=1))[0][0])
    assert np.abs(np.delete(diff, iR0, axis=0)).max() < 1e-10
    d = [i for i, lab in enumerate(sm.labels) if lab.startswith("Ni1|d")]
    off_block = diff[iR0].copy()
    off_block[np.ix_(d, d)] = 0.0
    # outside the Ni d block only the (diagonal) Stoner term of other shells
    assert np.abs(off_block - np.diag(np.diag(off_block))).max() < 1e-10


def test_hubbard_zero_is_nonmagnetic(model):
    from slakonet.hubbard_spin import build_hubbard_model

    nio = bulk("NiO", "rocksalt", a=4.17)
    zero = {28: {0: 0.0, 1: 0.0, 2: 0.0}, 8: {0: 0.0, 1: 0.0, 2: 0.0}}
    sm = build_hubbard_model(
        nio,
        model,
        hubbard_U={"Ni": 0.0},
        stoner_I=zero,
        initial_moments=[2.0, 0.0],
        kmesh=(3, 3, 3),
        use_scc=False,
        verbose=False,
    )
    assert np.abs(sm.moments).max() < 1e-3


def test_hubbard_default_U_by_anion(model):
    from ase.spacegroup import crystal

    from slakonet.hubbard_spin import (
        ANION_U_EV,
        HARTREE_EV,
        UNIVERSAL_U_EV,
        build_hubbard_model,
    )

    fef2 = crystal(
        ["Fe", "F"],
        basis=[(0, 0, 0), (0.3016, 0.3016, 0)],
        spacegroup=136,
        cellpar=[4.697, 4.697, 3.309, 90, 90, 90],
    )
    m0 = [2.0 if s == "Fe" else 0.0 for s in fef2.get_chemical_symbols()]
    kw = dict(initial_moments=m0, kmesh=(2, 2, 3), max_iter=3, verbose=False)
    sm = build_hubbard_model(fef2, model, **kw)
    assert set(round(u * HARTREE_EV, 3) for u in sm.hubbard_U.values()) == {
        ANION_U_EV["F"]
    }
    sm = build_hubbard_model(fef2, model, u_table="universal", **kw)
    assert set(round(u * HARTREE_EV, 3) for u in sm.hubbard_U.values()) == {
        UNIVERSAL_U_EV
    }
