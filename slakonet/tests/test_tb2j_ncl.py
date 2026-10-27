"""Tests for the SOC / non-collinear slakonet -> TB2J workflow."""

import numpy as np
import pytest
from ase.build import bulk

from slakonet.tb2j_ncl import (
    SOC_ZETA_CM1,
    default_lambda_soc,
    soc_onsite,
    spinor_hamiltonian,
)


@pytest.fixture(scope="module")
def nio_model():
    from slakonet.optim import default_model
    from slakonet.tb2j_interface import build_spin_model

    nio = bulk("NiO", "rocksalt", a=4.17)
    return nio, build_spin_model(
        nio,
        default_model(),
        stoner_I={28: {2: 0.12}},
        initial_moments=[2.0, 0.0],
        kmesh=(3, 3, 3),
        verbose=False,
    )


def test_soc_onsite_spectrum(nio_model):
    """Ni d shell: j = 5/2 at +zeta (x6) and j = 3/2 at -3 zeta/2 (x4)."""
    nio, sm = nio_model
    H = soc_onsite(sm.labels, nio)
    assert np.allclose(H, H.conj().T)
    d = [i for i, lab in enumerate(sm.labels) if lab.startswith("Ni1|d")]
    idx = np.array([2 * i + s for i in d for s in (0, 1)])
    zeta = SOC_ZETA_CM1["Ni"][2] / 219474.63 * 27.211386
    e = np.sort(np.linalg.eigvalsh(H[np.ix_(idx, idx)]))
    assert np.allclose(e, [-1.5 * zeta] * 4 + [zeta] * 6, atol=1e-10)
    assert default_lambda_soc()[53][1] > default_lambda_soc()[17][1]  # I > Cl


def test_spinor_hamiltonian_structure(nio_model):
    nio, sm = nio_model
    R, HR, SR, labels = spinor_hamiltonian(
        sm, axis=(1, 0, 0), soc_strength=0.0
    )
    assert HR.shape[1] == 2 * sm.HR_up.shape[1] and labels[1].endswith("|down")
    # spin along x: the up/down blocks are equal, the exchange is in the
    # off-diagonal (sigma_x) part
    assert np.allclose(HR[:, ::2, ::2], HR[:, 1::2, 1::2])
    assert np.allclose(HR[:, ::2, 1::2].real, 0.5 * (sm.HR_up - sm.HR_dn))
    # along z it reduces to the two collinear channels
    R, HR, SR, _ = spinor_hamiltonian(sm, axis=(0, 0, 1), soc_strength=0.0)
    assert np.allclose(HR[:, ::2, ::2], sm.HR_up) and np.allclose(
        HR[:, 1::2, 1::2], sm.HR_dn
    )


def test_soc_zero_reproduces_collinear(nio_model, tmp_path):
    pytest.importorskip("TB2J")
    from TB2J.io_exchange import SpinIO

    from slakonet.tb2j_interface import gen_exchange_slakonet
    from slakonet.tb2j_ncl import gen_exchange_slakonet_soc

    nio, sm = nio_model
    ex_cl, _ = gen_exchange_slakonet(
        nio,
        ["Ni"],
        spin_model=sm,
        kmesh=(3, 3, 3),
        nz=30,
        Rcut=4.3,
        output_path=str(tmp_path / "cl"),
        verbose=False,
    )
    for soc, out in ((0.0, "soc0"), (1.0, "soc1")):
        gen_exchange_slakonet_soc(
            nio,
            ["Ni"],
            spin_model=sm,
            soc_strength=soc,
            efermi=sm.efermi,
            kmesh=(3, 3, 3),
            nz=30,
            Rcut=4.3,
            output_path=str(tmp_path / out),
            verbose=False,
        )
    d0 = SpinIO.load_pickle(str(tmp_path / "soc0"))
    d1 = SpinIO.load_pickle(str(tmp_path / "soc1"))
    for key, J in ex_cl.exchange_Jdict.items():
        assert d0.exchange_Jdict[key] == pytest.approx(J, rel=1e-6, abs=1e-9)
        assert np.abs(d0.Jani_dict[key]).max() < 1e-9
        assert np.abs(d0.dmi_ddict[key]).max() < 1e-9
        # SOC on: rocksalt Ni-Ni bonds have an inversion centre -> no DMI
        assert np.abs(d1.dmi_ddict[key]).max() < 1e-8
