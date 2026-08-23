"""Both ASE calculators: energy/forces/stress plus the band structure,
DOS and Hamiltonian/overlap accessors.

Forces and stress are checked against finite differences of the model's
own energy, which is what catches unit-conversion and sign errors -- the
class of bug that a "does it run" test sails straight past.
"""

import numpy as np
import pytest
import torch
from ase.build import bulk

from slakonet.optim import default_model
from slakonet.main import SlakoNetCalculator
from slakonet.ase_calc import SlaKoNetCalculator

KPTS = [3, 3, 3]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = default_model().float()


def _si(displace=0.0):
    a = bulk("Si", "diamond", a=5.43)
    if displace:
        a.positions[1, 0] += displace
    return a


@pytest.fixture(scope="module")
def calc():
    return SlakoNetCalculator(model=model, kpoints_array=KPTS, device=DEVICE)


def test_energy_forces_stress_shapes(calc):
    atoms = _si(0.1)
    atoms.calc = calc
    assert np.isfinite(atoms.get_potential_energy())
    assert atoms.get_forces().shape == (2, 3)
    assert atoms.get_stress().shape == (6,)


def test_forces_match_finite_difference(calc):
    """Analytic forces must equal -dE/dx of the same energy."""
    base = _si(0.1)
    atoms = base.copy()
    atoms.calc = calc
    f_analytic = atoms.get_forces()[1, 0]

    h = 0.005

    def energy(shift):
        a = base.copy()
        a.positions[1, 0] += shift
        a.calc = calc
        return a.get_potential_energy()

    f_fd = -(energy(h) - energy(-h)) / (2 * h)
    assert f_analytic == pytest.approx(f_fd, abs=5e-3)


def test_stress_matches_finite_difference(calc):
    """Analytic stress must equal (1/V) dE/d(strain)."""
    atoms = _si()
    atoms.calc = calc
    s_analytic = atoms.get_stress()[0]
    volume = atoms.get_volume()

    d = 0.002

    def energy(eps):
        a = _si()
        m = np.eye(3)
        m[0, 0] += eps
        a.set_cell(a.cell @ m.T, scale_atoms=True)
        a.calc = calc
        return a.get_potential_energy()

    s_fd = (energy(d) - energy(-d)) / (2 * d) / volume
    assert s_analytic == pytest.approx(s_fd, abs=5e-4)


def test_symmetry_forces_vanish_with_kspacing():
    """Ideal diamond Si has zero forces by symmetry.

    A fixed coarse mesh leaves a large spurious residual, so this also
    guards the kspacing mesh selection.
    """
    c = SlakoNetCalculator(model=model, kspacing=0.25, device=DEVICE)
    atoms = _si()
    atoms.calc = c
    assert np.abs(atoms.get_forces()).max() < 0.02


def test_kpoints_for_scales_with_cell():
    c = SlakoNetCalculator(model=model, kspacing=0.30, device=DEVICE)
    small = c.kpoints_for(_si())
    large = c.kpoints_for(bulk("Si", "diamond", a=5.43, cubic=True).repeat(2))
    assert all(s >= l for s, l in zip(small, large))
    assert min(small) >= 1 and min(large) >= 1

    slab = bulk("Si", "diamond", a=5.43, cubic=True)
    slab.cell[2, 2] += 15.0
    slab.pbc = [True, True, False]
    assert c.kpoints_for(slab)[2] == 1  # no sampling along vacuum


def test_get_dos(calc):
    energies, dos = calc.get_dos(_si())
    assert energies.shape == dos.shape
    assert (dos >= 0).all()
    assert energies.min() < 0 < energies.max()


def test_get_bandstructure(calc):
    bs = calc.get_bandstructure(_si())
    assert bs["eigenvalues"].ndim == 2
    assert len(bs["labels"]) == len(bs["kpoints"])
    assert bs["bandgap"] >= 0.0
    assert bs["cbm"] >= bs["vbm"]


def test_get_HS_shape_and_hermiticity(calc):
    H, S = calc.get_HS(_si())
    nk = int(np.prod(KPTS))
    assert H.shape == S.shape
    assert H.shape[0] == nk
    assert H.shape[1] == H.shape[2]
    for k in (0, nk // 2, nk - 1):
        assert np.allclose(H[k], H[k].conj().T, atol=1e-6)
        assert np.allclose(S[k], S[k].conj().T, atol=1e-6)


def test_get_HS_reproduces_eigenvalues(calc):
    """eigh(H, S) * Hartree - E_F must give back the eigenvalues.

    This pins both the Hartree unit of H and the non-orthogonal
    (generalized) eigenproblem documented for get_HS.
    """
    import scipy.linalg as sla

    atoms = _si()
    atoms.calc = calc
    atoms.get_potential_energy()

    H, S = calc.get_HS(atoms)
    w = sla.eigh(H[0], S[0], eigvals_only=True) * 27.211
    w = w - calc.get_fermi_energy()

    reference = np.sort(calc.results["eigenvalues"][0][0])
    assert np.allclose(np.sort(w)[:6], reference[:6], atol=1e-3)


def test_ase_calc_agrees_with_main_calculator(calc):
    """The two calculator classes must give the same numbers."""
    other = SlaKoNetCalculator(model, kpoints=tuple(KPTS))
    a1 = _si(0.1)
    a1.calc = calc
    a2 = _si(0.1)
    a2.calc = other
    assert a1.get_potential_energy() == pytest.approx(
        a2.get_potential_energy(), abs=1e-4
    )
    assert np.allclose(a1.get_forces(), a2.get_forces(), atol=1e-4)


def test_model_reuse_across_structures(calc):
    """Switching elements must re-filter the SKF pairs, not go stale."""
    si = _si()
    si.calc = calc
    e_si = si.get_potential_energy()

    ge = bulk("Ge", "diamond", a=5.66)
    ge.calc = calc
    e_ge = ge.get_potential_energy()

    si2 = _si()
    si2.calc = calc
    assert si2.get_potential_energy() == pytest.approx(e_si, abs=1e-6)
    assert e_ge != pytest.approx(e_si, abs=1e-6)


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (SlaKoNetCalculator, {"model": model}),
        (SlakoNetCalculator, {"model": model}),
    ],
)
def test_unknown_kwarg_raises(cls, kwargs):
    """A keyword neither we nor ASE implement must not be swallowed.

    ASE's Calculator base ends in **kwargs and files anything it does
    not recognise into self.parameters, so an unimplemented keyword used
    to be accepted in silence -- `kspacing=` did nothing on
    SlaKoNetCalculator, leaving every structure on the fixed 3x3x3 mesh
    that puts a spurious 1.5 eV gap on fcc Al.
    """
    with pytest.raises(TypeError, match="unexpected keyword"):
        cls(kspaceing=0.25, **kwargs)  # deliberate typo


def test_kspacing_is_honoured_by_both_calculators():
    """kspacing must actually change the mesh, not just be accepted."""
    si = _si()
    coarse = SlaKoNetCalculator(model, kspacing=1.0, device=DEVICE)
    fine = SlaKoNetCalculator(model, kspacing=0.2, device=DEVICE)
    assert coarse.kpoints_for(si) != fine.kpoints_for(si)
    assert all(n >= 8 for n in fine.kpoints_for(si))

    other = SlakoNetCalculator(
        model=model, kspacing=0.2, kpoints_array=[1, 1, 1], device=DEVICE
    )
    assert other.kpoints_for(si) == fine.kpoints_for(si)


def test_metal_gap_vanishes_on_a_converged_mesh():
    """fcc Al is a metal; a coarse mesh invents a gap for it."""
    al = bulk("Al", "fcc", a=4.05)
    al.calc = SlaKoNetCalculator(
        model,
        kspacing=0.15,
        device=DEVICE,
        compute_forces=False,
        compute_stress=False,
    )
    al.get_potential_energy()
    assert al.calc.get_bandgap() == pytest.approx(0.0, abs=1e-6)


def _klines_for(atoms, line_density=10):
    from jarvis.core.atoms import ase_to_atoms
    from jarvis.core.kpoints import Kpoints3D
    from slakonet.optim import kpts_to_klines

    kp = Kpoints3D().kpath(ase_to_atoms(atoms), line_density=line_density)
    return kpts_to_klines(kp.kpts, default_points=2)


def test_get_HS_along_klines():
    """klines must give H/S at every point of the band path."""
    si = _si()
    kl = _klines_for(si)
    calc = SlaKoNetCalculator(model, klines=kl, device=DEVICE)
    H, S = calc.get_HS(si)
    n_k = len(kl) * 2  # default_points=2 per segment
    assert H.shape == (n_k, S.shape[-1], S.shape[-1])
    assert H.shape == S.shape
    # S must stay Hermitian positive-definite along the whole path
    assert np.allclose(S[0], S[0].conj().T, atol=1e-8)
    assert np.linalg.eigvalsh(S[0]).min() > 0


def test_klines_and_mesh_are_mutually_exclusive():
    """A band path and a BZ mesh cannot both be in force."""
    kl = _klines_for(_si())
    with pytest.raises(ValueError, match="only one of"):
        SlaKoNetCalculator(model, kpoints=KPTS, klines=kl, device=DEVICE)
    with pytest.raises(ValueError, match="only one of"):
        SlaKoNetCalculator(model, klines=kl, kspacing=0.2, device=DEVICE)


def test_klines_calculator_refuses_energies():
    """A band path is not a quadrature; energies from it would be wrong."""
    si = _si()
    calc = SlaKoNetCalculator(model, klines=_klines_for(si), device=DEVICE)
    si.calc = calc
    with pytest.raises(ValueError, match="cannot integrate"):
        si.get_potential_energy()
