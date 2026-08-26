"""wannier90_hr.dat export: the file must reproduce the model's bands.

The export is only correct if two things hold, and neither is obvious
from a "does it run" check. The basis has to be orthogonalised, because
hr.dat assumes S = I and a non-orthogonal H written directly gives
plausible but wrong bands. And the mesh has to be large enough for the
*orthogonalised* blocks, which reach further than the Slater-Koster
interaction does.

So the tests parse the written file back the way a consumer would and
compare eigenvalues against the generalised solve, at k-points that
were never on the storage mesh -- which is also what catches a
row/column transposition in the writer.
"""

import numpy as np
import pytest
import torch
from ase.build import bulk

from slakonet.optim import default_model
from slakonet.hr_export import (
    check,
    hr_auto,
    hr_from_model,
    lowdin,
    write_hr,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = default_model(model_name="slakonet_v1a_full").float()


def _si():
    return bulk("Si", "diamond", a=5.43)


def _read_hr(path):
    """Parse wannier90_hr.dat exactly as an external consumer would."""
    lines = open(path).read().split("\n")
    n_wann, n_r = int(lines[1]), int(lines[2])
    deg, i = [], 3
    while len(deg) < n_r:
        deg += [int(x) for x in lines[i].split()]
        i += 1
    blocks = {}
    for ln in lines[i:]:
        p = ln.split()
        if len(p) != 7:
            continue
        r = (int(p[0]), int(p[1]), int(p[2]))
        m, n = int(p[3]) - 1, int(p[4]) - 1
        blk = blocks.setdefault(r, np.zeros((n_wann, n_wann), complex))
        blk[m, n] = float(p[5]) + 1j * float(p[6])
    return n_wann, np.array(deg), blocks


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    hr, rs, nw, mesh, edge = hr_auto(
        _si(), model, device=DEVICE, edge_tol=1e-4, verbose=False
    )
    path = tmp_path_factory.mktemp("hr") / "wannier90_hr.dat"
    write_hr(str(path), hr, rs, nw)
    return str(path), hr, rs, nw, edge


def test_lowdin_preserves_generalised_eigenvalues():
    """S^-1/2 H S^-1/2 must have the spectrum of H c = e S c."""
    from slakonet.negf import _make_calc, hs_at_kpoints

    calc = _make_calc(_si(), model, cutoff=10.0, device=DEVICE)
    kpts = np.array([[0.0, 0.0, 0.0], [0.3, 0.1, 0.7]])
    h, s = hs_at_kpoints(calc, kpts)
    h = np.asarray(h.detach().cpu().numpy())
    s = np.asarray(s.detach().cpu().numpy())
    if h.shape[-1] == len(kpts):
        h, s = np.transpose(h, (2, 0, 1)), np.transpose(s, (2, 0, 1))
    for q in range(len(kpts)):
        ref = np.sort(np.real(np.linalg.eigvals(np.linalg.solve(s[q], h[q]))))
        got = np.sort(np.linalg.eigvalsh(lowdin(h[q], s[q])))
        assert np.abs(ref - got).max() < 1e-3


def test_mesh_grows_past_the_hamiltonian_estimate(exported):
    """The H(R) mesh is not sufficient for H'(R)."""
    _, _, _, _, edge = exported
    assert edge <= 1e-4
    # the un-grown mesh leaves real weight on the star edge
    hr, rs, _, _ = hr_from_model(_si(), model, mesh=[5, 5, 5], device=DEVICE)
    naive = np.abs(hr[np.abs(rs).max(axis=1) == np.abs(rs).max()]).max()
    assert naive > 1e-3


def test_written_file_round_trips(exported):
    path, hr, rs, nw, _ = exported
    n_read, deg, blocks = _read_hr(path)
    assert n_read == nw
    assert len(blocks) == len(rs)
    assert set(deg.tolist()) == {1}


def test_bands_from_file_match_generalised_solve(exported):
    """Off-mesh k-points, so a transposed writer cannot pass."""
    from slakonet.negf import _make_calc, hs_at_kpoints

    path, _, _, _, _ = exported
    _, _, blocks = _read_hr(path)
    rs = np.array(list(blocks.keys()))
    mats = np.stack([blocks[tuple(r)] for r in rs])

    kpts = np.array([[0.5, 0.0, 0.5], [0.3, 0.1, 0.7], [0.25, 0.25, 0.25]])
    calc = _make_calc(_si(), model, cutoff=10.0, device=DEVICE)
    h, s = hs_at_kpoints(calc, kpts)
    h = np.asarray(h.detach().cpu().numpy())
    s = np.asarray(s.detach().cpu().numpy())
    if h.shape[-1] == len(kpts):
        h, s = np.transpose(h, (2, 0, 1)), np.transpose(s, (2, 0, 1))
    for q in range(len(kpts)):
        ref = np.sort(np.linalg.eigvalsh(lowdin(h[q], s[q])))
        phase = np.exp(2j * np.pi * (kpts[q] @ rs.T))
        got = np.sort(np.linalg.eigvalsh(np.einsum("n,nab->ab", phase, mats)))
        assert np.abs(ref - got).max() < 1e-3


def test_hermitian_star(exported):
    """H(-R) = H(R)^dagger, else the exported bands are not real."""
    path, _, _, _, _ = exported
    _, _, blocks = _read_hr(path)
    worst = 0.0
    for r, blk in blocks.items():
        mirror = tuple(-x for x in r)
        if mirror in blocks:
            worst = max(worst, np.abs(blocks[mirror] - blk.conj().T).max())
    assert worst < 1e-6


def test_check_reports_agreement(exported):
    _, hr, rs, _, _ = exported
    err, edge, big = check(hr, rs, _si(), model, device=DEVICE)
    assert err < 1e-3
    assert big > 1.0
