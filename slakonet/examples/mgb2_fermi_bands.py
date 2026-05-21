"""MgB2: band structure + DOS, 3D bands, and 2D / 3D Fermi surfaces.

MgB2 is the classic 39 K superconductor -- metallic, hexagonal
(P6/mmm), with a textbook Fermi surface (sigma tubes around Gamma-A
plus pi sheets), which makes it a good demo for the Fermi-surface
tools.

This script mirrors the analyses in the SlaKoNet web backend
(``custom_routes/slakonet.py``) as standalone, matplotlib-only
examples:

  1. band structure + DOS          -> MgB2_bands_dos.png
  2. 3D band structure over the BZ -> MgB2_bands3d.png
  3. 2D Fermi surface (contours)   -> MgB2_fermi2d.png
  4. 3D Fermi surface (isosurface) -> MgB2_fermi3d.png  (needs skimage)

All four reuse one helper that runs SlaKoNet on a Cartesian k-mesh.
The model is loaded once and reused.
"""

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from jarvis.core.atoms import Atoms

from slakonet.optim import default_model, kpts_to_klines
from slakonet.atoms import Geometry
from slakonet.main import generate_shell_dict_upto_Z65
from slakonet.predict_slakonet import plot_band_dos_atoms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------
def mgb2_atoms() -> Atoms:
    """MgB2 -- hexagonal P6/mmm, a=3.086 A, c=3.524 A."""
    a, c = 3.086, 3.524
    lattice = [
        [a, 0.0, 0.0],
        [-a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0],
        [0.0, 0.0, c],
    ]
    coords = [
        [0.0, 0.0, 0.0],          # Mg
        [1.0 / 3.0, 2.0 / 3.0, 0.5],  # B
        [2.0 / 3.0, 1.0 / 3.0, 0.5],  # B
    ]
    return Atoms(
        lattice_mat=lattice, coords=coords,
        elements=["Mg", "B", "B"], cartesian=False,
    )


# --------------------------------------------------------------------
def kmesh_eigs(atoms, model, shell_dict, nk, three_d=False):
    """Run SlaKoNet on a symmetric Cartesian k-grid (kz=0 plane, or a
    full 3D box) and return the grid axes + eigenvalue grid (eV,
    Fermi-referenced).

    Returns dict with ``kx``, ``ky`` (and ``kz`` if 3D) axis arrays,
    ``eig`` reshaped grid, ``nb`` band count, ``bandgap``.
    """
    recip = atoms.lattice.reciprocal_lattice().matrix  # includes 2*pi
    signs = [-1, 1]
    if three_d:
        corners = np.array([[s1 * .5, s2 * .5, s3 * .5]
                            for s1 in signs for s2 in signs
                            for s3 in signs])
    else:
        corners = np.array([[s1 * .5, s2 * .5, 0.0]
                            for s1 in signs for s2 in signs])
    cc = corners @ recip
    kx_max = float(np.abs(cc[:, 0]).max()) * 1.05
    ky_max = float(np.abs(cc[:, 1]).max()) * 1.05
    kx = np.linspace(-kx_max, kx_max, nk)
    ky = np.linspace(-ky_max, ky_max, nk)
    if three_d:
        kz_max = float(np.abs(cc[:, 2]).max()) * 1.05
        kz = np.linspace(-kz_max, kz_max, nk)
        gx, gy, gz = np.meshgrid(kx, ky, kz, indexing="ij")
        kpts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    else:
        gx, gy = np.meshgrid(kx, ky, indexing="ij")
        kpts = np.column_stack(
            [gx.ravel(), gy.ravel(), np.zeros(gx.size)]
        )

    geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
    klines = kpts_to_klines(kpts.tolist(), default_points=2)
    with torch.no_grad():
        props, ok = model.compute_multi_element_properties(
            geometry=geometry, shell_dict=shell_dict, klines=klines,
            get_fermi=True, with_eigenvectors=False, device=DEVICE,
        )
    assert ok, "SlaKoNet calculation failed"
    eig = props["eigenvalues"].detach().cpu().numpy().squeeze(0)
    nk_sk, nb = eig.shape

    out = {"nb": nb,
           "bandgap": float(props["bandgap"].detach().cpu().numpy())}
    if three_d:
        nkz = nk_sk // (nk * nk)
        npts = nk * nk * nkz
        out["eig"] = eig[:npts].reshape(nk, nk, nkz, nb)
        out["kx"], out["ky"], out["kz"] = kx, ky, kz[:nkz]
    else:
        nky = nk_sk // nk
        npts = nk * nky
        out["eig"] = eig[:npts].reshape(nk, nky, nb)
        out["kx"], out["ky"] = kx, ky[:nky]
    return out


# --------------------------------------------------------------------
def example_bands_dos(atoms, model):
    """1. Band structure + DOS along a high-symmetry k-path."""
    plot_band_dos_atoms(
        atoms=atoms, model=model, filename="MgB2_bands_dos.png",
        energy_range=(-12, 12),
    )
    print("[1] band structure + DOS  -> MgB2_bands_dos.png")


def example_bands3d(atoms, model, shell_dict, nk=24, window=4.0):
    """2. 3D band structure: bands near E_F as surfaces over the BZ."""
    r = kmesh_eigs(atoms, model, shell_dict, nk, three_d=False)
    gx, gy = np.meshgrid(r["kx"], r["ky"], indexing="ij")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    nplot = 0
    for ib in range(r["nb"]):
        b = r["eig"][:, :, ib]
        if b.min() <= window and b.max() >= -window:
            ax.plot_surface(gx, gy, b, alpha=0.7, linewidth=0)
            nplot += 1
    ax.set_xlabel("kx"); ax.set_ylabel("ky")
    ax.set_zlabel(r"E - E$_F$ (eV)")
    ax.set_title(f"MgB2 — {nplot} bands near E$_F$")
    fig.tight_layout()
    fig.savefig("MgB2_bands3d.png", dpi=180)
    plt.close(fig)
    print(f"[2] 3D band structure ({nplot} bands)  -> MgB2_bands3d.png")


def example_fermi2d(atoms, model, shell_dict, nk=48, window=0.5):
    """3. 2D Fermi surface: E=0 contours of bands crossing E_F."""
    r = kmesh_eigs(atoms, model, shell_dict, nk, three_d=False)
    gx, gy = np.meshgrid(r["kx"], r["ky"], indexing="ij")
    fig, ax = plt.subplots(figsize=(6, 6))
    crossed = 0
    for ib in range(r["nb"]):
        b = r["eig"][:, :, ib]
        if b.min() <= window and b.max() >= -window:
            ax.contour(gx, gy, b, levels=[0.0], linewidths=1.6)
            crossed += 1
    ax.set_aspect("equal")
    ax.set_xlabel("kx"); ax.set_ylabel("ky")
    ax.set_title(f"MgB2 — 2D Fermi surface (kz=0), {crossed} sheets")
    fig.tight_layout()
    fig.savefig("MgB2_fermi2d.png", dpi=180)
    plt.close(fig)
    print(f"[3] 2D Fermi surface ({crossed} sheets)  -> MgB2_fermi2d.png")


def example_fermi3d(atoms, model, shell_dict, nk=20, window=0.5):
    """4. 3D Fermi surface: isosurfaces at E=0 via marching cubes."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print("[4] 3D Fermi surface  -> SKIPPED (pip install scikit-image)")
        return
    r = kmesh_eigs(atoms, model, shell_dict, nk, three_d=True)
    kx, ky, kz = r["kx"], r["ky"], r["kz"]
    dx = kx[1] - kx[0]
    dy = ky[1] - ky[0]
    dz = (kz[1] - kz[0]) if len(kz) > 1 else 1.0
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    nsheets = 0
    for ib in range(r["nb"]):
        b = r["eig"][:, :, :, ib]
        if not (b.min() <= window and b.max() >= -window):
            continue
        try:
            verts, faces, _, _ = marching_cubes(
                b, level=0.0, spacing=(dx, dy, dz)
            )
        except (ValueError, RuntimeError):
            continue
        verts[:, 0] += kx[0]; verts[:, 1] += ky[0]; verts[:, 2] += kz[0]
        mesh = Poly3DCollection(verts[faces], alpha=0.5)
        ax.add_collection3d(mesh)
        nsheets += 1
    ax.set_xlim(kx[0], kx[-1]); ax.set_ylim(ky[0], ky[-1])
    ax.set_zlim(kz[0], kz[-1])
    ax.set_xlabel("kx"); ax.set_ylabel("ky"); ax.set_zlabel("kz")
    ax.set_title(f"MgB2 — 3D Fermi surface, {nsheets} sheets")
    fig.tight_layout()
    fig.savefig("MgB2_fermi3d.png", dpi=180)
    plt.close(fig)
    print(f"[4] 3D Fermi surface ({nsheets} sheets)  -> MgB2_fermi3d.png")


# --------------------------------------------------------------------
if __name__ == "__main__":
    model = default_model()
    shell_dict = generate_shell_dict_upto_Z65(model=model)
    atoms = mgb2_atoms()
    print(f"MgB2: {atoms.num_atoms} atoms, formula "
          f"{atoms.composition.reduced_formula}")

    example_bands_dos(atoms, model)
    example_bands3d(atoms, model, shell_dict)
    example_fermi2d(atoms, model, shell_dict)
    example_fermi3d(atoms, model, shell_dict)
    print("done")
