"""High-level analysis helpers for slakonet.

These are framework-agnostic wrappers around slakonet's core calculation
routines. They take a jarvis.core.atoms.Atoms object (plus an optional
trained model) and return plain-Python dicts / BytesIO buffers that are
easy to pass through FastAPI, Flask, Streamlit, or plain scripts.

Public API:
    compute_bandstructure(atoms, ...)
    compute_bandstructure_3d(atoms, ...)
    compute_fermi_surface_2d(atoms, ...)
    compute_fermi_surface_3d(atoms, ...)

Lower-level shared helper:
    compute_kmesh_2d(atoms, ...)
"""
from __future__ import annotations

import io
import os
import uuid
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_model(model):
    if model is None:
        from slakonet.optim import default_model
        model = default_model()
    return model


def _to_list(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().tolist()
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


# ---------------------------------------------------------------------------
# 1) Bandstructure + DOS + (optional) PDOS plot with summary dict
# ---------------------------------------------------------------------------
def compute_bandstructure(
    atoms,
    model=None,
    energy_range: Tuple[float, float] = (-8.0, 8.0),
    filename: Optional[str] = None,
) -> Tuple[io.BytesIO, dict]:
    """Run SlakoNet and produce a band-structure + DOS PNG plus a summary dict.

    Parameters
    ----------
    atoms : jarvis.core.atoms.Atoms
    model : trained slakonet model (optional; defaults to default_model())
    energy_range : plot window around the Fermi level, in eV
    filename : if given, write the PNG here as well; otherwise only returned

    Returns
    -------
    (img_buffer, band_data) where img_buffer is a BytesIO containing a PNG and
    band_data is a dict with keys:
        formula, num_atoms, elements, bandgap, vbm, cbm, eigenvalues,
        dos_energies, dos_values, energy_range, atom_pdos?, pdos_energy_grid?
    """
    from slakonet.predict_slakonet import plot_band_dos_atoms

    model = _resolve_model(model)

    own_tempfile = filename is None
    tmp = filename or f"_slakonet_{uuid.uuid4().hex}.png"
    try:
        _fig, properties, atom_pdos, energy_grid, orbital_pdos, _plotly = \
            plot_band_dos_atoms(
                atoms=atoms,
                model=model,
                energy_range=list(energy_range),
                filename=tmp,
            )
        buf = io.BytesIO()
        with open(tmp, "rb") as f:
            buf.write(f.read())
        buf.seek(0)
    finally:
        if own_tempfile and os.path.exists(tmp):
            os.remove(tmp)

    band_gap = properties["bandgap"].detach().cpu().numpy().flatten().tolist()
    eigenvalues = properties["eigenvalues"].detach().cpu().numpy()
    dos_energies = (
        properties["dos_energy_grid_tensor"].detach().cpu().numpy().flatten().tolist()
    )
    dos_values = (
        properties["dos_values_tensor"].detach().cpu().numpy().flatten().tolist()
    )

    band_data = {
        "formula": atoms.composition.reduced_formula,
        "num_atoms": atoms.num_atoms,
        "elements": atoms.elements,
        "bandgap": float(band_gap[0] if isinstance(band_gap, list) else band_gap),
        "vbm": (
            float(properties["vbm"].detach().cpu().numpy())
            if "vbm" in properties
            else None
        ),
        "cbm": (
            float(properties["cbm"].detach().cpu().numpy())
            if "cbm" in properties
            else None
        ),
        "eigenvalues": eigenvalues[0].tolist(),
        "dos_energies": dos_energies,
        "dos_values": dos_values,
        "energy_range": list(energy_range),
    }
    if atom_pdos is not None and energy_grid is not None:
        band_data["atom_pdos"] = {k: _to_list(v) for k, v in atom_pdos.items()}
        band_data["pdos_energy_grid"] = _to_list(energy_grid)
    if orbital_pdos is not None:
        band_data["orbital_pdos"] = {
            a: {sh: _to_list(p) for sh, p in d.items()}
            for a, d in orbital_pdos.items()
        }

    return buf, band_data


# ---------------------------------------------------------------------------
# 2) Shared k-mesh driver on kz=0 plane (Cartesian grid)
# ---------------------------------------------------------------------------
def compute_kmesh_2d(atoms, model=None, nk_per_dim: int = 30) -> dict:
    """Run SlakoNet on a 2D Cartesian k-mesh at kz=0 covering the full BZ.

    Using a Cartesian rather than fractional grid avoids half-BZ artefacts
    from non-orthogonal reciprocal lattices.

    Returns a raw-result dict consumed by compute_bandstructure_3d and
    compute_fermi_surface_2d.
    """
    from slakonet.optim import kpts_to_klines
    from slakonet.atoms import Geometry
    from slakonet.main import generate_shell_dict_upto_Z65

    model = _resolve_model(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shell_dict = generate_shell_dict_upto_Z65(model=model)

    recip_lat = atoms.lattice.reciprocal_lattice().matrix  # 2 pi included

    corners_frac = np.array(
        [[s1 * 0.5, s2 * 0.5, 0.0] for s1 in (-1, 1) for s2 in (-1, 1)]
    )
    corners_cart = corners_frac @ recip_lat
    kx_max = float(np.abs(corners_cart[:, 0]).max()) * 1.05
    ky_max = float(np.abs(corners_cart[:, 1]).max()) * 1.05

    kx_1d = np.linspace(-kx_max, kx_max, nk_per_dim)
    ky_1d = np.linspace(-ky_max, ky_max, nk_per_dim)
    kx_grid, ky_grid = np.meshgrid(kx_1d, ky_1d, indexing="ij")
    kz_grid = np.zeros_like(kx_grid)

    kpoints_cart = np.column_stack(
        [kx_grid.ravel(), ky_grid.ravel(), kz_grid.ravel()]
    )
    nk_total = kpoints_cart.shape[0]

    geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
    klines = kpts_to_klines(kpoints_cart.tolist(), default_points=2)

    with torch.no_grad():
        props, success = model.compute_multi_element_properties(
            geometry=geometry,
            shell_dict=shell_dict,
            klines=klines,
            get_fermi=True,
            with_eigenvectors=False,
            device=device,
        )
    if not success:
        raise RuntimeError("SlakoNet calculation failed")

    eigenvalues_raw = props["eigenvalues"].detach().cpu().numpy().squeeze(0)
    nk_sk, nb = eigenvalues_raw.shape
    eigenvalues = eigenvalues_raw  # fermi_energy = 0

    nk_use = min(nk_sk, nk_total)
    nky_actual = nk_use // nk_per_dim
    n_pts = nk_per_dim * nky_actual

    kx_2d = kpoints_cart[:n_pts, 0].reshape(nk_per_dim, nky_actual)
    ky_2d = kpoints_cart[:n_pts, 1].reshape(nk_per_dim, nky_actual)
    eig_grid = eigenvalues[:n_pts].reshape(nk_per_dim, nky_actual, nb)

    a_lat = float(np.linalg.norm(atoms.lattice_mat[0]))
    k0 = 4 * np.pi / (3 * a_lat) if a_lat > 0 else 1.0
    bz_angles = np.linspace(0, 2 * np.pi, 7)
    bz_x = [float(k0 * np.cos(a)) for a in bz_angles]
    bz_y = [float(k0 * np.sin(a)) for a in bz_angles]

    return {
        "props": props,
        "eigenvalues": eigenvalues,
        "eig_grid": eig_grid,
        "kx_2d": kx_2d,
        "ky_2d": ky_2d,
        "nb": nb,
        "nk_per_dim": nk_per_dim,
        "nky_actual": nky_actual,
        "bz_x": bz_x,
        "bz_y": bz_y,
        "k0": k0,
        "bandgap": float(props["bandgap"].detach().cpu().numpy()),
        "vbm": (
            float(props["vbm"].detach().cpu().numpy()) if "vbm" in props else None
        ),
        "cbm": (
            float(props["cbm"].detach().cpu().numpy()) if "cbm" in props else None
        ),
    }


# ---------------------------------------------------------------------------
# 3) 3D band structure (2D kmesh x bands)
# ---------------------------------------------------------------------------
def compute_bandstructure_3d(atoms, model=None, nk_per_dim: int = 30) -> dict:
    """Return bandstructure data over a 2D kz=0 mesh, serialisable as JSON."""
    r = compute_kmesh_2d(atoms, model=model, nk_per_dim=nk_per_dim)
    return {
        "formula": atoms.composition.reduced_formula,
        "num_atoms": atoms.num_atoms,
        "elements": atoms.elements,
        "nk": r["nk_per_dim"],
        "nky": r["nky_actual"],
        "nbands": r["nb"],
        "fermi_energy": 0.0,
        "bandgap": r["bandgap"],
        "vbm": r["vbm"],
        "cbm": r["cbm"],
        "kx_grid": r["kx_2d"].tolist(),
        "ky_grid": r["ky_2d"].tolist(),
        "bands": [r["eig_grid"][:, :, ib].tolist() for ib in range(r["nb"])],
        "bz_x": r["bz_x"],
        "bz_y": r["bz_y"],
    }


# ---------------------------------------------------------------------------
# 4) 2D Fermi surface
# ---------------------------------------------------------------------------
def compute_fermi_surface_2d(
    atoms, model=None, nk_per_dim: int = 40, energy_window: float = 0.5
) -> dict:
    """2D Fermi-surface slice at kz=0, with band-resolved E(kx,ky) grids."""
    r = compute_kmesh_2d(atoms, model=model, nk_per_dim=nk_per_dim)
    eig_grid, nb, k0 = r["eig_grid"], r["nb"], r["k0"]

    fermi_bands = []
    band_info = []
    for ib in range(nb):
        bvals = eig_grid[:, :, ib]
        bmin, bmax = float(bvals.min()), float(bvals.max())
        crosses = bmin <= energy_window and bmax >= -energy_window
        if crosses:
            fermi_bands.append(ib)
        band_info.append(
            {"index": ib, "min": bmin, "max": bmax, "crosses_ef": crosses}
        )

    if not fermi_bands:
        dists = [
            (min(abs(bi["min"]), abs(bi["max"])), bi["index"]) for bi in band_info
        ]
        dists.sort()
        fermi_bands = [d[1] for d in dists[:2]]
        for fb in fermi_bands:
            band_info[fb]["crosses_ef"] = True

    K_angles = [np.pi / 6 + i * np.pi / 3 for i in range(6)]
    high_sym = {
        "Gamma": [0.0, 0.0],
        "K": [
            float(k0 * np.cos(K_angles[0])),
            float(k0 * np.sin(K_angles[0])),
        ],
        "Kp": [
            float(k0 * np.cos(K_angles[1])),
            float(k0 * np.sin(K_angles[1])),
        ],
        "M": [
            float((r["bz_x"][0] + r["bz_x"][1]) / 2),
            float((r["bz_y"][0] + r["bz_y"][1]) / 2),
        ],
    }

    return {
        "formula": atoms.composition.reduced_formula,
        "num_atoms": atoms.num_atoms,
        "elements": atoms.elements,
        "nk": r["nk_per_dim"],
        "nky": r["nky_actual"],
        "nbands": nb,
        "bandgap": r["bandgap"],
        "vbm": r["vbm"],
        "cbm": r["cbm"],
        "energy_window": energy_window,
        "fermi_bands": fermi_bands,
        "band_info": band_info,
        "kx_1d": r["kx_2d"][:, 0].tolist(),
        "ky_1d": r["ky_2d"][0, :].tolist(),
        "kx_grid": r["kx_2d"].tolist(),
        "ky_grid": r["ky_2d"].tolist(),
        "bands": [eig_grid[:, :, ib].tolist() for ib in range(nb)],
        "bz_x": r["bz_x"],
        "bz_y": r["bz_y"],
        "high_sym": high_sym,
    }


# ---------------------------------------------------------------------------
# 5) 3D Fermi surface via marching cubes
# ---------------------------------------------------------------------------
def compute_fermi_surface_3d(
    atoms, model=None, nk_per_dim: int = 20, energy_window: float = 0.5
) -> dict:
    """Full 3D Fermi isosurface via marching cubes on a Cartesian k-mesh.

    Requires scikit-image (`from skimage.measure import marching_cubes`).
    Returns mesh vertices + faces per Fermi-crossing band (Plotly-friendly).
    """
    from slakonet.optim import kpts_to_klines
    from slakonet.atoms import Geometry
    from slakonet.main import generate_shell_dict_upto_Z65
    from skimage.measure import marching_cubes

    model = _resolve_model(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shell_dict = generate_shell_dict_upto_Z65(model=model)

    recip_lat = atoms.lattice.reciprocal_lattice().matrix

    corners_frac = np.array(
        [
            [s1 * 0.5, s2 * 0.5, s3 * 0.5]
            for s1 in (-1, 1)
            for s2 in (-1, 1)
            for s3 in (-1, 1)
        ]
    )
    corners_cart = corners_frac @ recip_lat
    kx_max = float(np.abs(corners_cart[:, 0]).max()) * 1.05
    ky_max = float(np.abs(corners_cart[:, 1]).max()) * 1.05
    kz_max = float(np.abs(corners_cart[:, 2]).max()) * 1.05

    kx_1d = np.linspace(-kx_max, kx_max, nk_per_dim)
    ky_1d = np.linspace(-ky_max, ky_max, nk_per_dim)
    kz_1d = np.linspace(-kz_max, kz_max, nk_per_dim)
    kx_g, ky_g, kz_g = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing="ij")
    kpoints_cart = np.column_stack(
        [kx_g.ravel(), ky_g.ravel(), kz_g.ravel()]
    )
    nk_total = kpoints_cart.shape[0]

    geometry = Geometry.from_ase_atoms([atoms.ase_converter()])
    klines = kpts_to_klines(kpoints_cart.tolist(), default_points=2)

    with torch.no_grad():
        props, success = model.compute_multi_element_properties(
            geometry=geometry,
            shell_dict=shell_dict,
            klines=klines,
            get_fermi=True,
            with_eigenvectors=False,
            device=device,
        )
    if not success:
        raise RuntimeError("SlakoNet calculation failed")

    eigenvalues_raw = props["eigenvalues"].detach().cpu().numpy().squeeze(0)
    nk_sk, nb = eigenvalues_raw.shape
    eigenvalues = eigenvalues_raw

    bandgap = float(props["bandgap"].detach().cpu().numpy())
    vbm = (
        float(props["vbm"].detach().cpu().numpy()) if "vbm" in props else None
    )
    cbm = (
        float(props["cbm"].detach().cpu().numpy()) if "cbm" in props else None
    )

    nk_use = min(nk_sk, nk_total)
    nk3 = nk_per_dim
    nkz_actual = nk_use // (nk3 * nk3)
    n_pts = nk3 * nk3 * nkz_actual
    eig_grid = eigenvalues[:n_pts].reshape(nk3, nk3, nkz_actual, nb)

    dx = (2 * kx_max) / (nk3 - 1) if nk3 > 1 else 1.0
    dy = (2 * ky_max) / (nk3 - 1) if nk3 > 1 else 1.0
    dz = (
        (2 * kz_max) / (nkz_actual - 1) if nkz_actual > 1 else 1.0
    )

    fermi_bands = []
    band_info = []
    meshes = []

    for ib in range(nb):
        bvals = eig_grid[:, :, :, ib]
        bmin, bmax = float(bvals.min()), float(bvals.max())
        crosses = bmin <= energy_window and bmax >= -energy_window
        band_info.append(
            {"index": ib, "min": bmin, "max": bmax, "crosses_ef": crosses}
        )
        if not crosses:
            continue
        fermi_bands.append(ib)
        try:
            verts, faces, _n, _v = marching_cubes(
                bvals, level=0.0, spacing=(dx, dy, dz)
            )
            verts[:, 0] += -kx_max
            verts[:, 1] += -ky_max
            verts[:, 2] += -kz_max
            meshes.append(
                {
                    "band": ib,
                    "vertices_x": verts[:, 0].tolist(),
                    "vertices_y": verts[:, 1].tolist(),
                    "vertices_z": verts[:, 2].tolist(),
                    "faces_i": faces[:, 0].tolist(),
                    "faces_j": faces[:, 1].tolist(),
                    "faces_k": faces[:, 2].tolist(),
                    "n_vertices": len(verts),
                    "n_faces": len(faces),
                }
            )
        except Exception:
            pass

    if not fermi_bands:
        dists = [
            (min(abs(bi["min"]), abs(bi["max"])), bi["index"]) for bi in band_info
        ]
        dists.sort()
        fermi_bands = [d[1] for d in dists[:2]]

    a_lat = float(np.linalg.norm(atoms.lattice_mat[0]))
    k0 = 4 * np.pi / (3 * a_lat) if a_lat > 0 else 1.0
    bz_angles = np.linspace(0, 2 * np.pi, 7)
    bz_x = [float(k0 * np.cos(a)) for a in bz_angles]
    bz_y = [float(k0 * np.sin(a)) for a in bz_angles]

    return {
        "formula": atoms.composition.reduced_formula,
        "num_atoms": atoms.num_atoms,
        "elements": atoms.elements,
        "nk": nk_per_dim,
        "nkz": nkz_actual,
        "nbands": nb,
        "bandgap": bandgap,
        "vbm": vbm,
        "cbm": cbm,
        "fermi_bands": fermi_bands,
        "band_info": band_info,
        "meshes": meshes,
        "bz_x": bz_x,
        "bz_y": bz_y,
        "kx_range": [-kx_max, kx_max],
        "ky_range": [-ky_max, ky_max],
        "kz_range": [-kz_max, kz_max],
    }


__all__ = [
    "compute_bandstructure",
    "compute_bandstructure_3d",
    "compute_fermi_surface_2d",
    "compute_fermi_surface_3d",
    "compute_kmesh_2d",
]
