"""End-to-end slakonet analysis from a single JID.

Pass a JARVIS-DFT id and it runs (and plots as interactive Plotly HTML):
    1. band structure + total DOS + atom/orbital PDOS
    2. 2D and 3D Fermi surface
    3. E-V curve (scan lattice constants + repulsive + SCC total energies)
    4. equation-of-state fit (Murnaghan/Birch) for V0, E0, B0
    5. collinear spin-polarized bands
    6. spin-orbit-coupled bands
    7. dielectric function eps_1(omega), eps_2(omega)
    8. SCC Mulliken charges at equilibrium (Delta q)
    9. ASE optimization of the atomic positions + cell

Every task_* function returns a dict with at minimum:
    {"fig": plotly.graph_objects.Figure | None,  # the plot
     "data": { ...numerical results... }}       # raw data for later processing
The returned object is a TaskResult subclass of dict that also forwards
`.show()` to `result["fig"].show()` so notebook cells stay one-liners.

Usage (script):
    python full_demo.py --jid JVASP-1002
    python full_demo.py --jid JVASP-1002 --skip soc,dielectric

Usage (notebook):
    from slakonet.examples.full_demo import run_all, task_bands_dos
    figs = run_all('JVASP-1002', show=True)
    # or
    r = task_bands_dos(atoms, model, out='demo'); r.show(); print(r['bandgap'])
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from ase.optimize import BFGS
import plotly.graph_objects as go

from slakonet.atoms import Geometry
from slakonet.main import SimpleDftb, SlakoNetCalculator
from slakonet.optim import default_model, get_atoms
from slakonet import magnetism, soc, dielectric
from slakonet.analysis import (
    compute_bandstructure,
    compute_fermi_surface_2d,
    compute_fermi_surface_3d,
)

H2E = 27.211


# ---------------------------------------------------------------------------
# TaskResult: dict that forwards .show() to its 'fig'
# ---------------------------------------------------------------------------
class TaskResult(dict):
    """Dict with a convenience .show() that forwards to self['fig'].show()."""

    def show(self, *args, **kwargs):
        fig = self.get("fig")
        if fig is None:
            print("(no plotly figure attached)")
            return None
        return fig.show(*args, **kwargs)


def _to_np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _high_symmetry_klines(atoms, line_density=20):
    """Return (klines, xticks, xtick_labels) for the high-symmetry k-path."""
    from jarvis.core.kpoints import Kpoints3D as Kpoints
    from slakonet.optim import kpts_to_klines
    from slakonet.predict_slakonet import _format_kpath_ticks
    kp = Kpoints().kpath(atoms, line_density=line_density)
    klines = kpts_to_klines(kp.kpts, default_points=2)
    xticks, xtick_labels = _format_kpath_ticks(kp.labels)
    xtick_labels = [
        s.replace(r"$\Gamma$", "Γ").replace("$", "") for s in xtick_labels
    ]
    return klines, xticks, xtick_labels, kp.kpts, kp.labels


# ---------------------------------------------------------------------------
# Individual tasks (all return a TaskResult dict with a .fig key)
# ---------------------------------------------------------------------------

def task_bands_dos(atoms, model, out):
    from slakonet.predict_slakonet import plot_band_dos_atoms
    png = f"{out}_bands.png"
    _fig, props, atom_pdos, egrid, orb_pdos, plotly_fig = plot_band_dos_atoms(
        atoms=atoms, model=model, energy_range=(-8, 8), filename=png,
    )
    print(f"  wrote {png} and {png.replace('.png', '.html')}")
    return TaskResult(
        fig=plotly_fig,
        bandgap=float(_to_np(props["bandgap"]).flatten()[0]),
        vbm=float(_to_np(props["vbm"])) if "vbm" in props else None,
        cbm=float(_to_np(props["cbm"])) if "cbm" in props else None,
        eigenvalues=_to_np(props["eigenvalues"]),
        dos_energies=_to_np(props["dos_energy_grid_tensor"]).flatten(),
        dos_values=_to_np(props["dos_values_tensor"]).flatten(),
        atom_pdos={k: _to_np(v) for k, v in (atom_pdos or {}).items()},
        orbital_pdos={
            a: {sh: _to_np(p) for sh, p in d.items()}
            for a, d in (orb_pdos or {}).items()
        },
        pdos_energy_grid=_to_np(egrid),
    )


def task_fermi2d(atoms, model, out, nk_per_dim=30, energy_window=1.0):
    res = compute_fermi_surface_2d(atoms, model=model,
                                   nk_per_dim=nk_per_dim,
                                   energy_window=energy_window)
    kx = np.array(res["kx_grid"]); ky = np.array(res["ky_grid"])
    fig = go.Figure()
    for ib in res["fermi_bands"][:3]:
        Z = np.array(res["bands"][ib])
        fig.add_trace(go.Contour(
            x=kx[:, 0], y=ky[0, :], z=Z,
            contours=dict(start=0, end=0, size=1),
            line=dict(width=2), showscale=False, name=f"band {ib}",
        ))
    fig.add_trace(go.Scatter(x=res["bz_x"], y=res["bz_y"], mode="lines",
                             line=dict(color="black"), name="BZ"))
    fig.update_layout(title=f"2D Fermi surface ({res['formula']})",
                      xaxis_title="kx", yaxis_title="ky",
                      width=700, height=650, template="plotly_white")
    path = f"{out}_fermi2d.html"
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  wrote {path}  ({len(res['fermi_bands'])} Fermi-crossing bands)")
    return TaskResult(fig=fig, raw=res)


def task_fermi3d(atoms, model, out, nk_per_dim=15, energy_window=0.5):
    try:
        res = compute_fermi_surface_3d(atoms, model=model,
                                       nk_per_dim=nk_per_dim,
                                       energy_window=energy_window)
    except ImportError as e:
        print(f"  skipped 3D Fermi surface ({e})")
        return TaskResult(fig=None, error=str(e))
    fig = go.Figure()
    for mesh in res["meshes"]:
        fig.add_trace(go.Mesh3d(
            x=mesh["vertices_x"], y=mesh["vertices_y"], z=mesh["vertices_z"],
            i=mesh["faces_i"],    j=mesh["faces_j"],    k=mesh["faces_k"],
            opacity=0.5, name=f"band {mesh['band']}", showscale=False,
        ))
    fig.update_layout(title=f"3D Fermi surface ({res['formula']})",
                      width=800, height=700, template="plotly_white",
                      scene=dict(xaxis_title="kx", yaxis_title="ky",
                                 zaxis_title="kz"))
    path = f"{out}_fermi3d.html"
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  wrote {path}  ({len(res['meshes'])} isosurfaces)")
    return TaskResult(fig=fig, raw=res)


def task_ev_curve(
    atoms, model, out,
    scan=(0.9, 1.12, 9),
    kpoints=(3, 3, 3),
    use_scc=True,
    repulsive=True,
    alpha=1.0,
    scc_max_iter=60,
    scc_mixing=0.2,
    scc_tol=1e-5,
):
    factors = np.linspace(*scan)
    vols, Etot, Escc, Erep, Eelec, gaps = [], [], [], [], [], []
    for f in factors:
        a = atoms.ase_converter()
        a.set_cell(a.cell * f, scale_atoms=True)
        g = Geometry.from_ase_atoms([a])
        calc = SimpleDftb(g, model, kpoints=torch.tensor(list(kpoints)),
                          device="cpu", with_eigenvectors=False,
                          compute_forces=False, include_dos_data=False,
                          repulsive=repulsive, alpha=alpha,
                          use_scc=use_scc,
                          scc_max_iter=scc_max_iter,
                          scc_mixing=scc_mixing,
                          scc_tol=scc_tol)
        r = calc.calculate()
        vols.append(float(np.abs(np.linalg.det(a.cell))))
        Etot.append(float(r["energy"]))
        Erep.append(float(r["potential_energy"]))
        Eelec.append(float(r["electronic_energy"]))
        Escc.append(
            float(calc._scc_info["E_scc_eV"]) if use_scc else 0.0
        )
        gaps.append(float(r["bandgap"]))
    vols = np.array(vols); Etot = np.array(Etot)
    Erep = np.array(Erep); Eelec = np.array(Eelec); Escc = np.array(Escc)
    gaps = np.array(gaps); factors_arr = np.array(factors)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vols, y=Etot, mode="lines+markers",
                             name="E_tot", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=vols, y=Eelec, mode="lines+markers",
                             name="E_elec", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=vols, y=Erep, mode="lines+markers",
                             name="E_rep", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=vols, y=Escc, mode="lines+markers",
                             name="E_scc", line=dict(dash="dot")))
    imin = int(np.argmin(Etot))
    fig.add_vline(x=vols[imin], line=dict(color="gray", dash="dash"))
    fig.update_layout(
        title=f"E-V scan ({factors[0]:.2f}-{factors[-1]:.2f}) · a_ref",
        xaxis_title="Volume (A^3)", yaxis_title="Energy (eV)",
        template="plotly_white", width=800, height=500,
    )
    path = f"{out}_ev.html"
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  wrote {path}  (min @ V={vols[imin]:.2f} A^3,"
          f" E_tot={Etot[imin]:.3f} eV)")

    return TaskResult(
        fig=fig,
        factors=factors_arr,
        volumes=vols,
        E_tot=Etot, E_elec=Eelec, E_rep=Erep, E_scc=Escc,
        bandgap=gaps,
        V_min=float(vols[imin]), E_min=float(Etot[imin]),
        a_min_factor=float(factors[imin]),
    )


def task_eos(
    atoms, model, out,
    strain_range=(-0.05, 0.05),
    n_points=11,
    supercell=(1, 1, 1),
    kpoints=(3, 3, 3),
    use_scc=True,
    repulsive=True,
    alpha=1.0,
    eos_kind="murnaghan",
):
    """Jarvis-style EOS: strain_atoms(eps) loop + ase.eos fit.

    Uses jarvis's `atoms.strain_atoms(eps)` and `ase.eos.EquationOfState.fit()`
    to extract V0, E0, B0. B0 is converted to GPa via `B / kJ * 1e24`.
    """
    from ase.eos import EquationOfState
    from ase.units import kJ

    base = atoms
    if supercell != (1, 1, 1):
        base = base.make_supercell(list(supercell))

    eps_values = np.linspace(strain_range[0], strain_range[1], n_points)
    vols, energies = [], []
    for eps in eps_values:
        s = base.strain_atoms(float(eps))
        g = Geometry.from_ase_atoms([s.ase_converter()])
        calc = SimpleDftb(
            g, model, kpoints=torch.tensor(list(kpoints)),
            device="cpu", with_eigenvectors=False,
            compute_forces=False, include_dos_data=False,
            repulsive=repulsive, alpha=alpha, use_scc=use_scc,
        )
        r = calc.calculate()
        vols.append(s.volume)
        energies.append(float(r["energy"]))
    vols = np.array(vols); energies = np.array(energies)

    fit_ok = False
    try:
        eos = EquationOfState(vols.tolist(), energies.tolist(), eos=eos_kind)
        v0, e0, B = eos.fit()
        B_GPa = B / kJ * 1.0e24
        fit_ok = True
    except Exception as e:
        print(f"  EOS fit failed: {e}")
        v0 = float(vols[np.argmin(energies)])
        e0 = float(energies.min()); B_GPa = float("nan")

    # Plot data + Murnaghan fit curve
    v_fine = np.linspace(vols.min(), vols.max(), 200)
    def murn(V, V0, E0, B, Bp=4.0):
        return E0 + B*V/Bp * ((V0/V)**Bp/(Bp-1) + 1) - B*V0/(Bp-1)
    y_fine = (
        murn(v_fine, v0, e0, B_GPa * kJ / 1.0e24)
        if fit_ok else np.interp(v_fine, vols, energies)
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vols, y=energies, mode="markers",
                             marker=dict(size=8), name="slakonet"))
    fig.add_trace(go.Scatter(x=v_fine, y=y_fine, mode="lines",
                             line=dict(dash="dash"),
                             name=f"{eos_kind} fit"))
    fig.add_vline(x=v0, line=dict(color="gray", dash="dot"),
                  annotation_text=f"V0={v0:.2f} Å³",
                  annotation_position="top")
    fig.update_layout(
        title=(f"EOS fit: V0={v0:.3f} Å³, E0={e0:.3f} eV, "
               f"B0={B_GPa:.1f} GPa ({eos_kind})"),
        xaxis_title="Volume (Å³)", yaxis_title="E (eV)",
        template="plotly_white", width=800, height=500,
    )
    path = f"{out}_eos.html"
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  wrote {path}  V0={v0:.3f} Å³  E0={e0:.3f} eV  B0={B_GPa:.1f} GPa")

    return TaskResult(
        fig=fig,
        strains=eps_values,
        volumes=vols,
        energies=energies,
        V0=float(v0), E0=float(e0), B0_GPa=float(B_GPa),
        eos_kind=eos_kind, fit_ok=fit_ok,
    )


def task_spin(atoms, model, out,show=True):
    klines, xticks, xtick_labels, kpts, klabels = _high_symmetry_klines(atoms)
    g = Geometry.from_ase_atoms([atoms.ase_converter()])
    calc = SimpleDftb(g, model, klines=klines,
                      device="cpu", with_eigenvectors=True,
                      compute_forces=False, include_dos_data=False,
                      repulsive=False)
    calc.calculate()
    Natom = g.atomic_numbers.shape[-1]
    init = torch.zeros(Natom)
    for i, Z in enumerate(g.atomic_numbers.flatten().tolist()):
        if Z in (22, 23, 24, 25, 26, 27, 28, 29):
            init[i] = 2.0
    res = magnetism.compute_spin_polarized_bands(
        calc, initial_moments=init, scf=False,
    )
    eu = _to_np(res["eigenvalues_up"])
    ed = _to_np(res["eigenvalues_dn"])
    Ef = float(res["fermi_eV"])

    fig = go.Figure()
    kx = list(range(eu.shape[-1]))
    for b in range(eu.shape[0]):
        fig.add_trace(go.Scatter(x=kx, y=eu[b] - Ef, mode="lines",
                                 line=dict(color="crimson", width=1),
                                 showlegend=False))
    for b in range(ed.shape[0]):
        fig.add_trace(go.Scatter(x=kx, y=ed[b] - Ef, mode="lines",
                                 line=dict(color="royalblue", width=1,
                                           dash="dash"),
                                 showlegend=False))
    fig.add_hline(y=0.0, line=dict(color="black", dash="dot"))
    fig.update_layout(
        title=f"Spin-polarized bands (M={res['total_moment']:.2f})",
        xaxis_title="k-path", yaxis_title="E - E_F (eV)",
        template="plotly_white", width=800, height=500,
        xaxis=dict(tickmode="array", tickvals=xticks, ticktext=xtick_labels),
    )
    for x in xticks:
        fig.add_vline(x=x, line=dict(color="lightgray", width=0.5))
    path = f"{out}_spin.html"
    fig.write_html(path, include_plotlyjs="cdn")
    if show:
        fig.show()
    print(f"  wrote {path}  (total moment {res['total_moment']:.3f})")

    return TaskResult(
        fig=fig,
        eigenvalues_up=eu, eigenvalues_dn=ed,
        fermi_eV=Ef,
        total_moment=float(res["total_moment"]),
        moments=_to_np(res["moments"]),
        xticks=list(xticks), xtick_labels=list(xtick_labels),
        kpts=np.asarray(kpts), klabels=list(klabels),
    )


def task_soc(atoms, model, out):
    klines, xticks, xtick_labels, kpts, klabels = _high_symmetry_klines(atoms)
    g = Geometry.from_ase_atoms([atoms.ase_converter()])
    calc = SimpleDftb(g, model, klines=klines,
                      device="cpu", with_eigenvectors=True,
                      compute_forces=False, include_dos_data=False,
                      repulsive=False)
    calc.calculate()
    res = soc.compute_soc_bands(calc)
    e = _to_np(res["eigenvalues"])
    e_sorted = np.sort(e.flatten())
    Ef = float(e_sorted[e_sorted.shape[0] // 2])
    fig = go.Figure()
    kx = list(range(e.shape[-1]))
    for b in range(e.shape[0]):
        fig.add_trace(go.Scatter(x=kx, y=e[b] - Ef, mode="lines",
                                 line=dict(color="purple", width=0.8),
                                 showlegend=False))
    fig.add_hline(y=0.0, line=dict(color="black", dash="dot"))
    fig.update_layout(
        title="Bands with spin-orbit coupling",
        xaxis_title="k-path", yaxis_title="E - E_F (eV)",
        template="plotly_white", width=800, height=500,
        xaxis=dict(tickmode="array", tickvals=xticks, ticktext=xtick_labels),
    )
    for x in xticks:
        fig.add_vline(x=x, line=dict(color="lightgray", width=0.5))
    path = f"{out}_soc.html"
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  wrote {path}")

    return TaskResult(
        fig=fig,
        eigenvalues=e, fermi_eV=Ef,
        xticks=list(xticks), xtick_labels=list(xtick_labels),
        kpts=np.asarray(kpts), klabels=list(klabels),
    )


def task_dielectric(atoms, model, out, kgrid=(3, 3, 3),
                    omega_range_eV=(0.1, 10.0), n_omega=120,
                    smearing_eV=0.1):
    g = Geometry.from_ase_atoms([atoms.ase_converter()])
    calc = SimpleDftb(g, model, kpoints=torch.tensor([3, 3, 3]),
                      device="cpu", with_eigenvectors=False,
                      compute_forces=False, include_dos_data=False,
                      repulsive=False)
    calc.calculate()
    res = dielectric.compute_dielectric(
        calc, kgrid=kgrid, omega_range_eV=omega_range_eV,
        n_omega=n_omega, smearing_eV=smearing_eV,
    )
    w = _to_np(res["omega_eV"])
    e1 = _to_np(res["eps1_iso"])
    e2 = _to_np(res["eps2_iso"])
    eps_tensor = _to_np(res["eps2"])       # [3,3,n_omega]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w, y=e1, mode="lines", name="eps_1"))
    fig.add_trace(go.Scatter(x=w, y=e2, mode="lines", name="eps_2"))
    fig.add_hline(y=0, line=dict(color="black", dash="dot"))
    fig.update_layout(title="Dielectric function (isotropic avg.)",
                      xaxis_title="Energy (eV)", yaxis_title="epsilon(omega)",
                      template="plotly_white", width=800, height=500)
    path = f"{out}_dielectric.html"
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  wrote {path}")

    return TaskResult(
        fig=fig,
        omega_eV=w, eps1_iso=e1, eps2_iso=e2,
        eps2_tensor=eps_tensor,
        volume_bohr3=float(res["volume_bohr3"]),
        kgrid=tuple(res["kgrid"]),
    )


def task_scc_charges(atoms, model, out):
    g = Geometry.from_ase_atoms([atoms.ase_converter()])
    calc = SimpleDftb(g, model, kpoints=torch.tensor([3, 3, 3]),
                      device="cpu", with_eigenvectors=True,
                      compute_forces=False, include_dos_data=False,
                      repulsive=False, use_scc=True)
    r = calc.calculate()
    info = calc._scc_info
    dq = _to_np(info["delta_q"])
    syms = atoms.ase_converter().get_chemical_symbols()
    fig = go.Figure(go.Bar(
        x=[f"{s}{i}" for i, s in enumerate(syms)], y=dq,
    ))
    fig.update_layout(title=f"SCC Mulliken charge transfer Delta q  "
                           f"(E_scc = {float(info['E_scc_eV']):.3f} eV, "
                           f"{info['n_iter']} iters)",
                      xaxis_title="atom", yaxis_title="Delta q (e)",
                      template="plotly_white", width=700, height=400)
    path = f"{out}_scc.html"
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  wrote {path}  (Delta q = {dq.tolist()})")

    return TaskResult(
        fig=fig,
        delta_q=dq,
        symbols=list(syms),
        E_scc_eV=float(info["E_scc_eV"]),
        mu_Ha=float(info["mu_Ha"]),
        converged=bool(info["converged"]),
        n_iter=int(info["n_iter"]),
        bandgap=float(r["bandgap"]),
    )


def task_optimize(atoms, model, out, fmax=0.05, steps=20):
    ase_atoms = atoms.ase_converter()
    # fmax belongs to the optimizer (opt.run below), not the calculator.
    ase_atoms.calc = SlakoNetCalculator(model=model)
    E_before = float(ase_atoms.get_potential_energy())
    positions_before = ase_atoms.get_positions().copy()
    cell_before = np.array(ase_atoms.get_cell()).copy()

    opt = BFGS(ase_atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)

    E_after = float(ase_atoms.get_potential_energy())
    forces_after = ase_atoms.get_forces()
    max_force = float(np.abs(forces_after).max())
    print(f"  E_before = {E_before:.4f} eV   E_after = {E_after:.4f} eV   "
          f"dE = {E_after - E_before:+.4f} eV   "
          f"max|F| = {max_force:.4f} eV/A")

    return TaskResult(
        fig=None,
        E_before=E_before, E_after=E_after, dE=E_after - E_before,
        max_force=max_force,
        positions_before=positions_before,
        positions_after=ase_atoms.get_positions(),
        cell_before=cell_before,
        cell_after=np.array(ase_atoms.get_cell()),
        forces_after=forces_after,
        n_steps=int(opt.nsteps),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

TASKS = {
    "bands":      task_bands_dos,
    "fermi2d":    task_fermi2d,
    "fermi3d":    task_fermi3d,
    "ev":         task_ev_curve,
    "eos":        task_eos,
    "spin":       task_spin,
    "soc":        task_soc,
    "dielectric": task_dielectric,
    "scc":        task_scc_charges,
    "optimize":   task_optimize,
}


def run_all(jid="JVASP-1002", skip=None, show=False, out=None):
    """Drive every task in TASKS. Works in scripts and notebooks.

    Returns
    -------
    dict {task_name: TaskResult | None}  - each TaskResult is a dict with a
    'fig' key (plotly figure or None) and other numerical result keys.
    """
    skip = set(skip or [])
    atoms, opt_gap, _mbj = get_atoms(jid)
    print(f"JID={jid}  formula={atoms.composition.reduced_formula}  "
          f"natoms={atoms.num_atoms}  DFT gap={opt_gap}")
    model = default_model()
    prefix = out or jid
    results = {}
    for name, fn in TASKS.items():
        if name in skip:
            results[name] = None; continue
        print(f"\n[{name}]")
        try:
            r = fn(atoms, model, prefix)
            results[name] = r
            if show and isinstance(r, dict) and r.get("fig") is not None:
                r["fig"].show()
        except Exception as e:
            results[name] = None
            print(f"  ERROR in {name}: {type(e).__name__}: {e}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jid", default="JVASP-1002")
    ap.add_argument("--skip", default="",
                    help="comma-separated task names")
    args = ap.parse_args()
    run_all(args.jid, skip={t for t in args.skip.split(",") if t})


if __name__ == "__main__":
    main()
