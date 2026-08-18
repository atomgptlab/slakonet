"""ASE calculator for a pre-loaded SlaKoNet model.

One model load, reused for every structure and every call. Standard
ASE properties (energy / forces / stress) plus first-class
``band_structure()`` and ``dos()`` accessors.

The model is *injected*, never loaded here -- load it once with
``slakonet.predict_slakonet.load_trained_model`` and hand it in.

Example
-------
    from slakonet.predict_slakonet import load_trained_model
    from slakonet.ase_calc import SlaKoNetCalculator
    from ase.build import bulk

    model = load_trained_model(MODEL_PT, prefer="pt").float()   # once
    calc  = SlaKoNetCalculator(model, kpoints=(3, 3, 3))

    si = bulk("Si", "diamond", a=5.43); si.calc = calc
    si.get_potential_energy()           # eV
    si.get_forces()                     # eV/Ang  (N,3)
    si.get_stress()                     # eV/Ang^3 Voigt(6) -- see note
    bs = calc.band_structure(si, path="GXWKGL", npoints=120)
    e, d = calc.dos(si)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from pydantic import BaseModel, field_validator

from slakonet.atoms import Geometry
from slakonet.main import (
    SimpleDftb,
    _check_calculator_kwargs,
    generate_shell_dict_upto_Z65,
)
from slakonet.optim import kpts_to_klines

# eV/Ang^3  <->  GPa  (slakonet returns stress in GPa, Voigt order)
_GPA_TO_EV_A3 = 1.0 / 160.21766208


class SlaKoNetConfig(BaseModel):
    """Declarative knobs / output toggles for ``SlaKoNetCalculator``.

    The trained model is injected separately (it is a loaded torch
    object, not config-serializable); everything else lives here so a
    run can be fully described by a JSON file.
    """

    kpoints: List[int] = [3, 3, 3]  # Monkhorst-Pack grid
    cutoff: float = 10.0  # Bohr
    kT: float = 0.025  # Fermi smearing (eV)
    # alpha scales the band-structure energy and beta the forces. Both are
    # 1.0 for the standard DFTB total energy E = E_band + E_rep and its
    # exact gradient; changing them breaks energy/force consistency.
    alpha: float = 1.0
    beta: float = 1.0
    use_scc: bool = False
    compute_forces: bool = True
    compute_stress: bool = True  # needs compute_forces + periodic
    include_dos: bool = False
    device: Optional[str] = None

    model_config = {"extra": "forbid"}

    @field_validator("kpoints")
    @classmethod
    def _k3(cls, v):
        v = list(v)
        if len(v) != 3:
            raise ValueError("kpoints must have 3 integers")
        return [int(x) for x in v]

    @classmethod
    def from_file(cls, path: str) -> "SlaKoNetConfig":
        """Load from a JSON file."""
        import json

        with open(path) as fh:
            return cls(**json.load(fh))


class SlaKoNetCalculator(Calculator):
    """ASE Calculator wrapping an already-loaded SlaKoNet model.

    Args:
        model: loaded SlaKoNet model (e.g. from ``load_trained_model``).
            Stored and reused -- never reloaded.
        config: ``SlaKoNetConfig`` | dict | JSON path | None. Declares
            the knobs/toggles; explicit keywords below override it.
        shell_dict: optional; derived from ``model`` if omitted.
        kpoints: Monkhorst-Pack grid for the energy/force/stress solve.
        cutoff: interaction cutoff (Bohr), matches slakonet default.
        kT, alpha, beta: Fermi smearing / mixing / force scaling knobs
            (passed straight through to ``SimpleDftb``).
        use_scc: self-consistent charges (slower).
        compute_forces: toggle force evaluation (autograd). Off => fast
            energy-only path (``get_forces`` then unavailable).
        compute_stress: toggle stress (requires ``compute_forces`` and a
            periodic cell).
        include_dos: also compute DOS during ``calculate`` (else use the
            on-demand ``dos()`` method).
        device: 'cuda' / 'cpu' (auto if None).
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model,
        config=None,
        shell_dict=None,
        *,
        kpoints=None,
        kspacing: Optional[float] = None,
        cutoff: Optional[float] = None,
        kT: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        use_scc: Optional[bool] = None,
        compute_forces: Optional[bool] = None,
        compute_stress: Optional[bool] = None,
        include_dos: Optional[bool] = None,
        device: Optional[str] = None,
        **kw,
    ):
        """``config`` may be a ``SlaKoNetConfig``, a dict, a JSON path,
        or None. Any explicit keyword (kpoints=, beta=, ...) overrides
        the corresponding config field, so existing call sites that pass
        plain kwargs keep working unchanged."""
        _check_calculator_kwargs(type(self), kw)
        Calculator.__init__(self, **kw)
        self.model = model  # loaded ONCE; reused across all calls
        self.shell_dict = shell_dict or generate_shell_dict_upto_Z65(
            model=model
        )

        if config is None:
            cfg = SlaKoNetConfig()
        elif isinstance(config, SlaKoNetConfig):
            cfg = config
        elif isinstance(config, str):
            cfg = SlaKoNetConfig.from_file(config)
        elif isinstance(config, dict):
            cfg = SlaKoNetConfig(**config)
        else:
            raise TypeError(
                "config must be SlaKoNetConfig | dict | JSON path | None"
            )

        overrides = {
            k: v
            for k, v in dict(
                kpoints=kpoints,
                cutoff=cutoff,
                kT=kT,
                alpha=alpha,
                beta=beta,
                use_scc=use_scc,
                compute_forces=compute_forces,
                compute_stress=compute_stress,
                include_dos=include_dos,
                device=device,
            ).items()
            if v is not None
        }
        if overrides:
            cfg = cfg.model_copy(update=overrides)
        self.cfg = cfg

        self.kpoints = tuple(cfg.kpoints)
        # Density-based mesh, as in SlakoNetCalculator.kpoints_for. Taken
        # straight from **kw before, which meant ASE's base Calculator
        # silently absorbed it and every structure kept the fixed mesh --
        # a 3x3x3 grid puts a spurious 1.5 eV "gap" on fcc Al.
        self.kspacing = kspacing
        self.cutoff = cfg.cutoff
        self.kT = cfg.kT
        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.use_scc = cfg.use_scc
        self.compute_forces = cfg.compute_forces
        self.compute_stress = cfg.compute_stress
        self.include_dos = cfg.include_dos
        self.device = cfg.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ---- ASE entry point -------------------------------------------------

    def kpoints_for(self, atoms):
        """Monkhorst-Pack divisions for `atoms`.

        With ``kspacing`` set the mesh follows the reciprocal cell so
        every structure is sampled to the same density,
        ``n_i = ceil(|b_i| / kspacing)``; non-periodic directions get one
        k-point. Without it the fixed ``kpoints`` is returned unchanged.
        """
        if self.kspacing is None:
            return list(self.kpoints)
        cell = np.asarray(atoms.get_cell())
        if abs(np.linalg.det(cell)) < 1e-8:
            return list(self.kpoints)
        recip = 2.0 * np.pi * np.linalg.inv(cell).T
        pbc = np.asarray(atoms.get_pbc())
        mesh = []
        for i in range(3):
            if not pbc[i]:
                mesh.append(1)
                continue
            mesh.append(
                max(
                    1,
                    int(
                        np.ceil(
                            np.linalg.norm(recip[i]) / self.kspacing - 1e-8
                        )
                    ),
                )
            )
        return mesh

    def calculate(
        self, atoms=None, properties=("energy",), system_changes=all_changes
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        geo = Geometry.from_ase_atoms([self.atoms])
        sim = SimpleDftb(
            geo,
            self.model,
            kpoints=torch.tensor(self.kpoints_for(self.atoms)),
            device=self.device,
            with_eigenvectors=False,
            compute_forces=self.compute_forces,
            include_dos_data=self.include_dos,
            repulsive=True,
            alpha=self.alpha,
            beta=self.beta,
            kT=self.kT,
            use_scc=self.use_scc,
        )
        r = sim.calculate()

        e = float(r["energy"].detach().cpu().item())
        self.results["energy"] = e
        self.results["free_energy"] = e

        if self.compute_forces and r.get("forces") is not None:
            self.results["forces"] = (
                r["forces"].detach().cpu().numpy().reshape(-1, 3)
            )

        if self.compute_stress and r.get("stress") is not None:
            # slakonet -> Voigt(6) in GPa; ASE wants eV/Ang^3.
            st = r["stress"].detach().cpu().numpy().reshape(-1)[:6]
            self.results["stress"] = st * _GPA_TO_EV_A3

        # convenience extras (not ASE-standard, kept on the calculator)
        for key in ("bandgap", "fermi_energy", "vbm", "cbm"):
            v = r.get(key)
            if v is not None:
                self.results[key] = (
                    float(v.detach().cpu().item())
                    if torch.is_tensor(v)
                    else float(v)
                )
        if self.include_dos and "dos_energy_grid_tensor" in r:
            self.results["dos"] = (
                r["dos_energy_grid_tensor"].detach().cpu().numpy(),
                r["dos_values_tensor"].detach().cpu().numpy(),
            )
        self._last_result = r

    # ---- band structure --------------------------------------------------
    def band_structure(
        self,
        atoms=None,
        path: Optional[str] = None,
        npoints: int = 80,
        savefig: Optional[str] = None,
        emin: float = -6.0,
        emax: float = 8.0,
        mask_ev: float = 30.0,
    ):
        """Band structure along an ASE k-path (non-SCC).

        Returns a dict: ``kpts`` (frac), ``energies`` (nk, nband, eV,
        referenced to mid-gap), ``labels``, ``path``, ``gap``, ``vbm``,
        ``cbm``. Writes a PNG if ``savefig`` is given.
        """
        atoms = atoms if atoms is not None else self.atoms
        ase_atoms = atoms
        bp = (
            ase_atoms.cell.bandpath(npoints=npoints)
            if path is None
            else ase_atoms.cell.bandpath(path=path, npoints=npoints)
        )
        kpts_frac = bp.kpts
        labels = [""] * len(kpts_frac)
        for name, pt in bp.special_points.items():
            i = int(
                np.argmin(np.linalg.norm(kpts_frac - np.asarray(pt), axis=1))
            )
            labels[i] = (labels[i] + "|" + name) if labels[i] else name

        klines = kpts_to_klines(kpts_frac.tolist(), default_points=2)
        geo = Geometry.from_ase_atoms([ase_atoms])
        with torch.no_grad():
            props, ok = self.model.compute_multi_element_properties(
                geometry=geo,
                shell_dict=self.shell_dict,
                klines=klines,
                get_fermi=True,
                with_eigenvectors=bool(savefig),
                device=self.device,
                cutoff=self.cutoff,
                use_scc=self.use_scc,
                kT=self.kT,
                alpha=self.alpha,
            )
        assert ok, "compute_multi_element_properties failed"

        eigenvalues = props["eigenvalues"].detach().cpu().numpy()
        ev = eigenvalues[0].copy()
        ev[np.abs(ev) > mask_ev] = np.nan

        flat = ev[~np.isnan(ev)].ravel()
        flat.sort()
        d = np.diff(flat)
        lo, hi = flat[:-1], flat[1:]
        near = (lo <= 0.5) & (hi >= -0.5)
        j = int(np.argmax(d * near)) if near.any() else int(np.argmax(d))
        vbm, cbm = float(lo[j]), float(hi[j])
        gap = max(cbm - vbm, 0.0)
        mid = 0.5 * (vbm + cbm)

        out = {
            "kpts": np.asarray(kpts_frac),
            "energies": ev - mid,
            "labels": labels,
            "path": bp.path,
            "gap": gap,
            "vbm": vbm,
            "cbm": cbm,
            "properties": props,
        }

        if savefig:
            self._plot_bands(
                eigenvalues,
                labels,
                mid,
                gap,
                atoms,
                emin,
                emax,
                mask_ev,
                savefig,
            )
        return out

    @staticmethod
    def _plot_bands(
        eigenvalues, labels, mid, gap, atoms, emin, emax, mask_ev, savefig
    ):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from slakonet.predict_slakonet import (
            _split_path_discontinuities,
            _format_kpath_ticks,
        )

        eigs, plabels = _split_path_discontinuities(eigenvalues, labels)
        eigs = eigs[0] - mid
        xticks, xlab = _format_kpath_ticks(plabels)
        ep = eigs.astype(float).copy()
        ep[np.abs(ep) > mask_ev] = np.nan
        ep = np.sort(ep, axis=-1)
        if ep.shape[0] > 1:
            big = np.abs(np.diff(ep, axis=0)) > 2.0
            nm = np.concatenate([np.zeros_like(big[:1]), big], 0).astype(bool)
            ep[nm] = np.nan

        fig, ax = plt.subplots(figsize=(8, 5))
        for ib in range(ep.shape[-1]):
            ax.plot(ep[:, ib], lw=0.8)
        ax.axhline(0.0, color="k", ls="--", lw=0.6, alpha=0.6)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlab)
        ax.set_ylabel(r"E - E$_F$  (eV)")
        ax.set_xlim(0, ep.shape[0] - 1)
        ax.set_ylim(emin, emax)
        ax.set_title(
            f"{atoms.get_chemical_formula()} - "
            f"E$_g$={gap:.3f} eV  (slakonet)"
        )
        for x in xticks:
            ax.axvline(x, color="gray", lw=0.4, alpha=0.5)
        fig.tight_layout()
        plt.savefig(savefig, dpi=200)
        plt.close(fig)

    # ---- DOS -------------------------------------------------------------
    def dos(
        self,
        atoms=None,
        energy_range=(-10.0, 10.0),
        num_points: int = 3000,
        sigma: float = 0.1,
    ):
        """Total DOS (Fermi-referenced). Returns (energies_eV, dos)."""
        atoms = atoms if atoms is not None else self.atoms
        geo = Geometry.from_ase_atoms([atoms])
        sim = SimpleDftb(
            geo,
            self.model,
            kpoints=torch.tensor(self.kpoints_for(self.atoms)),
            device=self.device,
            with_eigenvectors=False,
            compute_forces=False,
            include_dos_data=False,
            repulsive=True,
            alpha=self.alpha,
            beta=self.beta,
            kT=self.kT,
            use_scc=self.use_scc,
        )
        sim.calculate()
        e_grid, dos = sim.calculate_dos(
            energy_range=energy_range,
            num_points=num_points,
            sigma=sigma,
            fermi_shift=True,
        )
        return (
            e_grid.detach().cpu().numpy(),
            dos.detach().cpu().numpy(),
        )

    def get_HS(self, atoms=None, kpoints=None):
        """k-resolved Hamiltonian and overlap matrices.

        Returns ``(H, S)`` as numpy arrays of shape
        ``(n_kpoints, n_orbitals, n_orbitals)``. H is in **Hartree** (the
        SKF native unit) and the basis is non-orthogonal, so band energies
        come from the generalized problem ``H c = e S c``:

            w = scipy.linalg.eigh(H[k], S[k], eigvals_only=True)
            eigenvalues_eV = w * 27.211 - calc.get_fermi_level()

        The matrices are complex in general and Hermitian at every k.

        `kpoints` overrides the calculator's Monkhorst-Pack mesh for this
        call only, e.g. ``get_HS(kpoints=(1, 1, 1))`` for Gamma only.
        """
        atoms = atoms if atoms is not None else self.atoms
        mesh = list(kpoints) if kpoints is not None else list(self.kpoints)
        geo = Geometry.from_ase_atoms([atoms])
        sim = SimpleDftb(
            geo,
            self.model,
            kpoints=torch.tensor(mesh),
            device=self.device,
            with_eigenvectors=False,
            compute_forces=False,
            include_dos_data=False,
            include_HS=True,
            repulsive=True,
            alpha=self.alpha,
            beta=self.beta,
            kT=self.kT,
            use_scc=self.use_scc,
        )
        sim.calculate()
        # SimpleDftb stores these as (batch, n_orb, n_orb, n_k); move the
        # k axis to the front so H[k] is a matrix.
        H = sim._results["hamiltonian"][0].permute(2, 0, 1)
        S = sim._results["overlap"][0].permute(2, 0, 1)
        return (
            H.detach().cpu().numpy(),
            S.detach().cpu().numpy(),
        )

    # ---- convenience -----------------------------------------------------
    def get_bandstructure(self, atoms=None, **kwargs):
        """Alias for :meth:`band_structure`."""
        return self.band_structure(atoms=atoms, **kwargs)

    def get_dos(self, atoms=None, **kwargs):
        """Alias for :meth:`dos`."""
        return self.dos(atoms=atoms, **kwargs)

    def get_bandgap(self, atoms=None):
        if "bandgap" not in self.results:
            self.get_potential_energy(atoms)
        return self.results.get("bandgap")

    def get_fermi_level(self, atoms=None):
        if "fermi_energy" not in self.results:
            self.get_potential_energy(atoms)
        return self.results.get("fermi_energy")
