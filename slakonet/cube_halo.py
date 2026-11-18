"""Large-system SlaKoNet energies and forces (Gamma point, non-SCC).

The dense ``SimpleDftb`` path builds several Norb x Norb index, mask and
distance tensors before it diagonalises; for 2,744 Si atoms that is
~107 GB although H itself is 2.4 GB. This module avoids them by building
H and S with ``sparse_sk.hs_matrix_sparse`` (neighbour list, O(N)) and
offers two solvers:

``SparseSlakoNet.gamma_exact``
    Whole periodic cell at k = Gamma. Sparse assembly, then one dense
    generalised eigensolve. Same energy as the dense path at Gamma;
    memory ~6 Norb^2 floats.

``SparseSlakoNet.cube_halo``
    Divide and conquer (the cube-and-halo idea used for ALIGNN-FF). The
    cell is split into boxes; each box plus a halo of width ``halo`` is
    cut out as a finite cluster and diagonalised; only the box ("core")
    share of the band energy and electrons is kept, with one Fermi level
    for the whole system. Memory is set by one cluster and time grows
    linearly with N. Unlike ALIGNN-FF this is an approximation: the
    error decays (exponentially for gapped systems) with the halo width,
    so converge ``halo`` against ``gamma_exact`` on a smaller cell.

Core partition: with D the core-atom orbital mask and P_ij = (D_i + D_j)
/ 2, state n contributes q_n = sum_{i in core} c_in (S c_n)_i electrons
and q_n * eps_n band energy.

Forces (both solvers): F = -sum_ij [rho_ij dH_ij/dR - W_ij dS_ij/dR] with
rho = sum_n f_n c_n c_n^T and W = sum_n f_n eps_n c_n c_n^T (times P for
cube_halo), evaluated by autograd through the sparse assembly with rho,
W held fixed. The pair-repulsive energy is local and computed exactly
over the whole cell.

Example::

    from ase.build import bulk
    from slakonet.optim import default_model
    from slakonet.cube_halo import SparseSlakoNet

    si = bulk("Si", "diamond", a=5.43, cubic=True).repeat(12)
    calc = SparseSlakoNet(default_model(), elements=["Si"])
    r = calc.cube_halo(si, ncubes=(4, 4, 4), halo=8.0)
    r["energy"], r["forces"]
"""

from __future__ import annotations

import copy
import itertools
import time
from typing import Optional, Sequence

import numpy as np
import torch
from ase.neighborlist import neighbor_list

from slakonet.atoms import Geometry
from slakonet.basis import Basis
from slakonet.sparse_sk import hs_matrix_sparse
from slakonet.utils import create_feeds, generate_shell_dict_upto_Z65

BOHR = 0.5291772109  # Angstrom per Bohr
H2E = 27.211  # Hartree -> eV, as in SimpleDftb


def _t(v, device):
    v = v.detach().to(device)
    return v.double() if v.is_floating_point() else v


def _move(obj, device, depth=0):
    """Move tensor attributes of an SKF-like object to ``device`` (float64)."""
    if depth > 4 or not hasattr(obj, "__dict__"):
        return obj
    for k, v in vars(obj).items():
        if torch.is_tensor(v):
            setattr(obj, k, _t(v, device))
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if torch.is_tensor(vv):
                    v[kk] = _t(vv, device)
        elif hasattr(v, "__dict__") and not isinstance(v, type):
            _move(v, device, depth + 1)
    return obj


def _fermi_dirac(e, mu, kT):
    """Occupation (0..1) per spin; numpy, overflow-safe."""
    from scipy.special import expit

    return expit(-(e - mu) / kT)


class SparseSlakoNet:
    """SlaKoNet energies/forces for large cells without dense Norb^2 setup.

    Args:
        model: loaded SlaKoNet model (``slakonet.optim.default_model()``).
        elements: chemical symbols present (feeds are built only for these).
        device: torch device for assembly and eigensolves.
        cutoff: H/S interaction cutoff in Bohr (dense default 10).
        eig_dtype: dtype of the dense eigensolve (float64 default; float32
            halves memory and is faster on GPUs with weak FP64, check the
            accuracy for your system).
    """

    def __init__(
        self,
        model,
        elements: Sequence[str],
        device: str = "cuda",
        cutoff: float = 10.0,
        eig_dtype=torch.float64,
    ):
        from jarvis.core.specie import Specie

        self.device = torch.device(device)
        self.cutoff = float(cutoff)
        self.eig_dtype = eig_dtype
        syms = set(elements)
        skfs = {
            k: v
            for k, v in model.get_updated_skfs().items()
            if set(str(k).split("-")) <= syms
        }
        if not skfs:
            raise ValueError(f"model has no parameters for {sorted(syms)}")
        # spline tables follow the device of the SKF tensors: copy the
        # (small) filtered set onto the compute device
        skfs = {
            k: _move(copy.deepcopy(v), self.device) for k, v in skfs.items()
        }
        self.shell_dict = generate_shell_dict_upto_Z65(model=model)
        self.h_feed = create_feeds(skfs, self.shell_dict, "H")
        self.s_feed = create_feeds(skfs, self.shell_dict, "S")

        # valence electrons per element and orbitals per atom
        self.nelec = {}
        for key, skf in skfs.items():
            a, b = str(key).split("-")
            if a != b:
                continue
            ad = skf.to_dict().get("atomic_data") or {}
            occ = ad.get("occupations")
            if occ is not None:
                self.nelec[Specie(a).Z] = float(torch.as_tensor(occ).sum())
        self.norb = {
            z: sum(2 * int(ang) + 1 for ang in ls)
            for z, ls in self.shell_dict.items()
        }

        # pair-repulsive splines (Bohr in, eV out), as SimpleDftb
        self.rep = {}
        self.rep_cut = 0.0
        for key, skf in skfs.items():
            rs = getattr(skf, "r_spline", None)
            if not rs:
                continue
            a, b = str(key).split("-")
            if str(key) != "-".join(sorted([a, b])):
                continue  # SimpleDftb uses the sorted-name pair only
            cut = float(torch.as_tensor(rs.cutoff).flatten()[0])
            kw = dict(device=self.device, dtype=torch.float64)
            self.rep[(Specie(a).Z, Specie(b).Z)] = dict(
                grid=torch.as_tensor(rs.grid).to(**kw),
                cutoff=cut,
                exp=torch.as_tensor(rs.exp_coef).to(**kw) * H2E,
                spl=torch.as_tensor(rs.spline_coef).to(**kw) * H2E,
                tail=torch.as_tensor(rs.tail_coef).to(**kw) * H2E,
            )
            self.rep_cut = max(self.rep_cut, cut)

    # ------------------------------------------------------------------
    # building blocks
    # ------------------------------------------------------------------
    def _coo(self, pos, z, cell=None, kpoint=None):
        """Sparse H and S of one system as COO triplets (Hartree)."""
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)  # SK rotation buffers
        try:
            return self._coo64(pos, z, cell, kpoint)
        finally:
            torch.set_default_dtype(old)

    def _coo64(self, pos, z, cell, kpoint):
        geo = Geometry(
            z.unsqueeze(0),
            pos.unsqueeze(0),
            None if cell is None else cell.unsqueeze(0),
            units="bohr",
        )
        basis = Basis(geo.atomic_numbers, self.shell_dict)
        out = []
        for feed in (self.h_feed, self.s_feed):
            r, c, v, n = hs_matrix_sparse(
                geo,
                basis,
                feed,
                cutoff=self.cutoff,
                kpoint=kpoint,
                return_coo=True,
            )
            if v.is_complex():  # phases are +-1 when 2k is integer
                if v.imag.abs().max() > 1e-10:
                    raise ValueError("kpoint must have 2k integer")
                v = v.real
            out.append((r, c, v.to(torch.float64)))
        return out[0], out[1], n

    def _dense(self, trip, n):
        r, c, v = trip
        m = torch.zeros(n, n, dtype=self.eig_dtype, device=self.device)
        m.index_put_((r, c), v.detach().to(self.eig_dtype), accumulate=True)
        return m

    def _geneig(self, th, ts, n):
        """Solve H c = e S c. Returns e (Ha), C, S C."""
        S = self._dense(ts, n)
        L = torch.linalg.cholesky(S)
        del S
        A = torch.linalg.solve_triangular(L, self._dense(th, n), upper=False)
        A = torch.linalg.solve_triangular(L, A.mT, upper=False)
        e, V = torch.linalg.eigh(A)
        del A
        C = torch.linalg.solve_triangular(L.mT, V, upper=True)
        SC = L @ V
        return e, C, SC

    def _orb_atom(self, z):
        counts = torch.tensor(
            [self.norb[int(a)] for a in z.tolist()], device=self.device
        )
        return torch.repeat_interleave(
            torch.arange(len(z), device=self.device), counts
        )

    def _repulsive(self, atoms, pos, cell):
        """Exact pair-repulsive energy (eV) of the whole periodic cell."""
        if not self.rep:
            return pos.sum() * 0.0
        i, j, S = neighbor_list("ijS", atoms, self.rep_cut * BOHR)
        i = torch.as_tensor(i, device=self.device)
        j = torch.as_tensor(j, device=self.device)
        S = torch.as_tensor(S, dtype=torch.float64, device=self.device)
        d = (pos[j] - pos[i] + S @ cell).norm(dim=1)
        zn = torch.as_tensor(atoms.numbers, device=self.device)
        zi, zj = zn[i], zn[j]
        e = torch.zeros_like(d)
        for (za, zb), p in self.rep.items():
            m = ((zi == za) & (zj == zb)) | ((zi == zb) & (zj == za))
            g = p["grid"]
            head = m & (d < g[0])
            spl = m & (d >= g[0]) & (d < g[-1])
            tail = m & (d >= g[-1]) & (d < p["cutoff"])
            if head.any():
                x = d[head]
                e[head] = (
                    torch.exp(-p["exp"][0] * x + p["exp"][1]) + p["exp"][2]
                )
            if spl.any():
                x = d[spl]
                k = torch.searchsorted(g, x, right=True) - 1
                k = k.clamp(0, p["spl"].shape[0] - 1)
                c = p["spl"][k]
                dx = x - g[k]
                e[spl] = (
                    c[:, 0] + c[:, 1] * dx + c[:, 2] * dx**2 + c[:, 3] * dx**3
                )
            if tail.any():
                dx = d[tail] - g[-1]
                acc = torch.zeros_like(dx)
                for cf in p["tail"].flip(0):
                    acc = acc * dx + cf
                e[tail] = acc
        return 0.5 * e.sum()

    def _setup(self, atoms, forces):
        z = torch.as_tensor(atoms.numbers, device=self.device)
        pos = torch.tensor(
            atoms.positions / BOHR,
            dtype=torch.float64,
            device=self.device,
            requires_grad=forces,
        )
        cell = torch.tensor(
            np.array(atoms.cell) / BOHR,
            dtype=torch.float64,
            device=self.device,
        )
        n_el = sum(self.nelec[int(a)] for a in atoms.numbers)
        return z, pos, cell, n_el

    @staticmethod
    def _hf_surrogate(th, ts, rho, W, weight=None):
        """sum rho_ij H_ij - W_ij S_ij over COO entries (Hartree)."""
        rh, ch, vh = th
        rs, cs, vs = ts
        a = rho[rh, ch].to(torch.float64)
        b = W[rs, cs].to(torch.float64)
        if weight is not None:
            a = a * 0.5 * (weight[rh] + weight[ch])
            b = b * 0.5 * (weight[rs] + weight[cs])
        return (a * vh).sum() - (b * vs).sum()

    # ------------------------------------------------------------------
    # whole cell, exact at Gamma
    # ------------------------------------------------------------------
    def gamma_exact(self, atoms, forces: bool = True, kpoint=None):
        """Energy (eV) and forces (eV/A) of a periodic cell at one k-point.

        Integer filling (kT = 0) as in the dense SimpleDftb path. The
        default k is Gamma. ``kpoint`` (fractional) may be any point with
        2k integer, where H(k) is real; note that SimpleDftb with
        ``kpoints=(1, 1, 1)`` samples k = (1/2, 1/2, 1/2), not Gamma.
        """
        t0 = time.perf_counter()
        z, pos, cell, n_el = self._setup(atoms, forces)
        with torch.set_grad_enabled(forces):
            th, ts, n = self._coo(pos, z, cell, kpoint)
        e, C, SC = self._geneig(th, ts, n)
        del SC
        nocc = int(round(n_el / 2))
        e_band = 2.0 * e[:nocc].sum().item() * H2E
        out = dict(n_orb=n, nocc=nocc, gap_eV=None)
        if nocc < n:
            out["gap_eV"] = float(e[nocc] - e[nocc - 1]) * H2E
        with torch.set_grad_enabled(forces):
            e_rep = self._repulsive(atoms, pos, cell)
        F = None
        if forces:
            Co = C[:, :nocc]
            rho = 2.0 * Co @ Co.mT
            W = 2.0 * (Co * e[:nocc]) @ Co.mT
            del C, Co
            L = self._hf_surrogate(th, ts, rho, W) * H2E + e_rep
            del rho, W
            (g,) = torch.autograd.grad(L, pos)
            F = (-g / BOHR).detach().cpu().numpy()
        out.update(
            energy=e_band + float(e_rep),
            e_band=e_band,
            e_rep=float(e_rep),
            forces=F,
            seconds=time.perf_counter() - t0,
        )
        return out

    # ------------------------------------------------------------------
    # cube and halo
    # ------------------------------------------------------------------
    def pieces(self, atoms, ncubes, halo):
        """Split a periodic cell into (atom indices, core mask, xyz) pieces.

        Boxes are an ncubes grid in fractional coordinates; the halo is
        ``halo`` Angstrom measured perpendicular to the cell faces.
        """
        cell = np.array(atoms.cell)
        n = np.asarray(ncubes, dtype=int)
        vol = abs(np.linalg.det(cell))
        heights = vol / np.linalg.norm(
            np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1
        )
        hfrac = halo / heights
        if np.any(1.0 / n + 2.0 * hfrac >= 1.0):
            raise ValueError(
                "box + 2*halo must be smaller than the cell along every "
                "axis; use more cubes, a smaller halo, a larger cell, or "
                "gamma_exact()"
            )
        frac = atoms.get_scaled_positions(wrap=True)
        box = np.floor(frac * n).astype(int) % n
        out = []
        for b in itertools.product(*[range(k) for k in n]):
            centre = (np.asarray(b) + 0.5) / n
            d = frac - centre
            d -= np.round(d)
            sel = np.nonzero(np.all(np.abs(d) < 0.5 / n + hfrac, axis=1))[0]
            core = np.all(box[sel] == np.asarray(b), axis=1)
            out.append((sel, core, (centre + d[sel]) @ cell))
        return out

    def _piece_solve(self, z_all, sel, core, xyz, grad):
        z = z_all[sel]
        pos = torch.tensor(
            xyz / BOHR,
            dtype=torch.float64,
            device=self.device,
            requires_grad=grad,
        )
        with torch.set_grad_enabled(grad):
            th, ts, n = self._coo(pos, z)
        e, C, SC = self._geneig(th, ts, n)
        cmask = torch.as_tensor(core, device=self.device)
        w_orb = cmask[self._orb_atom(z)].to(C.dtype)
        q = ((C * SC) * w_orb[:, None]).sum(0)
        del SC
        return pos, th, ts, e, C, q, w_orb

    def cube_halo(
        self,
        atoms,
        ncubes=(2, 2, 2),
        halo: float = 8.0,
        kT: float = 0.025,
        forces: bool = True,
        mu: Optional[float] = None,
        verbose: bool = False,
    ):
        """Energy (eV) and forces (eV/A) by cube-and-halo.

        Args:
            atoms: periodic ``ase.Atoms``.
            ncubes: boxes along each cell vector.
            halo: halo width in Angstrom (converge it).
            kT: Fermi smearing in eV for the global Fermi level.
            forces: compute forces.
            mu: Fermi level in eV. If given (e.g. from the previous MD
                step), the global Fermi search is skipped and each piece
                is solved once; otherwise energies need one pass and
                forces a second pass with the converged mu.
        """
        t0 = time.perf_counter()
        z_all = torch.as_tensor(atoms.numbers, device=self.device)
        n_el = sum(self.nelec[int(a)] for a in atoms.numbers)
        parts = self.pieces(atoms, ncubes, halo)
        kT_ha = kT / H2E
        sizes = [len(p[0]) for p in parts]
        if verbose:
            print(
                f"[cube_halo] {len(parts)} pieces, atoms per piece "
                f"{min(sizes)}-{max(sizes)} (cell {len(atoms)}), "
                f"duplication x{sum(sizes) / len(atoms):.2f}",
                flush=True,
            )
        F = np.zeros((len(atoms), 3))
        e_band = 0.0
        one_pass = mu is not None

        if not one_pass:
            # pass 1: eigenvalues and core weights -> global Fermi level
            es, qs = [], []
            for k, (sel, core, xyz) in enumerate(parts):
                _, _, _, e, C, q, _ = self._piece_solve(
                    z_all, sel, core, xyz, grad=False
                )
                es.append(e.double().cpu().numpy())
                qs.append(q.double().cpu().numpy())
                del C
                if verbose:
                    print(
                        f"  pass1 piece {k + 1}/{len(parts)} "
                        f"{time.perf_counter() - t0:.0f}s",
                        flush=True,
                    )
            es, qs = np.concatenate(es), np.concatenate(qs)

            def count(m):
                return 2.0 * (_fermi_dirac(es, m, kT_ha) * qs).sum()

            lo, hi = es.min() - 1.0, es.max() + 1.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if count(mid) < n_el:
                    lo = mid
                else:
                    hi = mid
            mu_ha = 0.5 * (lo + hi)
            occ = 2.0 * _fermi_dirac(es, mu_ha, kT_ha)
            e_band = float((occ * es * qs).sum()) * H2E
            n_count = float((occ * qs).sum())
        else:
            mu_ha = mu / H2E
            n_count = 0.0

        if forces or one_pass:
            for k, (sel, core, xyz) in enumerate(parts):
                pos, th, ts, e, C, q, w_orb = self._piece_solve(
                    z_all, sel, core, xyz, grad=forces
                )
                ed = e.double().cpu().numpy()
                f = 2.0 * _fermi_dirac(ed, mu_ha, kT_ha)
                if one_pass:
                    qd = q.double().cpu().numpy()
                    e_band += float((f * ed * qd).sum()) * H2E
                    n_count += float((f * qd).sum())
                if forces:
                    keep = np.nonzero(f > 1e-12)[0]
                    kk = torch.as_tensor(keep, device=self.device)
                    Co = C[:, kk]
                    fo = torch.as_tensor(f[keep], device=self.device).to(
                        C.dtype
                    )
                    rho = (Co * fo) @ Co.mT
                    W = (Co * (fo * e[kk])) @ Co.mT
                    del C, Co
                    L = self._hf_surrogate(th, ts, rho, W, w_orb) * H2E
                    del rho, W
                    (g,) = torch.autograd.grad(L, pos)
                    F[sel] -= g.detach().cpu().numpy() / BOHR
                if verbose:
                    print(
                        f"  pass2 piece {k + 1}/{len(parts)} "
                        f"{time.perf_counter() - t0:.0f}s",
                        flush=True,
                    )

        z, pos_all, cell, _ = self._setup(atoms, forces)
        with torch.set_grad_enabled(forces):
            e_rep = self._repulsive(atoms, pos_all, cell)
        if forces:
            (g,) = torch.autograd.grad(e_rep, pos_all)
            F -= g.detach().cpu().numpy() / BOHR
        return dict(
            energy=e_band + float(e_rep),
            e_band=e_band,
            e_rep=float(e_rep),
            forces=F if forces else None,
            mu=mu_ha * H2E,
            n_electrons=n_count,
            n_electrons_expected=n_el,
            n_pieces=len(parts),
            max_piece_atoms=max(sizes),
            seconds=time.perf_counter() - t0,
        )
