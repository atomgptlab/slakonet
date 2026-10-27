"""Training SlaKoNet Slater-Koster parameters on spin-polarized DFT(+U) bands.

The magnetic part of the model (atomic Stoner + Dudarev U + SCC, see
``slakonet.hubbard_spin``) is kept as it is. What is trained are small,
smooth corrections to the Slater-Koster Hamiltonian integrals of the pairs
that matter for magnetism (3d metal - ligand and 3d metal - 3d metal), and
on-site level shifts, so that the spin-up and spin-down bands of the
converged +U model reproduce DFT(+U) bands of collinear magnetic
configurations.

Frozen-potential scheme
-----------------------
For every training structure the +U model is converged once with the
current parameters; its potentials (SCC shift, Stoner splitting, U blocks)
are stored. During training

    H_sigma(k) = H0(k; theta) + delta(theta) + 0.5 S(k) o (s_i + s_j)
                 + diag(+-V) + U (1/2 - n_sigma)

with S(k) and the potentials fixed, so the gradient goes only through the
Slater-Koster integrals theta. The potentials are refreshed between training
rounds (``refresh`` stage), which makes the scheme self-consistent overall.

Parametrization (per tied integral: element pair, l of each atom, bond type)

    h(r) = h0(r) * exp(a - b (r - r_ref)),  a = A tanh(u),  b = B tanh(v)

i.e. a bounded scale and a bounded change of decay length; on-site levels
of the trained elements shift by D tanh(w) per shell. The overlap is not
changed (it stays positive definite). ``u, v, w`` start at zero = the base
parameter set.

Loss: per structure, mean squared difference of the matched band energies
of both spins (top valence bands within a window below E_F and the lowest
conduction bands), after one rigid shift common to the two spins (so the
exchange splitting is fitted, not absorbed), plus an L2 penalty on u, v, w.

Data: OUTCAR files (final eigenvalues of both spins at the irreducible
k-points), read once into compact .npz records by ``build_dataset``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import torch

H2E = 27.211386
TRANSITION_3D = ("Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn")
IONIC_LIGANDS = ("O", "F", "Cl", "Br", "I", "S", "Se", "Te")


# ---------------------------------------------------------------------------
# 1. DFT data
# ---------------------------------------------------------------------------
def read_outcar_bands(path):
    """Final band energies of a spin-polarized VASP run.

    Returns dict with kpts [nk, 3] (reduced), weights [nk], eig and occ
    [2, nk, nb] (eV), efermi (eV), moments [natom] or None."""
    text = Path(path).read_text(errors="replace")
    lines = text.splitlines()
    nk = int(re.search(r"NKPTS\s*=\s*(\d+)", text).group(1))
    nb = int(re.search(r"NBANDS\s*=\s*(\d+)", text).group(1))
    start = next(
        i
        for i, l in enumerate(lines)
        if "k-points in reciprocal lattice and weights" in l
    )
    kw = np.array(
        [
            [float(x) for x in lines[start + 1 + i].split()[:4]]
            for i in range(nk)
        ]
    )
    kpts, weights = kw[:, :3], kw[:, 3] / kw[:, 3].sum()
    efermi = float(re.findall(r"E-fermi :\s*([-\d.]+)", text)[-1])

    s1 = [i for i, l in enumerate(lines) if l.strip() == "spin component 1"][
        -1
    ]
    eig = np.zeros((2, nk, nb))
    occ = np.zeros((2, nk, nb))
    i = s1
    for s in range(2):
        while lines[i].strip() != f"spin component {s + 1}":
            i += 1
        for k in range(nk):
            while not lines[i].strip().startswith("k-point"):
                i += 1
            i += 2  # "band No.  band energies  occupation"
            for b in range(nb):
                parts = lines[i + b].split()
                eig[s, k, b], occ[s, k, b] = float(parts[1]), float(parts[2])
            i += nb
    return {
        "kpts": kpts,
        "weights": weights,
        "eig": eig,
        "occ": occ,
        "efermi": efermi,
        "moments": _outcar_moments(lines),
    }


def _outcar_moments(lines):
    starts = [
        k
        for k, l in enumerate(lines)
        if l.strip().startswith("magnetization (x)")
    ]
    if not starts:
        return None
    mom = []
    for l in lines[starts[-1] + 4 :]:
        parts = l.split()
        if not parts or not parts[0].isdigit():
            break
        mom.append(float(parts[-1]))
    return np.array(mom)


def _incar(path):
    tags = {}
    for l in Path(path).read_text().splitlines():
        if "=" in l:
            k, v = l.split("=", 1)
            tags[k.strip().upper()] = v.split("#")[0].strip()
    return tags


def _magmom_sigma(incar_tags, natom):
    vals = []
    for tok in incar_tags["MAGMOM"].split():
        n, v = tok.split("*") if "*" in tok else ("1", tok)
        vals += [float(v)] * int(n)
    if len(vals) != natom:
        raise ValueError(f"MAGMOM has {len(vals)} entries for {natom} atoms")
    return np.sign(vals)


def _dft_U(incar_tags, symbols_poscar_order):
    """U_eff (eV) on the 3d species of a run (0 without DFT+U)."""
    if incar_tags.get("LDAU", "").upper().strip(".") not in ("TRUE", "T"):
        return 0.0
    us = [float(x) for x in incar_tags["LDAUU"].split()]
    vals = [
        u for el, u in zip(symbols_poscar_order, us) if el in TRANSITION_3D
    ]
    return max(vals) if vals else 0.0


def _converged(cdir):
    """Same acceptance as the J collectors: CONVERGED marker or an OUTCAR that
    reached EDIFF; FAILED_FINAL rejected."""
    if (cdir / "FAILED_FINAL").exists():
        return False
    if (cdir / "CONVERGED").exists():
        return True
    out = cdir / "OUTCAR"
    return (
        out.exists()
        and "aborting loop because EDIFF is reached"
        in out.read_text(errors="replace")
    )


def scan_batch(batch_dir, spin_state_mub=0.5, threshold=0.15):
    """Usable configurations of one batch directory (with manifest.json).

    Returns (U_eff eV, [config dirs]): converged, collinear MAGMOM pattern kept,
    |m| within spin_state_mub of the batch median."""
    from ase.io import read

    batch_dir = Path(batch_dir)
    man = json.loads((batch_dir / "manifest.json").read_text())
    rows, U = [], None
    for c in man["configs"]:
        cdir = batch_dir / c["dir"]
        if not (cdir / "OUTCAR").exists() or not _converged(cdir):
            continue
        incar = _incar(
            cdir / "INCAR.orig"
            if (cdir / "INCAR.orig").exists()
            else cdir / "INCAR"
        )
        species = (cdir / "POSCAR").read_text().splitlines()[5].split()
        if U is None:
            U = _dft_U(incar, species)
        atoms = read(cdir / "POSCAR")
        sigma = _magmom_sigma(incar, len(atoms))
        mom = _outcar_moments(
            (cdir / "OUTCAR").read_text(errors="replace").splitlines()
        )
        if mom is None or len(mom) != len(sigma):
            continue
        mag = sigma != 0
        if np.any(np.abs(mom[mag]) < threshold):
            continue
        rel = np.sign(mom[mag]) * sigma[mag]
        if not np.all(rel == rel[0]):
            continue
        rows.append(
            (cdir, c.get("label", c["dir"]), float(np.abs(mom[mag]).mean()))
        )
    if not rows:
        return U, []
    med = float(np.median([r[2] for r in rows]))
    return U, [
        (d, lab) for d, lab, m in rows if abs(m - med) <= spin_state_mub
    ]


def _pick_configs(configs, max_configs):
    """FM and the reference/ground-state order first, then the others."""

    def rank(item):
        lab = item[1].lower()
        return (
            0 if lab == "fm" else 1 if lab in ("ref", "ins_gs") else 2,
            str(item[0]),
        )

    return sorted(configs, key=rank)[:max_configs]


def build_dataset(batch_dirs, out_dir, u_windows, max_configs=4, holdout=()):
    """Write one .npz record per selected configuration and an index.json.

    batch_dirs : iterable of batch directories (each with manifest.json).
    u_windows : {anion class: (U_min, U_max)} in eV; for each compound the
        batch whose DFT U lies in the window of its class (closest to the
        window centre) is used.
    holdout : compound names kept for validation (flag in the index).
    """
    from ase.io import read

    from slakonet.magnetic_sites import site_anion_class

    out_dir = Path(out_dir)
    (out_dir / "structures").mkdir(parents=True, exist_ok=True)
    by_compound = {}
    for b in batch_dirs:
        b = Path(b)
        if not (b / "manifest.json").exists():
            continue
        man = json.loads((b / "manifest.json").read_text())
        name = man.get("compound") or b.name.rsplit("_U", 1)[0]
        U, configs = scan_batch(b)
        if configs:
            by_compound.setdefault(name, []).append((U, b, configs))

    index = []
    for name, batches in sorted(by_compound.items()):
        atoms0 = read(batches[0][2][0][0] / "POSCAR")
        sig0 = _magmom_sigma(
            _incar(_incar_path(batches[0][2][0][0])), len(atoms0)
        )
        cls = site_anion_class(atoms0, int(np.where(sig0 != 0)[0][0]))
        lo, hi = u_windows.get(cls, u_windows.get("default", (2.5, 5.5)))
        inside = [
            x
            for x in batches
            if x[0] is not None and lo - 1e-6 <= x[0] <= hi + 1e-6
        ]
        if not inside:
            print(
                f"{name} ({cls}): no batch with U in [{lo}, {hi}] eV "
                f"(available {sorted({x[0] for x in batches})}), skipped"
            )
            continue
        # one U per compound: closest to the window centre; all its batches
        u_best = min(
            {x[0] for x in inside}, key=lambda u: abs(u - 0.5 * (lo + hi))
        )
        configs = [c for x in inside if x[0] == u_best for c in x[2]]
        for cdir, label in _pick_configs(configs, max_configs):
            atoms = read(cdir / "POSCAR")
            sigma = _magmom_sigma(_incar(_incar_path(cdir)), len(atoms))
            bands = read_outcar_bands(cdir / "OUTCAR")
            rid = f"{name}__{cdir.parent.name}__{cdir.name}"
            np.savez_compressed(
                out_dir / "structures" / f"{rid}.npz",
                symbols=np.array(atoms.get_chemical_symbols()),
                positions=atoms.get_positions(),
                cell=np.array(atoms.cell),
                sigma=sigma,
                U_dft=u_best,
                anion_class=cls,
                kpts=bands["kpts"],
                weights=bands["weights"],
                eig=bands["eig"],
                occ=bands["occ"],
                efermi=bands["efermi"],
                moments=(
                    bands["moments"]
                    if bands["moments"] is not None
                    else np.zeros(0)
                ),
            )
            index.append(
                {
                    "id": rid,
                    "compound": name,
                    "class": cls,
                    "U_dft": u_best,
                    "label": label,
                    "natoms": len(atoms),
                    "source": str(cdir),
                    "holdout": name in set(holdout),
                }
            )
        print(
            f"{name:14s} {cls:13s} U_dft={u_best:4.2f}  {min(len(configs), max_configs)} configs"
        )
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    return index


def _incar_path(cdir):
    return (
        cdir / "INCAR.orig"
        if (cdir / "INCAR.orig").exists()
        else cdir / "INCAR"
    )


def load_record(path):
    with np.load(path, allow_pickle=False) as z:
        rec = {k: z[k] for k in z.files}
    rec["symbols"] = [str(s) for s in rec["symbols"]]
    for k in ("U_dft", "efermi"):
        rec[k] = float(rec[k])
    rec["anion_class"] = str(rec["anion_class"])
    return rec


def record_atoms(rec):
    from ase import Atoms

    return Atoms(
        rec["symbols"], positions=rec["positions"], cell=rec["cell"], pbc=True
    )


# ---------------------------------------------------------------------------
# 2. Frozen +U potentials
# ---------------------------------------------------------------------------
def precompute_potential(
    rec, model, u_table="universal", kmesh=None, max_iter=300, hubbard_U=None
):
    """Converge the +U model of one record; return the frozen potentials and
    checks (TB moments against DFT), as a dict of arrays for np.savez.

    hubbard_U : passed to build_hubbard_model; {symbol: 0.0} for every 3d
        metal gives the model without U (consistent with PBE, U = 0 bands)."""
    from slakonet.hubbard_spin import build_hubbard_model

    atoms = record_atoms(rec)
    if kmesh is None:  # ~ 24 / |a_i| k-points per direction, as the J pipeline
        lengths = atoms.cell.lengths()
        kmesh = tuple(int(max(1, math.ceil(24.0 / L))) for L in lengths)
    m_dft = rec["moments"] if len(rec["moments"]) == len(atoms) else None
    init = rec["sigma"] * (np.abs(m_dft) if m_dft is not None else 3.0)
    sm = build_hubbard_model(
        atoms,
        model,
        hubbard_U=hubbard_U,
        u_table=u_table,
        initial_moments=init,
        kmesh=kmesh,
        use_scc=True,
        max_iter=max_iter,
        verbose=False,
    )
    pot = sm.potentials
    blocks = sorted(pot["U_up"])
    out = {
        "converged": sm.converged,
        "efermi_tb": sm.efermi,
        "moments_tb": sm.moments,
        "scc_shift": pot["scc_shift"],
        "stoner_V": pot["stoner_V"],
        "U_atoms": np.array(blocks, dtype=int),
        "U_idx": np.array([pot["U_up"][a][0] for a in blocks]).reshape(-1, 5),
        "U_up": np.array([pot["U_up"][a][1] for a in blocks]).reshape(
            -1, 5, 5
        ),
        "U_dn": np.array([pot["U_dn"][a][1] for a in blocks]).reshape(
            -1, 5, 5
        ),
        "kmesh": np.array(kmesh),
    }
    mag = rec["sigma"] != 0
    if m_dft is not None:
        same_sign = np.all(np.sign(sm.moments[mag]) == np.sign(m_dft[mag]))
        dm = float(np.abs(np.abs(sm.moments[mag]) - np.abs(m_dft[mag])).max())
    else:
        same_sign, dm = True, 0.0
    out["usable"] = bool(sm.converged and same_sign and dm < 1.0)
    out["max_moment_error"] = dm
    return out


# ---------------------------------------------------------------------------
# 3. Trainable corrections
# ---------------------------------------------------------------------------
class SKCorrections(torch.nn.Module):
    """Bounded scale / decay corrections of the H integrals and on-site shifts.

    Integrals are tied across the two orientations of a pair: the id is
    (sorted element pair, l on the first, l on the second, bond index)."""

    def __init__(
        self,
        train_elements,
        r_ref_bohr=4.0,
        max_log_scale=0.4,
        max_decay=0.3,
        max_onsite_ev=2.0,
        pair_mode="all",
        onsite_elements=None,
    ):
        """pair_mode: "all" (3d metal with any trained element) or "ionic" (3d
        metal - ionic ligand only: metal-metal and metal-pnictogen pairs, which
        control the metallic magnets, stay at the base values).
        onsite_elements: elements whose on-site levels may shift (default: all
        trained elements); e.g. the ligands only, so that the charge-transfer
        energy is tuned without moving the metal d levels in every compound."""
        super().__init__()
        self.train_elements = sorted(set(train_elements))
        self.r_ref, self.A, self.B, self.D = (
            r_ref_bohr,
            max_log_scale,
            max_decay,
            max_onsite_ev,
        )
        if pair_mode not in ("all", "ionic"):
            raise ValueError(
                f"pair_mode must be 'all' or 'ionic', got {pair_mode!r}"
            )
        self.pair_mode = pair_mode
        self.onsite_elements = (
            sorted(set(onsite_elements))
            if onsite_elements is not None
            else list(self.train_elements)
        )
        self.u = torch.nn.ParameterDict()
        self.v = torch.nn.ParameterDict()
        self.w = torch.nn.ParameterDict()

    @staticmethod
    def _safe(key):
        return key.replace("-", "_").replace("|", "__")

    def is_trained_pair(self, e1, e2):
        t = set(self.train_elements)
        if not (
            e1 in t
            and e2 in t
            and (e1 in TRANSITION_3D or e2 in TRANSITION_3D)
        ):
            return False
        if self.pair_mode == "ionic":
            return (e1 in IONIC_LIGANDS) != (e2 in IONIC_LIGANDS)
        return True

    def integral_id(self, e1, e2, l1, l2, row):
        if (e1, l1) > (e2, l2) and e1 != e2:
            e1, e2, l1, l2 = e2, e1, l2, l1
        elif e1 == e2 and l1 > l2:
            l1, l2 = l2, l1
        return f"{e1}-{e2}|{l1}-{l2}|{row}"

    def ensure(self, e1, e2, h_keys_rows):
        """Create the parameters of a pair (called once per pair)."""
        for key, nrow in h_keys_rows:
            l1, l2 = (int(x) for x in key.split("-"))
            for row in range(nrow):
                sid = self._safe(self.integral_id(e1, e2, l1, l2, row))
                if sid not in self.u:
                    self.u[sid] = torch.nn.Parameter(torch.zeros(()))
                    self.v[sid] = torch.nn.Parameter(torch.zeros(()))

    def ensure_onsite(self, element, ls):
        if element not in self.onsite_elements:
            return
        for l in ls:
            sid = f"{element}_{l}"
            if sid not in self.w:
                self.w[sid] = torch.nn.Parameter(torch.zeros(()))

    def scale(self, e1, e2, l1, l2, row, r):
        sid = self._safe(self.integral_id(e1, e2, l1, l2, row))
        a = self.A * torch.tanh(self.u[sid])
        b = self.B * torch.tanh(self.v[sid])
        return torch.exp(a - b * (r - self.r_ref))

    def onsite_ev(self, element, l):
        sid = f"{element}_{l}"
        if sid not in self.w:
            return 0.0
        return self.D * torch.tanh(self.w[sid])

    def penalty(self):
        p = [x**2 for d in (self.u, self.v, self.w) for x in d.values()]
        return torch.stack(p).sum() if p else torch.zeros(())

    def summary(self):
        out = {"integrals": {}, "onsite_eV": {}}
        for sid in self.u:
            out["integrals"][sid] = {
                "log_scale": float(self.A * torch.tanh(self.u[sid])),
                "decay_per_bohr": float(self.B * torch.tanh(self.v[sid])),
            }
        for sid in self.w:
            out["onsite_eV"][sid] = float(self.D * torch.tanh(self.w[sid]))
        return out

    def export(self, path, meta=None):
        torch.save(
            {
                "state": self.state_dict(),
                "train_elements": self.train_elements,
                "hyper": [self.r_ref, self.A, self.B, self.D],
                "pair_mode": self.pair_mode,
                "onsite_elements": self.onsite_elements,
                "meta": meta or {},
            },
            path,
        )
        Path(path).with_suffix(".json").write_text(
            json.dumps({"meta": meta or {}, **self.summary()}, indent=1)
        )

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=False)
        obj = cls(
            d["train_elements"],
            *d["hyper"],
            pair_mode=d.get("pair_mode", "all"),
            onsite_elements=d.get("onsite_elements"),
        )
        for name in ("u", "v", "w"):
            for k in d["state"]:
                if k.startswith(name + "."):
                    getattr(obj, name)[k.split(".", 1)[1]] = (
                        torch.nn.Parameter(d["state"][k].clone())
                    )
        return obj


class _ScaledInterp:
    """Wraps a Slater-Koster interpolator: h(r) -> h(r) * scale(r) per bond
    type. slakonet's spline interpolator is built with scipy (no gradient),
    so the correction is applied to the interpolated values; for a smooth
    scale this equals interpolating the scaled table."""

    def __init__(self, base, corr, e1, e2, l1, l2):
        self.base, self.corr, self.key = base, corr, (e1, e2, l1, l2)

    def __call__(self, rr, *args, **kwargs):
        y = self.base(rr, *args, **kwargs)  # [..., n_bond]
        e1, e2, l1, l2 = self.key
        r = torch.as_tensor(rr, dtype=y.dtype)
        while r.dim() < y.dim():
            r = r.unsqueeze(-1)
        scales = torch.cat(
            [
                self.corr.scale(e1, e2, l1, l2, row, r).to(y.dtype)
                for row in range(y.shape[-1])
            ],
            dim=-1,
        )
        return y * scales


def build_feeds(model, corr):
    """Hamiltonian feed of the whole model with the corrections installed on
    the trained pairs, and the (frozen) overlap feed. Built once; the
    corrections are read from ``corr`` at every evaluation."""
    from jarvis.core.specie import atomic_numbers_to_symbols

    from slakonet.utils import generate_shell_dict_upto_Z65

    shell_dict = generate_shell_dict_upto_Z65(model=model)
    with torch.no_grad():
        skfs = model.get_updated_skfs()
    h_feed = model._create_comprehensive_feed(skfs, shell_dict, "H")
    sym = {}
    for key in list(h_feed.off_site_dict):
        z1, z2, l1, l2 = (int(x) for x in key)
        for z in (z1, z2):
            if z not in sym:
                sym[z] = atomic_numbers_to_symbols([z])[0]
        e1, e2 = sym[z1], sym[z2]
        if not corr.is_trained_pair(e1, e2):
            continue
        base = h_feed.off_site_dict[key]
        nrow = min(l1, l2) + 1
        corr.ensure(e1, e2, [(f"{l1}-{l2}", nrow)])
        h_feed.off_site_dict[key] = _ScaledInterp(base, corr, e1, e2, l1, l2)
    for el in corr.train_elements:
        z = next((zz for zz, s in sym.items() if s == el), None)
        if z is not None and z in shell_dict:
            corr.ensure_onsite(el, sorted(set(shell_dict[z])))
    return h_feed, shell_dict


# ---------------------------------------------------------------------------
# 4. One training structure
# ---------------------------------------------------------------------------
def potential_k(pot, Sk, orb_atom, flip=None):
    """Frozen potential in k space, [dH_up, dH_dn] (eV, [nk, N, N]):
    0.5 S(k) o (s_i + s_j) + diag(+-V) + U blocks.

    flip : optional [Natom] array of +-1; atoms with -1 get their spin
    potential reversed (Stoner V sign, U up/down blocks swapped): the rigid
    rotation used to build another collinear configuration from a reference
    one (magnetic force theorem)."""
    N = Sk.shape[-1]
    oa = torch.as_tensor(orb_atom)
    s_orb = torch.as_tensor(pot["scc_shift"], dtype=torch.float64)[oa]
    base = 0.5 * Sk * (s_orb[:, None] + s_orb[None, :]).to(torch.complex128)
    V = torch.as_tensor(pot["stoner_V"], dtype=torch.float64)
    if flip is not None:
        V = V * torch.as_tensor(np.asarray(flip, dtype=float))[oa]
    dH = [
        base + torch.diag(V).to(torch.complex128),
        base - torch.diag(V).to(torch.complex128),
    ]
    blocks = {
        int(a): (idx, up, dn)
        for a, idx, up, dn in zip(
            pot["U_atoms"], pot["U_idx"], pot["U_up"], pot["U_dn"]
        )
    }
    for s in range(2):
        onsite = torch.zeros((N, N), dtype=torch.complex128)
        for a, (idx, up, dn) in blocks.items():
            swap = flip is not None and flip[a] < 0
            blk = (dn if s == 0 else up) if swap else (up if s == 0 else dn)
            onsite[np.ix_(idx, idx)] += torch.as_tensor(
                blk, dtype=torch.complex128
            )
        dH[s] = dH[s] + onsite
    return dH


def _fermi_mu(e, w, nelec, kT):
    """Chemical potential (detached) with sum_k w_k sum_s,n f = nelec."""
    e, w = e.detach(), w[None, :, None]
    lo, hi = float(e.min()) - 1.0, float(e.max()) + 1.0
    for _ in range(60):
        mu = 0.5 * (lo + hi)
        n = float((w * torch.special.expit(-(e - mu) / kT)).sum())
        lo, hi = (mu, hi) if n < nelec else (lo, mu)
    return 0.5 * (lo + hi)


class TrainingStructure:
    """Everything fixed for one record: geometry objects, S(k), the frozen
    potential in k space, band matching."""

    def __init__(
        self,
        rec,
        pot,
        model,
        h_feed,
        corr,
        cutoff=10.0,
        window_v=8.0,
        window_c=3.0,
        n_cond_max=12,
        device="cpu",
    ):
        from slakonet.atoms import Geometry
        from slakonet.main import SimpleDftb
        from slakonet.tb2j_interface import (
            _bloch,
            extract_real_space_hs,
            orbital_labels,
        )

        self.rec, self.pot = rec, pot
        atoms = record_atoms(rec)
        self.elements = set(rec["symbols"])
        self.calc = SimpleDftb(
            Geometry.from_ase_atoms([atoms]),
            model,
            cutoff=cutoff,
            kpoints=torch.tensor([1, 1, 1]),
            device=device,
            compute_forces=False,
            include_dos_data=False,
            repulsive=False,
        )
        self.h_feed = h_feed
        labels, orb_atom, orb_l = orbital_labels(self.calc, atoms)
        self.orb_atom, self.orb_l = orb_atom, orb_l
        self.orb_el = [rec["symbols"][a] for a in orb_atom]
        Rlist, _, SR = extract_real_space_hs(self.calc)
        k = torch.as_tensor(rec["kpts"], dtype=torch.float64)
        self.kpts = k
        self.weights = torch.as_tensor(rec["weights"], dtype=torch.float64)
        Sk = _bloch(
            torch.as_tensor(Rlist, dtype=torch.float64), torch.as_tensor(SR), k
        )
        # Cholesky factor in numpy: no gradient is needed here, and torch's
        # multithreaded MKL batched LU failed on an HPC cluster (ZLASWP
        # parameter error)
        try:
            L = np.linalg.cholesky(Sk.numpy())
        except np.linalg.LinAlgError as err:
            raise ValueError(
                f"S(k) not positive definite for {rec.get('id', '?')}: {err}"
            )
        self.Linv = torch.from_numpy(np.linalg.inv(L))
        self.Sk = Sk
        self.nelectron = float(self.calc.nelectron.flatten()[0])
        self.dH = potential_k(pot, Sk, orb_atom)  # eV, [nk, N, N] per spin
        self._set_kpoints()
        self._match_bands(window_v, window_c, n_cond_max, corr)

    def _set_kpoints(self):
        per = self.calc.periodic
        per.kpoints = self.kpts.to(per.kpoints.dtype).unsqueeze(0)
        per.n_kpoints = torch.tensor([len(self.kpts)])

    def h0k(self, corr):
        """H0(k) in eV, [nk, N, N] complex, differentiable in corr."""
        from slakonet.slaterkoster import hs_matrix

        Hk = hs_matrix(self.calc.periodic, self.calc.basis, self.h_feed)[
            0
        ]  # [N, N, nk]
        Hk = Hk.permute(2, 0, 1).to(torch.complex128) * H2E
        shifts = []
        for el, l in zip(self.orb_el, self.orb_l):
            d = (
                corr.onsite_ev(el, int(l))
                if el in corr.train_elements
                else 0.0
            )
            shifts.append(d if torch.is_tensor(d) else torch.tensor(float(d)))
        return Hk + torch.diag_embed(torch.stack(shifts).to(torch.complex128))

    def bands(self, corr):
        """TB band energies of both spins, eV, [2, nk, N]."""
        H0 = self.h0k(corr)
        out = []
        for s in range(2):
            Ht = (
                self.Linv
                @ (H0 + self.dH[s])
                @ self.Linv.conj().transpose(-1, -2)
            )
            Ht = 0.5 * (Ht + Ht.conj().transpose(-1, -2))
            out.append(torch.linalg.eigvalsh(Ht))
        return torch.stack(out)

    def bands_with(self, corr, dH):
        H0 = self.h0k(corr)
        out = []
        for s in range(2):
            Ht = self.Linv @ (H0 + dH[s]) @ self.Linv.conj().transpose(-1, -2)
            Ht = 0.5 * (Ht + Ht.conj().transpose(-1, -2))
            out.append(torch.linalg.eigvalsh(Ht))
        return torch.stack(out)

    def band_energy(self, corr, dH, kT=0.05):
        """sum_k w_k sum_s,n f eps (eV) at fixed electron number, Fermi
        smearing kT (eV); differentiable in corr."""
        e = self.bands_with(corr, dH)
        mu = _fermi_mu(e, self.weights, self.nelectron, kT)
        f = torch.special.expit(-(e - mu) / kT)
        return (self.weights[None, :, None] * f * e).sum()

    def _match_bands(self, window_v, window_c, n_cond_max, corr):
        """Index pairs (TB band, DFT band) per spin and k: the top occupied
        bands within window_v below E_F counted down from the top of the
        occupied manifold, and the lowest empty bands within window_c."""
        with torch.no_grad():  # corrections at their current values
            tb = self.bands(corr).numpy()
        ef_tb, ef_dft = self.pot["efermi_tb"], self.rec["efermi"]
        eig, occ = self.rec["eig"], self.rec["occ"]
        self.pairs = []  # (spin, k, tb index array, dft index array)
        for s in range(2):
            for k in range(eig.shape[1]):
                d_occ = np.where(occ[s, k] > 0.5)[0]
                d_occ = d_occ[eig[s, k, d_occ] > ef_dft - window_v][::-1]
                t_occ = np.where(tb[s, k] < ef_tb)[0]
                t_occ = t_occ[tb[s, k, t_occ] > ef_tb - window_v - 2.0][::-1]
                nv = min(len(d_occ), len(t_occ))
                d_emp = np.where(
                    (occ[s, k] <= 0.5) & (eig[s, k] < ef_dft + window_c)
                )[0]
                t_emp = np.where(tb[s, k] >= ef_tb)[0]
                nc = min(len(d_emp), len(t_emp), n_cond_max)
                ti = np.concatenate([t_occ[:nv], t_emp[:nc]]).astype(int)
                di = np.concatenate([d_occ[:nv], d_emp[:nc]]).astype(int)
                if len(ti):
                    self.pairs.append((s, k, ti, di, nv))
        self.n_matched = sum(len(p[2]) for p in self.pairs)

    def loss(self, corr, w_cond=0.5):
        """Weighted mean squared band error (eV^2) after a common rigid shift."""
        e = self.bands(corr)
        res, wts = [], []
        for s, k, ti, di, nv in self.pairs:
            diff = e[s, k, torch.as_tensor(ti)] - torch.as_tensor(
                self.rec["eig"][s, k, di], dtype=e.dtype
            )
            w = torch.full((len(ti),), float(self.weights[k]), dtype=e.dtype)
            w[nv:] *= w_cond
            res.append(diff)
            wts.append(w)
        r, w = torch.cat(res), torch.cat(wts)
        shift = (w * r).sum() / w.sum()
        return (w * (r - shift) ** 2).sum() / w.sum()


class EnergyGroup:
    """Magnetic energy differences of the configurations of one DFT batch
    (same cell, same settings), by the magnetic force theorem:

        dE_TB(b) = E_band[b, rotated reference potential] - E_band[ref]

    The reference potential (FM if present) is rotated rigidly onto each
    configuration (site spins flipped), so the double counting cancels; the
    loss compares dE_TB with the DFT total-energy differences, per magnetic
    atom."""

    def __init__(self, members, energies, kT=0.05):
        """members: [TrainingStructure] of one batch; energies: {id: E_dft eV}."""
        self.kT = kT
        ref = next(
            (
                m
                for m in members
                if str(m.rec.get("label", "")).lower() == "fm"
            ),
            members[0],
        )
        pos0 = ref.rec["positions"]
        self.ref = ref
        self.members, self.target = [], []
        sig_ref = np.asarray(ref.rec["sigma"], dtype=float)
        self.n_mag = int((sig_ref != 0).sum())
        for m in members:
            if (
                m is ref
                or m.rec["id"] not in energies
                or ref.rec["id"] not in energies
            ):
                continue
            if (
                m.rec["positions"].shape != pos0.shape
                or np.abs(m.rec["positions"] - pos0).max() > 1e-3
            ):
                continue
            sig = np.asarray(m.rec["sigma"], dtype=float)
            flip = np.where((sig != 0) & (sig_ref != 0), sig * sig_ref, 1.0)
            m.dH_rot = potential_k(ref.pot, m.Sk, m.orb_atom, flip)
            self.members.append(m)
            self.target.append(energies[m.rec["id"]] - energies[ref.rec["id"]])
        self.target = torch.as_tensor(self.target, dtype=torch.float64)

    def __len__(self):
        return len(self.members)

    def energies(self, corr):
        """dE_TB per configuration (eV per cell)."""
        e_ref = self.ref.band_energy(corr, self.ref.dH, self.kT)
        return torch.stack(
            [
                m.band_energy(corr, m.dH_rot, self.kT) - e_ref
                for m in self.members
            ]
        )

    def loss(self, corr, huber_delta=0.0):
        """Mean squared error of dE per magnetic atom (eV^2); with
        huber_delta > 0 (eV), quadratic below delta and linear above, so that
        a few configurations whose DFT energy differences are not purely
        magnetic (orbital-state changes) cannot dominate."""
        d = (self.energies(corr) - self.target) / max(self.n_mag, 1)
        if huber_delta > 0:
            a = d.abs()
            return torch.where(
                a <= huber_delta, d**2, 2 * huber_delta * a - huber_delta**2
            ).mean()
        return (d**2).mean()


def vasp_energy(cdir):
    """Free energy (eV) of a converged VASP run (CONVERGED marker, else the
    last F= of OSZICAR), or None."""
    cdir = Path(cdir)
    marker = cdir / "CONVERGED"
    if marker.exists():
        fields = dict(
            kv.split("=", 1) for kv in marker.read_text().split() if "=" in kv
        )
        if "energy_eV" in fields:
            return float(fields["energy_eV"])
    osz = cdir / "OSZICAR"
    if osz.exists():
        lines = [l for l in osz.read_text().splitlines() if "F=" in l]
        if lines:
            return float(lines[-1].split("F=")[1].split()[0])
    return None


def group_key(rid):
    """Records of the same DFT batch: '<compound>__<batch dir>'."""
    return rid.rsplit("__", 1)[0]


# ---------------------------------------------------------------------------
# 5. Using the trained corrections
# ---------------------------------------------------------------------------
def apply_corrections(model, corr):
    """Bake the corrections into ``model`` in place: scaled H tables of the
    trained pairs and shifted on-site levels of the trained elements. After
    this the model is an ordinary slakonet model (the TB2J / +U pipeline uses
    it unchanged)."""
    from jarvis.core.specie import atomic_numbers_to_symbols

    with torch.no_grad():
        for key, opt in model.skf_optimizers.items():
            zs = [
                int(z)
                for z in (
                    opt.atom_pair.tolist()
                    if hasattr(opt.atom_pair, "tolist")
                    else opt.atom_pair
                )
            ]
            e1, e2 = atomic_numbers_to_symbols(zs)
            if corr.is_trained_pair(e1, e2):
                grid = torch.as_tensor(opt.grid, dtype=torch.float64)
                for k, p in opt.h_params.items():
                    l1, l2 = (int(x) for x in k.split("-"))
                    sid = corr._safe(corr.integral_id(e1, e2, l1, l2, 0))
                    if sid not in corr.u:
                        continue
                    for row in range(p.shape[0]):
                        p.data[row] = (
                            p.data[row].to(torch.float64)
                            * corr.scale(e1, e2, l1, l2, row, grid)
                        ).to(p.dtype)
            if e1 == e2 and e1 in corr.train_elements:
                on = opt.skf_dict["atomic_data"]["on_sites"]
                for l in range(len(on)):
                    d = corr.onsite_ev(e1, l)
                    on[l] = on[l] + (
                        float(d) / H2E if torch.is_tensor(d) else d / H2E
                    )
    return model


def maybe_corrected(model):
    """Apply the corrections named by $SLAKONET_SK_CORRECTIONS (a .pt file from
    train_spin_bands.py) to ``model``; no-op when the variable is unset."""
    import os

    path = os.environ.get("SLAKONET_SK_CORRECTIONS")
    if not path:
        return model
    print(f"[slakonet] applying SK corrections from {path}")
    return apply_corrections(model, SKCorrections.load(path))
