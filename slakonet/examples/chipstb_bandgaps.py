"""ChIPS-TB tutorial: band-gap benchmarking with SlaKoNetCalculator.

For a curated set of JARVIS-DFT materials this evaluates SlaKoNet
tight-binding band gaps **two ways** and reports which agrees better
with the MBJ reference gaps:

  1. **3x3x3 Monkhorst-Pack grid** -- ``calc.get_bandgap()``; one
     uniform-mesh solve, fast.
  2. **High-symmetry band path** -- ``calc.band_structure()``; the
     ASE standard k-path (PATH_NPOINTS k-points/material), which
     samples band extrema along the symmetry lines.

The model is loaded once and the calculator reused for every
structure -- the recommended high-throughput pattern.

Formation energies per atom are also computed
(`E_form = (E_total - sum_i n_i * mu_i) / N_atoms`) using slakonet's
bundled chemical potentials, or a user file via `MU_JSON`.

Outputs
-------
* ``chipstb_bandgaps.csv``  -- per-material MP & path gaps (+ E_form)
* ``chipstb_parity.png``    -- parity plot, both schemes vs MBJ

Run
---
    python chipstb_bandgaps.py
"""

import csv
import json
import os
import time

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms

from slakonet.optim import default_model, default_mu
from slakonet.ase_calc import SlaKoNetCalculator

# ---- benchmark set --------------------------------------------------
CHIPS_TB_JIDS = [
    1174, 1002, 1195, 8118, 8158, 107, 1327, 91, 41, 104, 113, 1145,
    116, 1180, 1183, 1189, 1198, 1201, 1267, 1294, 1300, 1312, 1315,
    1393, 1408, 1453, 17, 1702, 1954, 23, 299, 30, 32, 39, 5, 54, 57,
    7630, 7678, 7762, 7844, 7860, 8003, 8169, 8566, 8583, 890, 95, 96,
    97,
]

# Per-element chemical potentials for formation energies. Leave as None
# to use the chemical potentials bundled with slakonet
# (`slakonet.optim.default_mu()`); or set this to your own
# `{"mu_per_atom_eV": {...}}` JSON to override.
MU_JSON = None

KPOINTS = (3, 3, 3)
# k-points along the high-symmetry band path (fewer -> faster; 20 is
# enough to bracket VBM/CBM near the symmetry lines).
PATH_NPOINTS = 20

# Device for the eigensolves. CPU is the default because the
# high-symmetry band-path eigensolve currently hits a reproducible
# CUDA "illegal instruction" fault on new GPU architectures
# (Blackwell sm_120) -- the MP-grid solve runs fine on GPU, but the
# band-path path does not, and the fault poisons the CUDA context so
# the whole run cascades. CUDA_LAUNCH_BLOCKING=1 helps in isolated
# tests but is not a reliable fix. Use "cuda" only if your GPU stack
# handles the band-path eigensolve cleanly.
DEVICE = "cpu"


def main():
    # ---- load the model + calculator ONCE --------------------------
    t0 = time.perf_counter()
    model = default_model().float()
    # alpha=1.0 gives a physically meaningful total energy (needed for
    # formation energies); it does NOT affect eigenvalues, so band gaps
    # are unchanged. compute_forces/stress off -> fast energy+gap only.
    calc = SlaKoNetCalculator(
        model, kpoints=KPOINTS, alpha=1.0, device=DEVICE,
        compute_forces=False, compute_stress=False,
    )
    print(f"[*] model + calculator ready in "
          f"{time.perf_counter() - t0:.1f}s")

    if MU_JSON and os.path.exists(MU_JSON):
        mu = json.load(open(MU_JSON))["mu_per_atom_eV"]
        print(f"[*] formation energies ON — user mu.json "
              f"({len(mu)} elements)")
    else:
        mu = default_mu()  # bundled with slakonet
        print(f"[*] formation energies ON — slakonet default_mu "
              f"({len(mu)} elements)")

    # ---- JARVIS-DFT database (indexed by jid) ----------------------
    print("[*] loading JARVIS dft_3d ...")
    index = {row["jid"]: row for row in data("dft_3d")}

    rows = []
    # collected only where an MBJ reference exists, for the MAE compare
    mbj_arr, mp_arr, path_arr = [], [], []
    for n, jnum in enumerate(CHIPS_TB_JIDS, 1):
        jid = f"JVASP-{jnum}"
        entry = index.get(jid)
        if entry is None:
            print(f"  [{n:2d}/{len(CHIPS_TB_JIDS)}] {jid}: not in dft_3d")
            continue
        try:
            jatoms = Atoms.from_dict(entry["atoms"])
            ase_atoms = jatoms.ase_converter()
            ase_atoms.calc = calc

            t1 = time.perf_counter()
            e_total = ase_atoms.get_potential_energy()  # eV
            # (1) gap from the 3x3x3 Monkhorst-Pack grid
            gap_mp = float(calc.get_bandgap())
            # (2) gap from a high-symmetry band path (separate solve);
            #     keep the MP result even if the band path fails.
            try:
                gap_path = float(
                    calc.band_structure(
                        ase_atoms, npoints=PATH_NPOINTS
                    )["gap"]
                )
            except Exception as e:
                gap_path = None
                print(f"        (band-path failed: "
                      f"{type(e).__name__})")
            dt = time.perf_counter() - t1

            # formation energy (optional)
            e_form = None
            if mu is not None:
                comp = jatoms.composition.to_dict()
                if all(el in mu and abs(mu[el]) > 1e-9 for el in comp):
                    ref = sum(int(c) * mu[el] for el, c in comp.items())
                    e_form = (e_total - ref) / jatoms.num_atoms

            mbj = entry.get("mbj_bandgap")
            mbj_val = (
                float(mbj) if mbj not in (None, "na", "") else None
            )
            formula = jatoms.composition.reduced_formula
            rows.append({
                "jid": jid, "formula": formula,
                "sk_gap_mp_eV": gap_mp,
                "sk_gap_path_eV": gap_path,
                "mbj_gap_eV": mbj_val,
                "e_form_eV_per_atom": e_form,
            })
            if mbj_val is not None:
                mbj_arr.append(mbj_val)
                mp_arr.append(gap_mp)
                path_arr.append(
                    gap_path if gap_path is not None else np.nan
                )
            ef_str = (
                f"  E_form={e_form:+.3f}" if e_form is not None else ""
            )
            gp = f"{gap_path:.3f}" if gap_path is not None else "na"
            print(
                f"  [{n:2d}/{len(CHIPS_TB_JIDS)}] {jid} {formula:<10s} "
                f"gap_MP={gap_mp:.3f}  gap_path={gp}  "
                f"MBJ={mbj_val if mbj_val is not None else 'na'}"
                f"{ef_str}  ({dt:.1f}s)"
            )
        except Exception as e:
            print(f"  [{n:2d}/{len(CHIPS_TB_JIDS)}] {jid}: "
                  f"FAILED {type(e).__name__}: {str(e)[:120]}")

    # ---- write CSV -------------------------------------------------
    with open("chipstb_bandgaps.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["jid", "formula", "sk_gap_mp_eV",
                        "sk_gap_path_eV", "mbj_gap_eV",
                        "e_form_eV_per_atom"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\n[*] wrote chipstb_bandgaps.csv ({len(rows)} materials)")

    # ---- compare the two k-sampling schemes vs MBJ -----------------
    if mbj_arr:
        mbj = np.array(mbj_arr)
        mp = np.array(mp_arr)
        path = np.array(path_arr)

        mae_mp = float(np.mean(np.abs(mp - mbj)))
        m = np.isfinite(path)
        mae_path = (
            float(np.mean(np.abs(path[m] - mbj[m])))
            if m.any() else float("nan")
        )
        print()
        print(f"[*] band-gap MAE vs MBJ:")
        print(f"      3x3x3 MP grid     : {mae_mp:.3f} eV "
              f"(n={len(mbj)})")
        print(f"      high-symmetry path: {mae_path:.3f} eV "
              f"(n={int(m.sum())})")
        if np.isfinite(mae_path):
            best = ("3x3x3 MP grid" if mae_mp <= mae_path
                    else "high-symmetry path")
            print(f"[*] LOWEST ERROR: {best}")

        lim = max(mp.max(), np.nanmax(path) if m.any() else 0,
                  mbj.max(), 1.0) * 1.1
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(mbj, mp, s=30, alpha=0.8,
                   label=f"3x3x3 MP  (MAE {mae_mp:.3f})")
        if m.any():
            ax.scatter(mbj[m], path[m], s=30, alpha=0.8, marker="^",
                       label=f"high-sym path  (MAE {mae_path:.3f})")
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        ax.set_xlabel("MBJ band gap (eV)")
        ax.set_ylabel("SlaKoNet band gap (eV)")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_title("ChIPS-TB band gaps: MP grid vs high-sym path")
        fig.tight_layout()
        fig.savefig("chipstb_parity.png", dpi=200)
        plt.close(fig)
        print("[*] wrote chipstb_parity.png")
    else:
        print("[*] no MBJ references available -- skipped comparison")


if __name__ == "__main__":
    main()
