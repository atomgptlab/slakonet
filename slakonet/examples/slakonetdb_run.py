"""SlaKoNetDB shard runner: one Slurm task processes a stride slice."""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
import numpy as np


def runnable_jids(model_elements, ehull_max=1e-6):
    from jarvis.db.figshare import data

    out = []
    for r in data("dft_3d"):
        try:
            e = float(r.get("ehull"))
        except (TypeError, ValueError):
            continue
        if np.isnan(e) or e > ehull_max:
            continue
        if set(r["atoms"]["elements"]) <= model_elements:
            out.append(r)
    out.sort(key=lambda r: r["jid"])  # deterministic sharding
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="slakonet_v1a_full")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-atoms", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    try:
        from slakonet.examples.slakonetdb_record import (
            build_record,
            write_record,
        )
    except ImportError:  # running the file directly from its directory
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from slakonetdb_record import build_record, write_record
    from jarvis.core.atoms import Atoms
    from slakonet.optim import default_model, default_mu
    import torch

    os.makedirs(a.out, exist_ok=True)
    model = default_model(model_name=a.model).float()
    elements = set(model.elements_in_system)
    mu = default_mu(model_name=a.model)

    rows = runnable_jids(elements)
    rows = [r for r in rows if len(r["atoms"]["elements"]) <= a.max_atoms]
    if a.limit:
        rows = rows[: a.limit]
    mine = rows[a.shard :: a.nshards]
    print(
        f"[shard {a.shard}/{a.nshards}] {len(mine)} of {len(rows)} "
        f"structures",
        flush=True,
    )

    logp = os.path.join(a.out, f"shard{a.shard:04d}.jsonl")
    done = 0
    with open(logp, "a") as log:
        for i, r in enumerate(mine, 1):
            jid = r["jid"]
            dest = os.path.join(a.out, f"{jid}.npz")
            if os.path.exists(dest):  # resumable
                continue
            t0 = time.time()
            try:
                at = Atoms.from_dict(r["atoms"]).ase_converter()
                rec = build_record(at, model, device=a.device, mu=mu)
                meta = dict(
                    jid=jid,
                    formula=at.get_chemical_formula(),
                    natoms=len(at),
                    model=a.model,
                    ehull=r.get("ehull"),
                    ref_gap_optb88vdw=r.get("optb88vdw_bandgap"),
                    ref_gap_mbj=r.get("mbj_bandgap"),
                    ref_eform=r.get("formation_energy_peratom"),
                    ref_bulk_modulus=r.get("bulk_modulus_kv"),
                    spg=r.get("spg_number"),
                )
                write_record(dest, rec, meta)
                log.write(
                    json.dumps(
                        dict(
                            jid=jid,
                            ok=True,
                            recon_err=rec["recon_err"],
                            gap=rec["gap"],
                            e_form=rec["e_form"],
                            fermi=rec["fermi"],
                            seconds=rec["seconds"],
                        )
                    )
                    + "\n"
                )
                done += 1
            except Exception as exc:
                log.write(
                    json.dumps(
                        dict(
                            jid=jid,
                            ok=False,
                            error=f"{type(exc).__name__}: {exc}",
                            tb=traceback.format_exc()[-400:],
                        )
                    )
                    + "\n"
                )
            log.flush()
            if i % 20 == 0:
                print(
                    f"  [{i}/{len(mine)}] {done} written "
                    f"({time.time() - t0:.0f}s last)",
                    flush=True,
                )
    print(f"[shard {a.shard}] finished, {done} records", flush=True)


if __name__ == "__main__":
    main()
