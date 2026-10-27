"""Train SlaKoNet Slater-Koster corrections on spin-polarized DFT(+U) bands.

Stages (see slakonet.spin_training for the method):

  dataset     OUTCARs -> compact .npz records + index.json (run where the
              VASP runs are; only the records travel to the cluster)
  precompute  converge the +U model of every record with the current
              parameters and store its frozen potentials (parallel)
  train       fit the corrections (data parallel, torch.distributed/gloo)
  evaluate    band RMSE per structure, base vs trained

A training "round" = precompute with the corrections of the previous round
+ train starting from them; 2-3 rounds make the frozen potential consistent.

Examples:
  python train_spin_bands.py dataset --data DATA \
      --sources ".../dft_round4_uscan/*_U*" ".../Magnetic_MLIP/dft_inputs/*"
  python train_spin_bands.py precompute --data DATA --round r0 --nproc 64
  python train_spin_bands.py train --data DATA --round r0 --nproc 64 --out DATA/corr_r0.pt
  python train_spin_bands.py precompute --data DATA --round r1 --corrections DATA/corr_r0.pt
  python train_spin_bands.py train --data DATA --round r1 --init DATA/corr_r0.pt --out DATA/corr_r1.pt
  python train_spin_bands.py evaluate --data DATA --round r1 --corrections DATA/corr_r1.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

TRANSITION_3D = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
LIGANDS = [
    "O",
    "F",
    "Cl",
    "Br",
    "I",
    "S",
    "Se",
    "Te",
    "N",
    "P",
    "As",
    "Sb",
    "Bi",
    "Sn",
    "Si",
    "Ge",
    "C",
]
DEFAULT_U_WINDOWS = (
    {  # DFT U_eff (eV) of the reference bands, per ligand class
        "O": (2.5, 5.5),
        "F": (2.5, 5.5),
        "halide": (2.5, 5.5),
        "chalcogenide": (2.5, 5.5),
        "metallic": (2.5, 5.5),
        "other": (2.5, 5.5),
    }
)


def _threads(n=1):
    for v in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[v] = str(n)
    import torch

    torch.set_num_threads(n)


def _index(data):
    return json.loads((Path(data) / "index.json").read_text())


def _model(corrections=None):
    from slakonet import spin_training as st
    from slakonet.optim import default_model

    model = default_model()
    if corrections:
        st.apply_corrections(model, st.SKCorrections.load(corrections))
    return model


# --- dataset -------------------------------------------------------------
def cmd_dataset(args):
    from slakonet import spin_training as st

    batches = sorted(
        {
            d
            for pat in args.sources
            for d in glob.glob(pat)
            if (Path(d) / "manifest.json").exists()
        }
    )
    print(f"{len(batches)} batch directories")
    windows = dict(DEFAULT_U_WINDOWS)
    for item in args.u_window or []:
        cls, rng = item.split("=")
        lo, hi = (float(x) for x in rng.split(":"))
        windows[cls] = (lo, hi)
    names = sorted(
        {
            json.loads((Path(b) / "manifest.json").read_text()).get("compound")
            or Path(b).name.rsplit("_U", 1)[0]
            for b in batches
        }
    )
    holdout = set(args.holdout or [])
    if args.holdout_frac and not holdout:
        rng = random.Random(args.seed)
        holdout = set(
            rng.sample(
                names, max(1, int(round(args.holdout_frac * len(names))))
            )
        )
    index = st.build_dataset(
        batches,
        args.data,
        windows,
        max_configs=args.max_configs,
        holdout=holdout,
    )
    n_hold = sum(r["holdout"] for r in index)
    energies = {r["id"]: st.vasp_energy(r["source"]) for r in index}
    (Path(args.data) / "energies.json").write_text(
        json.dumps(
            {k: v for k, v in energies.items() if v is not None}, indent=1
        )
    )
    print(f"{len(index)} records ({n_hold} held out) -> {args.data}")
    (Path(args.data) / "dataset_settings.json").write_text(
        json.dumps(
            {
                "u_windows": windows,
                "max_configs": args.max_configs,
                "holdout": sorted(holdout),
                "sources": args.sources,
            },
            indent=1,
        )
    )


# --- precompute ----------------------------------------------------------
def _precompute_one(job):
    rid, data, round_name, corrections, tb_u = job
    from slakonet import spin_training as st

    out = Path(data) / "pot" / round_name / f"{rid}.npz"
    if out.exists():
        return rid, "cached"
    global _PRE_MODEL
    if "_PRE_MODEL" not in globals():
        _PRE_MODEL = _model(corrections)
    t = time.time()
    try:
        rec = st.load_record(Path(data) / "structures" / f"{rid}.npz")
        hubbard_U = {s: 0.0 for s in TRANSITION_3D} if tb_u == "zero" else None
        pot = st.precompute_potential(rec, _PRE_MODEL, hubbard_U=hubbard_U)
    except (
        Exception
    ) as err:  # record the failure and go on with the next record
        out.with_suffix(".failed").write_text(traceback.format_exc())
        return rid, f"FAILED {type(err).__name__}: {err}"
    np.savez(out, **{k: v for k, v in pot.items()}, allow_pickle=False)
    return rid, (
        f"{'usable' if pot['usable'] else 'NOT usable'} "
        f"(max |m| error {pot['max_moment_error']:.2f}) {time.time() - t:.0f} s"
    )


def cmd_precompute(args):
    _threads(1)
    (Path(args.data) / "pot" / args.round).mkdir(parents=True, exist_ok=True)
    ids = [r["id"] for r in _index(args.data)]
    if args.task_id is not None:  # Slurm array: every ntasks-th record
        ids = ids[args.task_id :: args.ntasks]
    jobs = [
        (i, args.data, args.round, args.corrections, args.tb_u) for i in ids
    ]
    if args.nproc > 1:
        from multiprocessing import get_context

        with get_context("spawn").Pool(args.nproc) as pool:
            for rid, msg in pool.imap_unordered(_precompute_one, jobs):
                print(f"{rid}: {msg}", flush=True)
    else:
        for job in jobs:
            rid, msg = _precompute_one(job)
            print(f"{rid}: {msg}", flush=True)


# --- training ------------------------------------------------------------
def _usable_records(data, round_name):
    from slakonet import spin_training as st

    recs = []
    for r in _index(data):
        f = Path(data) / "pot" / round_name / f"{r['id']}.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            pot = {k: z[k] for k in z.files}
        if bool(pot["usable"]):
            rec = st.load_record(Path(data) / "structures" / f"{r['id']}.npz")
            rec.update(
                id=r["id"],
                label=r["label"],
                batch=str(Path(r["source"]).parent),
            )
            pot["efermi_tb"] = float(pot["efermi_tb"])
            recs.append((r, rec, pot))
    return recs


def _subset(all_recs, frac, seed):
    """Keep a random fraction of the training compounds (whole compounds, the
    same on every rank); held-out records are always kept."""
    if frac >= 1.0:
        return all_recs
    names = sorted({r["compound"] for r, _, _ in all_recs if not r["holdout"]})
    keep = set(
        random.Random(seed).sample(
            names, max(1, int(round(frac * len(names))))
        )
    )
    return [x for x in all_recs if x[0]["holdout"] or x[0]["compound"] in keep]


def _energies(data):
    f = Path(data) / "energies.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _cost(rec):
    n = len(rec["symbols"]) * 9
    return n**3 * len(rec["kpts"])


def _assign(all_recs, world, rank):
    """Whole DFT batches (needed together for the energy differences) to
    ranks, largest first, balancing the cost."""
    batches = {}
    for item in all_recs:
        batches.setdefault(item[1]["batch"], []).append(item)
    order = sorted(
        batches.values(), key=lambda b: -sum(_cost(x[1]) for x in b)
    )
    loads, mine = [0.0] * world, []
    for b in order:
        r = int(np.argmin(loads))
        loads[r] += sum(_cost(x[1]) for x in b)
        if r == rank:
            mine.extend(b)
    return mine


def _build(args, model, feed, corr, mine, energies):
    """TrainingStructures and EnergyGroups of this rank."""
    from slakonet import spin_training as st

    structs, by_batch = [], {}
    for r, rec, pot in mine:
        s = st.TrainingStructure(
            rec,
            pot,
            model,
            feed,
            corr,
            window_v=args.window_v,
            window_c=args.window_c,
        )
        structs.append((r, s))
        by_batch.setdefault(rec["batch"], []).append((r, s))
    groups = []
    for items in by_batch.values():
        if items[0][0]["compound"] in set(args.exclude_energy or []):
            continue
        g = st.EnergyGroup([s for _, s in items], energies, kT=args.kT)
        if len(g):
            groups.append((items[0][0]["holdout"], g))
    return structs, groups


def _worker(rank, world, args, init_file):
    import torch
    import torch.distributed as dist

    from slakonet import spin_training as st

    _threads(args.threads)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    log = (
        (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    )
    all_recs = _subset(
        _usable_records(args.data, args.round),
        args.train_frac,
        args.subset_seed,
    )
    log(
        f"training compounds: {len({r['compound'] for r, _, _ in all_recs if not r['holdout']})}"
        f" (train-frac {args.train_frac})"
    )
    mine = _assign(all_recs, world, rank)
    energies = _energies(args.data) if args.w_energy > 0 else {}

    model = _model()  # base parameters; corrections act through the feed
    train_el = args.train_elements or sorted(
        set(TRANSITION_3D)
        | {e for _, rec, _ in all_recs for e in rec["symbols"] if e in LIGANDS}
    )
    corr = st.SKCorrections(
        train_el,
        max_log_scale=args.max_log_scale,
        max_decay=args.max_decay,
        max_onsite_ev=args.max_onsite,
        pair_mode=args.pair_mode,
        onsite_elements=args.onsite_elements,
    )
    feed, _ = st.build_feeds(model, corr)
    if args.init:
        prev = st.SKCorrections.load(args.init)
        with torch.no_grad():
            for name in ("u", "v", "w"):
                for k, p in getattr(prev, name).items():
                    if k in getattr(corr, name):
                        getattr(corr, name)[k].copy_(p)
    t = time.time()
    structs, groups = _build(args, model, feed, corr, mine, energies)
    counts = torch.tensor(
        [
            sum(not r["holdout"] for r, _ in structs),
            sum(r["holdout"] for r, _ in structs),
            sum(not h for h, _ in groups),
            sum(h for h, _ in groups),
        ],
        dtype=torch.float64,
    )
    dist.all_reduce(counts)
    n_train, n_hold, g_train, g_hold = (max(int(x), 0) for x in counts)
    log(
        f"{len(all_recs)} usable records ({n_train} train, {n_hold} holdout), "
        f"{g_train + g_hold} energy groups ({g_train} train, {g_hold} holdout), "
        f"w_energy {args.w_energy}, {world} ranks, {len(list(corr.parameters()))} parameters, "
        f"setup {time.time() - t:.0f} s (rank 0)"
    )
    params = [p for _, p in sorted(corr.named_parameters())]
    opt = torch.optim.Adam(params, lr=args.lr)
    best, history = float("inf"), []
    out = Path(args.out)
    hd = args.huber_delta / 1000.0  # meV -> eV
    for epoch in range(args.epochs):
        t = time.time()
        opt.zero_grad()
        stats = torch.zeros(
            4, dtype=torch.float64
        )  # band train/hold, energy train/hold
        for r, s in structs:
            if r["holdout"]:
                with torch.no_grad():
                    stats[1] += float(s.loss(corr, args.w_cond)) / max(
                        n_hold, 1
                    )
            else:
                L = s.loss(corr, args.w_cond) / n_train
                if (
                    L.requires_grad
                ):  # e.g. a metal with --pair-mode ionic: nothing to train
                    L.backward()
                stats[0] += float(L)
        for hold, g in groups:
            if hold:
                with torch.no_grad():
                    stats[3] += float(g.loss(corr, hd)) / max(g_hold, 1)
            else:
                L = g.loss(corr, hd) / g_train
                if L.requires_grad:
                    (args.w_energy * L).backward()
                stats[2] += float(L)
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad)
        dist.all_reduce(stats)
        pen = args.penalty * corr.penalty()
        pen.backward()  # identical on every rank: added after the reduction
        opt.step()
        b_tr, b_ho = float(stats[0]) ** 0.5, float(stats[1]) ** 0.5
        e_tr, e_ho = 1e3 * float(stats[2]) ** 0.5, 1e3 * float(stats[3]) ** 0.5
        history.append(
            {
                "epoch": epoch,
                "train_rmse_eV": b_tr,
                "holdout_rmse_eV": b_ho,
                "train_dE_rmse_meV": e_tr,
                "holdout_dE_rmse_meV": e_ho,
                "penalty": float(pen),
                "seconds": time.time() - t,
            }
        )
        log(
            f"epoch {epoch:4d}  bands train {b_tr:.4f} holdout {b_ho:.4f} eV | "
            f"dE/mag-atom train {e_tr:6.1f} holdout {e_ho:6.1f} meV | "
            f"penalty {float(pen):.4f}  {time.time() - t:.1f} s"
        )
        if rank == 0:
            # model selection on the holdout: bands + energy, same weights as training
            score = (
                (float(stats[1]) + args.w_energy * float(stats[3]))
                if n_hold
                else (float(stats[0]) + args.w_energy * float(stats[2]))
            )
            meta = {
                "epoch": epoch,
                **history[-1],
                "round": args.round,
                "args": vars(args),
            }
            corr.export(out.with_name(out.stem + "_last.pt"), meta)
            if score < best:
                best = score
                corr.export(out, meta)
            out.with_name(out.stem + "_history.json").write_text(
                json.dumps(history, indent=1)
            )
    dist.destroy_process_group()


def cmd_train(args):
    import torch.multiprocessing as mp

    init_file = Path(args.data) / f".dist_init_{os.getpid()}"
    init_file.unlink(missing_ok=True)
    try:
        mp.spawn(
            _worker,
            args=(args.nproc, args, str(init_file.resolve())),
            nprocs=args.nproc,
            join=True,
        )
    finally:
        init_file.unlink(missing_ok=True)


# --- evaluation ----------------------------------------------------------
def cmd_evaluate(args):
    import torch

    from slakonet import spin_training as st

    _threads(args.threads)
    model = _model()
    energies = _energies(args.data)
    recs = _usable_records(args.data, args.round)
    rows, erows = [], []
    for label, path in (("base", None), ("trained", args.corrections)):
        if path:
            loaded = st.SKCorrections.load(path)
            corr = st.SKCorrections(
                loaded.train_elements,
                loaded.r_ref,
                loaded.A,
                loaded.B,
                loaded.D,
                pair_mode=loaded.pair_mode,
                onsite_elements=loaded.onsite_elements,
            )
            feed, _ = st.build_feeds(model, corr)
            with torch.no_grad():
                for name in ("u", "v", "w"):
                    for k, p in getattr(loaded, name).items():
                        if k in getattr(corr, name):
                            getattr(corr, name)[k].copy_(p)
        else:
            corr = st.SKCorrections([])
            feed, _ = st.build_feeds(model, corr)
        by_batch = {}
        for r, rec, pot in recs:
            s = st.TrainingStructure(rec, pot, model, feed, corr)
            by_batch.setdefault(rec["batch"], []).append((r, s))
            with torch.no_grad():
                rmse = float(s.loss(corr)) ** 0.5
            rows.append(
                {
                    "id": r["id"],
                    "compound": r["compound"],
                    "class": r["class"],
                    "holdout": r["holdout"],
                    "model": label,
                    "rmse_eV": rmse,
                }
            )
            print(
                f"{label:8s} {r['id']:45s} {'H' if r['holdout'] else ' '} {rmse:.4f}",
                flush=True,
            )
        for items in by_batch.values():
            g = st.EnergyGroup([s for _, s in items], energies)
            if not len(g):
                continue
            with torch.no_grad():
                dE = g.energies(corr)
            r0 = items[0][0]
            for m, e_tb, e_dft in zip(g.members, dE, g.target):
                erows.append(
                    {
                        "id": m.rec["id"],
                        "compound": r0["compound"],
                        "class": r0["class"],
                        "holdout": r0["holdout"],
                        "model": label,
                        "dE_tb_meV_per_mag": 1e3 * float(e_tb) / g.n_mag,
                        "dE_dft_meV_per_mag": 1e3 * float(e_dft) / g.n_mag,
                    }
                )
    out = Path(args.report or Path(args.data) / f"evaluate_{args.round}.json")
    out.write_text(json.dumps(rows, indent=1))
    out.with_name(out.stem + "_energies.json").write_text(
        json.dumps(erows, indent=1)
    )
    for label in ("base", "trained"):
        for hold in (False, True):
            x = [
                r["rmse_eV"]
                for r in rows
                if r["model"] == label and r["holdout"] == hold
            ]
            y = [
                r["dE_tb_meV_per_mag"] - r["dE_dft_meV_per_mag"]
                for r in erows
                if r["model"] == label and r["holdout"] == hold
            ]
            if x:
                print(
                    f"{label:8s} {'holdout' if hold else 'train':8s} N={len(x):4d} "
                    f"mean RMSE {np.mean(x):.4f} eV | dE N={len(y)} RMSE "
                    f"{np.sqrt(np.mean(np.square(y))) if y else float('nan'):.1f} meV/mag-atom"
                )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="stage", required=True)

    d = sub.add_parser("dataset")
    d.add_argument("--data", required=True)
    d.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="glob(s) of batch directories",
    )
    d.add_argument(
        "--u-window", nargs="*", help="class=lo:hi (eV), e.g. metallic=2.5:3.5"
    )
    d.add_argument("--max-configs", type=int, default=4)
    d.add_argument(
        "--holdout", nargs="*", help="compound names for validation"
    )
    d.add_argument("--holdout-frac", type=float, default=0.15)
    d.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("precompute")
    c.add_argument("--data", required=True)
    c.add_argument("--round", default="r0")
    c.add_argument(
        "--corrections", help="corrections .pt of the previous round"
    )
    c.add_argument("--nproc", type=int, default=1)
    c.add_argument("--task-id", type=int)
    c.add_argument("--ntasks", type=int, default=1)
    c.add_argument(
        "--tb-u",
        choices=["universal", "zero"],
        default="universal",
        help="U of the TB model: universal table, or zero (for U = 0 DFT bands)",
    )

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--round", default="r0")
    t.add_argument("--out", required=True)
    t.add_argument("--init", help="start from these corrections")
    t.add_argument("--nproc", type=int, default=4)
    t.add_argument(
        "--threads", type=int, default=1, help="torch threads per rank"
    )
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--lr", type=float, default=0.02)
    t.add_argument("--penalty", type=float, default=1e-3)
    t.add_argument("--w-cond", type=float, default=0.5)
    t.add_argument("--window-v", type=float, default=8.0)
    t.add_argument("--window-c", type=float, default=3.0)
    t.add_argument("--max-log-scale", type=float, default=0.4)
    t.add_argument("--max-decay", type=float, default=0.3)
    t.add_argument("--max-onsite", type=float, default=2.0)
    t.add_argument("--train-elements", nargs="*")
    t.add_argument(
        "--train-frac",
        type=float,
        default=1.0,
        help="fraction of the training compounds used (learning curve)",
    )
    t.add_argument("--subset-seed", type=int, default=0)
    t.add_argument(
        "--w-energy",
        type=float,
        default=0.0,
        help="weight of the magnetic energy-difference loss (eV^-2 relative "
        "to the band MSE); 0 = bands only. Needs data/energies.json",
    )
    t.add_argument(
        "--kT",
        type=float,
        default=0.05,
        help="smearing of the band energy (eV)",
    )
    t.add_argument(
        "--pair-mode",
        choices=["all", "ionic"],
        default="all",
        help="ionic: train only 3d metal - O/F/halogen/chalcogen pairs",
    )
    t.add_argument(
        "--onsite-elements",
        nargs="*",
        help="elements whose on-site levels may shift (default: all trained)",
    )
    t.add_argument(
        "--huber-delta",
        type=float,
        default=0.0,
        help="Huber threshold of the energy loss, meV per magnetic atom (0 = MSE)",
    )
    t.add_argument(
        "--exclude-energy",
        nargs="*",
        help="compounds left out of the energy loss (bands still used)",
    )

    e = sub.add_parser("evaluate")
    e.add_argument("--data", required=True)
    e.add_argument("--round", default="r0")
    e.add_argument("--corrections", required=True)
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--report")

    args = p.parse_args(argv)
    {
        "dataset": cmd_dataset,
        "precompute": cmd_precompute,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
    }[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
