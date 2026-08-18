"""Build a universal model directly from a directory of ``.skf`` files.

``slakonet_v1a`` is an untrained Slater-Koster set -- its H/S and its
repulsive come straight from the reference tables, never retrained
against each other, which is what keeps its equations of state physical.
But it was built over only the 64 elements with Z <= 65, so anything
containing a noble gas, Po, At, Ra, Th or Lu simply cannot be run.

The reference tables themselves reach 75 elements. This script rebuilds
the same kind of model over all of them. Where the two overlap it should
reproduce v1a pair for pair -- ``--check`` verifies that against the
cached v1a, and comes out at exactly 0.

Note what the tables do *not* contain: no true lanthanides (Ce-Yb) and no
actinides beyond Th. La, Lu and Th are all present as spd elements. 75
elements is the maximal periodic-table coverage they allow.

The reference Slater-Koster tables are published at

    https://zenodo.org/records/14289468

Unzip the archive and point ``--skf-dir`` at the directory holding the
``<A>-<B>.skf`` files:

    python slakonet/examples/build_v1a_extended.py \\
        --skf-dir /path/to/complete_set --name slakonet_base75

Building all 5625 pairs takes about 7 minutes and a few GB of RAM.
"""

from __future__ import annotations

import argparse
import os
import re
import time

import torch
from jarvis.core.utils import get_cache_dir

from slakonet.optim import MultiElementSkfParameterOptimizer

PAIR_RE = re.compile(r"^([A-Z][a-z]?)-([A-Z][a-z]?)\.skf$")


def discover_pairs(skf_dir, elements=None):
    """Return (pairs, elements) for every ``A-B.skf`` in `skf_dir`.

    If `elements` is given, pairs are restricted to that set.
    """
    keep = set(elements) if elements else None
    pairs, found = [], set()
    for name in sorted(os.listdir(skf_dir)):
        m = PAIR_RE.match(name)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        if keep is not None and (a not in keep or b not in keep):
            continue
        pairs.append(f"{a}-{b}")
        found.update((a, b))
    return pairs, sorted(found)


def check_against(model, reference_pt, n_probe=25):
    """Compare H/S of shared pairs against a cached ultra-compact model."""
    ref = torch.load(reference_pt, map_location="cpu", weights_only=False)
    ref_params = ref["trained_parameters"]
    shared = [
        p
        for p in model.skf_optimizers
        if f"skf_optimizers.{p}.h_params.0-0" in ref_params
    ]
    print(
        f"\nChecking against {os.path.basename(reference_pt)}: "
        f"{len(shared)} shared pairs, probing {min(n_probe, len(shared))}"
    )
    worst, worst_pair = 0.0, None
    for pair in shared[:n_probe]:
        opt = model.skf_optimizers[pair]
        for group, params in (
            ("h_params", opt.h_params),
            ("s_params", opt.s_params),
        ):
            for key, value in params.items():
                ref_value = ref_params.get(
                    f"skf_optimizers.{pair}.{group}.{key}"
                )
                if ref_value is None or ref_value.shape != value.shape:
                    continue
                d = (value.detach() - ref_value).abs().max().item()
                if d > worst:
                    worst, worst_pair = d, f"{pair}.{group}.{key}"
    print(f"  max |difference| = {worst:.3e}  ({worst_pair})")
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skf-dir",
        required=True,
        help="directory of <A>-<B>.skf files",
    )
    ap.add_argument("--name", default="slakonet_base75")
    ap.add_argument(
        "--elements",
        default=None,
        help="space-separated subset; default is every element found",
    )
    ap.add_argument(
        "--out", default=None, help="output .pt (default: slakonet cache)"
    )
    ap.add_argument(
        "--check",
        default=None,
        help="cached .pt to verify shared pairs against, e.g. slakonet_v1a",
    )
    args = ap.parse_args()

    t0 = time.time()
    elements = args.elements.split() if args.elements else None
    pairs, found = discover_pairs(args.skf_dir, elements)
    if not pairs:
        raise SystemExit(f"No <A>-<B>.skf files found in {args.skf_dir}")
    print(
        f"Found {len(pairs)} SKF pairs over {len(found)} elements:\n"
        f"  {' '.join(found)}\nBuilding ...",
        flush=True,
    )

    model = MultiElementSkfParameterOptimizer(
        args.skf_dir,
        available_skf_pairs=pairs,
        elements_in_system=found,
    )
    print(
        f"Built {len(model.skf_optimizers)} pair optimizers in "
        f"{(time.time() - t0) / 60:.1f} min",
        flush=True,
    )

    if args.check:
        ref = args.check
        if not os.path.exists(ref):
            ref = os.path.join(
                get_cache_dir("slakonet"), args.check, f"{args.check}.pt"
            )
        check_against(model, ref)

    out = args.out or os.path.join(
        get_cache_dir("slakonet"), args.name, f"{args.name}.pt"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    model.float().save_ultra_compact(out)
    print(
        f"\nSaved {out} ({os.path.getsize(out) / 1e6:.0f} MB) in "
        f"{(time.time() - t0) / 60:.1f} min total."
    )


if __name__ == "__main__":
    main()
