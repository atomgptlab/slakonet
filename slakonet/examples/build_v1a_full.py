"""Build ``slakonet_v1a_full``: v1a's refinements over the full range.

Checked directly against the reference Slater-Koster tables that
``slakonet_v1a`` was built from, v1a is:

* untrained H/S -- bit-identical to the skf files, for every one of its
  4096 pairs. Nothing was retrained.
* the untouched repulsive too, *except* for the 496 pairs listed in the
  file's ``r_spline_data`` block (Si-Si, C-C, Ga-As, Mg-O, O-*, ...),
  whose repulsive splines were refit. Those refits are the reason not
  to simply rebuild v1a from the skf files and be done with it.

What v1a lacks is reach: it stops at the 64 elements with Z <= 65, while
the reference tables cover 75 (adding He, Ne, Ar, Kr, Xe, Rn, Po, At,
Ra, Th and Lu). So build the untrained set over all 75 elements
(``build_v1a_extended.py``) and overlay v1a onto it. Every pair then
carries untrained H/S, and the repulsive is untouched everywhere except
the 496 pairs where v1a's refit takes over.

``slakonet_v1`` is deliberately *not* used as the base. It spans the
same 75 elements, but its H/S were retrained away from the repulsive it
ships with and its equations of state are badly over-stiff -- Al comes
out at 9261 GPa against 69 GPa DFT. Passing ``--base slakonet_v1`` still
works as a fallback if you have no skf tables to hand, at that cost.

The merge is done on the raw ultra-compact dicts rather than through
``load_ultra_compact`` / ``save_ultra_compact`` so that every pair's grid,
cutoff, atomic_data and repulsive spline travel verbatim -- sets may use
different H/S grids (719 vs 358 points) and each pair carries its own, so
nothing has to be resampled.

    python slakonet/examples/build_v1a_extended.py \\
        --skf-dir /path/to/complete_set --name slakonet_base75
    python slakonet/examples/build_v1a_full.py

Writes ``~/.cache/atomgptlab/slakonet/slakonet_v1a_full/`` alongside the
sets it was built from, so ``default_model("slakonet_v1a_full")`` picks it
up with no further setup.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from jarvis.core.utils import get_cache_dir


def _cached(model_name):
    return os.path.join(
        get_cache_dir("slakonet"), model_name, f"{model_name}.pt"
    )


def _pairs_of(compact):
    return list(compact["metadata"]["available_pairs"])


def merge_compact(base, overlay, name):
    """Overlay every pair of `overlay` onto `base`; keep base's extras.

    `base` and `overlay` are ultra-compact dicts as written by
    ``save_ultra_compact``. Returns a new dict; the inputs are untouched.
    """
    base_pairs = _pairs_of(base)
    overlay_pairs = set(_pairs_of(overlay))

    merged_pairs = list(dict.fromkeys(base_pairs + _pairs_of(overlay)))
    from_overlay = [p for p in merged_pairs if p in overlay_pairs]
    from_base = [p for p in merged_pairs if p not in overlay_pairs]

    params, skf_meta, rspl = {}, {}, {}
    for pair in merged_pairs:
        src = overlay if pair in overlay_pairs else base
        prefix = f"skf_optimizers.{pair}."
        for key, value in src["trained_parameters"].items():
            if key.startswith(prefix):
                params[key] = value
        skf_meta[pair] = src["skf_metadata"][pair]
        if pair in src.get("r_spline_data", {}):
            rspl[pair] = src["r_spline_data"][pair]

    elements = sorted(
        set(base["metadata"]["elements_in_system"])
        | set(overlay["metadata"]["elements_in_system"])
    )
    element_pairs = sorted(
        {tuple(p) for p in base["metadata"]["element_pairs"]}
        | {tuple(p) for p in overlay["metadata"]["element_pairs"]}
    )

    merged = {
        "metadata": {
            # Not inherited from either input: both carry the absolute
            # scratch path they happened to be built in, which is
            # meaningless (and stale) once merged.
            "skf_directory": "<merged; see the provenance json>",
            "elements_in_system": elements,
            "element_pairs": [list(p) for p in element_pairs],
            "available_pairs": merged_pairs,
            "class_name": "MultiElementSkfParameterOptimizer",
            "ultra_compact": True,
            "merged_from": {
                "name": name,
                "overlay_pairs": len(from_overlay),
                "base_pairs": len(from_base),
            },
        },
        "trained_parameters": params,
        "skf_metadata": skf_meta,
        "r_spline_data": rspl,
    }
    return merged, from_base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base",
        default="slakonet_base75",
        help="wide set that supplies the missing elements",
    )
    ap.add_argument(
        "--overlay",
        default="slakonet_v1a",
        help="preferred set; wins on every pair it defines",
    )
    ap.add_argument("--name", default="slakonet_v1a_full")
    ap.add_argument(
        "--out",
        default=None,
        help="output .pt (default: the slakonet cache dir)",
    )
    args = ap.parse_args()

    t0 = time.time()
    out = args.out or _cached(args.name)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"Loading base    {args.base} ...", flush=True)
    base = torch.load(
        _cached(args.base), map_location="cpu", weights_only=False
    )
    print(f"Loading overlay {args.overlay} ...", flush=True)
    overlay = torch.load(
        _cached(args.overlay), map_location="cpu", weights_only=False
    )

    merged, from_base = merge_compact(base, overlay, args.name)
    meta = merged["metadata"]
    filler_elements = sorted(
        set(meta["elements_in_system"])
        - set(overlay["metadata"]["elements_in_system"])
    )

    print(
        f"\n{args.overlay}: {len(_pairs_of(overlay))} pairs, "
        f"{len(overlay['metadata']['elements_in_system'])} elements\n"
        f"{args.base}: {len(_pairs_of(base))} pairs, "
        f"{len(base['metadata']['elements_in_system'])} elements\n"
        f"{args.name}: {len(meta['available_pairs'])} pairs, "
        f"{len(meta['elements_in_system'])} elements\n"
        f"  from {args.overlay}: {meta['merged_from']['overlay_pairs']}\n"
        f"  from {args.base}:  {meta['merged_from']['base_pairs']} "
        f"(elements {' '.join(filler_elements)})"
    )

    torch.save(merged, out)
    side = os.path.join(os.path.dirname(out), f"{args.name}.provenance.json")
    with open(side, "w") as f:
        json.dump(
            {
                "name": args.name,
                "overlay": args.overlay,
                "base": args.base,
                "filler_elements": filler_elements,
                "pairs_from_base": sorted(from_base),
            },
            f,
            indent=2,
        )
    print(
        f"\nSaved {out} ({os.path.getsize(out) / 1e6:.0f} MB) "
        f"in {(time.time() - t0) / 60:.1f} min\nProvenance: {side}"
    )


if __name__ == "__main__":
    main()
