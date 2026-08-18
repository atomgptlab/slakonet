"""Place a SlaKoNet run alongside the other TB sets ChIPS-TB publishes.

ChIPS-TB ships ``chipstb/results.csv``: band gaps for a range of
tight-binding parameter sets over a shared material list. Joining a
SlaKoNet evaluation to it on the JARVIS id answers the only question
that matters when judging a parameter set -- not "is the gap close?"
but "is it closer than what people already use?".

Reference gaps come from JARVIS-DFT ``mbj_bandgap``.

    python slakonet/examples/chipstb_eval.py --model slakonet_v1a_full \\
        --out mine.csv
    python slakonet/examples/chipstb_compare.py --ours mine.csv

Set labels are read from the CSV, never hardcoded, so the table tracks
whatever ChIPS-TB currently publishes.
"""

from __future__ import annotations

import argparse
import csv
import io
import urllib.request
from collections import defaultdict

import numpy as np
from jarvis.db.figshare import data as jarvis_data

RESULTS_URL = (
    "https://raw.githubusercontent.com/atomgptlab/chipstb/main/"
    "chipstb/results.csv"
)


def load_published(url_or_path):
    """Parse ChIPS-TB results.csv -> {label: {jid: gap}}.

    A few rows in the published file have a dropped comma (two fields
    fused into one). Those are skipped rather than guessed at.
    """
    if url_or_path.startswith("http"):
        raw = urllib.request.urlopen(url_or_path).read().decode("utf-8-sig")
    else:
        raw = open(url_or_path, encoding="utf-8-sig").read()

    out, skipped = defaultdict(dict), []
    for row in csv.reader(io.StringIO(raw)):
        if len(row) < 6 or not row[1].strip().isdigit():
            continue
        jid = f"JVASP-{row[1].strip()}"
        label = row[2].strip()
        try:
            gap = float(row[5])
        except ValueError:
            skipped.append((jid, label))
            continue
        out[label][jid] = gap
    if skipped:
        print(
            f"[!] skipped {len(skipped)} malformed row(s) in the published "
            f"file: {', '.join(f'{j}/{l}' for j, l in skipped)}"
        )
    return out


def load_ours(path):
    with open(path) as f:
        return {
            r["jid"]: float(r["gap"])
            for r in csv.DictReader(f)
            if r.get("gap") not in (None, "", "nan")
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", required=True, help="chipstb_eval CSV")
    ap.add_argument("--label", default="slakonet")
    ap.add_argument("--published", default=RESULTS_URL)
    args = ap.parse_args()

    published = load_published(args.published)
    ours = load_ours(args.ours)
    published[args.label] = ours

    index = {row["jid"]: row for row in jarvis_data("dft_3d")}

    def ref_gap(jid):
        try:
            g = float(index[jid]["mbj_bandgap"])
            return None if np.isnan(g) else g
        except (KeyError, TypeError, ValueError):
            return None

    # Score every set on the materials it actually reports.
    print(
        f"\n{'parameter set':16s} {'n':>4s} {'MAE':>8s} {'RMSE':>8s} "
        f"{'signed':>8s}   (eV vs MBJ)"
    )
    print("-" * 60)
    rows = []
    for label, gaps in published.items():
        errs = [
            g - ref_gap(jid)
            for jid, g in gaps.items()
            if ref_gap(jid) is not None
        ]
        if not errs:
            continue
        e = np.array(errs)
        rows.append(
            (label, len(e), np.abs(e).mean(), np.sqrt((e**2).mean()), e.mean())
        )
    for label, n, mae, rmse, bias in sorted(rows, key=lambda r: r[2]):
        mark = "  <-" if label == args.label else ""
        print(f"{label:16s} {n:4d} {mae:8.3f} {rmse:8.3f} {bias:+8.3f}{mark}")

    # The table above is NOT a ranking: each set is scored on whatever
    # materials it happens to report, and n ranges from 1 to ~16. The only
    # fair comparison is per-set, restricted to the materials that set and
    # ours both cover.
    print(
        "\nHead-to-head (each set vs ours, on the materials they share):\n"
        f"  {'set':14s} {'n':>3s} {'their MAE':>10s} {'our MAE':>9s} "
        f"{'delta':>8s}"
    )
    pairs = []
    for label, gaps in published.items():
        if label == args.label:
            continue
        shared = [j for j in gaps if j in ours and ref_gap(j) is not None]
        if not shared:
            continue
        theirs = np.abs([gaps[j] - ref_gap(j) for j in shared]).mean()
        mine = np.abs([ours[j] - ref_gap(j) for j in shared]).mean()
        pairs.append((label, len(shared), theirs, mine))
    for label, n, theirs, mine in sorted(pairs, key=lambda r: -r[1]):
        verdict = "we win" if mine < theirs else "they win"
        print(
            f"  {label:14s} {n:3d} {theirs:10.3f} {mine:9.3f} "
            f"{mine - theirs:+8.3f}  {verdict}"
        )


if __name__ == "__main__":
    main()
