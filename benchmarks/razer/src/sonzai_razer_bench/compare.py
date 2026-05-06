"""Side-by-side accuracy comparison for two-or-more Razer bench runs.

Usage::

    python -m sonzai_razer_bench.compare \\
        results/sonzai_*.json results/mempalace_*.json \\
        [--names sonzai,mempalace]

Prints a per-category accuracy table for each run, the overall total,
and a winner-per-row marker. Useful to read the head-to-head story at
a glance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Display order for category rows — keep the structured-memory categories
# (inventory, kb-relationship, multi-user, privacy) near the top because
# they're the differentiation story.
_CAT_ORDER = (
    "inventory",
    "kb-relationship",
    "multi-user-disambiguation",
    "privacy-boundary",
    "cross-device-continuity",
    "single-hop",
    "multi-hop",
    "temporal",
    "adversarial",
)


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _short(path: Path) -> str:
    """Default label = filename stem with the timestamp trimmed off."""
    stem = path.stem
    parts = stem.split("_")
    return parts[0] if parts else stem


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sonzai_razer_bench.compare")
    p.add_argument("inputs", type=Path, nargs="+", help="Two or more run JSON paths.")
    p.add_argument(
        "--names", default=None,
        help="Comma-separated labels (defaults to filename stems).",
    )
    args = p.parse_args(argv)
    if len(args.inputs) < 2:
        p.error("at least two inputs required")

    labels = (
        [n.strip() for n in args.names.split(",")] if args.names
        else [_short(pth) for pth in args.inputs]
    )
    if len(labels) != len(args.inputs):
        p.error("--names count must match number of inputs")

    runs = [_load(pth) for pth in args.inputs]
    aggs = [r.get("qa_aggregate", {}) for r in runs]

    # Build the union of categories across all runs.
    all_cats: list[str] = []
    seen: set[str] = set()
    for cat in _CAT_ORDER:
        if any(cat in agg for agg in aggs):
            all_cats.append(cat)
            seen.add(cat)
    for agg in aggs:
        for cat in agg:
            if cat == "TOTAL":
                continue
            if cat not in seen:
                all_cats.append(cat)
                seen.add(cat)

    # Header
    name_col_w = max(len("category"), max((len(c) for c in all_cats), default=0))
    col_w = max(8, max(len(l) for l in labels))

    print()
    print("=== Razer customer-companion — head-to-head ===\n")
    header = "category".ljust(name_col_w + 2)
    for label in labels:
        header += f"{label.rjust(col_w)}  "
    header += "winner"
    print(header)
    print("-" * (len(header)))

    for cat in all_cats:
        row = cat.ljust(name_col_w + 2)
        accs: list[float | None] = []
        for agg in aggs:
            r = agg.get(cat)
            if r and r.get("n"):
                acc = r["accuracy"]
                accs.append(acc)
                cell = f"{int(r['correct'])}/{int(r['n'])} ({acc * 100:.0f}%)"
            else:
                accs.append(None)
                cell = "—"
            row += cell.rjust(col_w) + "  "
        # Winner — highest accuracy, '—' for ties or all-missing
        valid_accs = [a for a in accs if a is not None]
        if not valid_accs:
            winner = "—"
        else:
            best = max(valid_accs)
            winner_idxs = [i for i, a in enumerate(accs) if a == best]
            if len(winner_idxs) == len(valid_accs):
                winner = "tie"
            else:
                winner = labels[winner_idxs[0]]
        row += winner
        print(row)

    print("-" * (len(header)))
    # TOTAL
    row = "TOTAL".ljust(name_col_w + 2)
    totals: list[float | None] = []
    for agg in aggs:
        t = agg.get("TOTAL")
        if t and t.get("n"):
            acc = t["accuracy"]
            totals.append(acc)
            cell = f"{int(t['correct'])}/{int(t['n'])} ({acc * 100:.0f}%)"
        else:
            totals.append(None)
            cell = "—"
        row += cell.rjust(col_w) + "  "
    valid_totals = [a for a in totals if a is not None]
    if valid_totals:
        best = max(valid_totals)
        wn = [i for i, a in enumerate(totals) if a == best]
        if len(wn) == len(valid_totals):
            row += "tie"
        else:
            row += labels[wn[0]]
    print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
