# scripts/make_labels.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

LABELS = {"rest": "rest", "fist": "fist", "extend": "extend"}

def is_shifted(rel_path: Path, shift_key: str) -> bool:
    parts = rel_path.parts
    if not parts:
        return False

    top = parts[0].lower()
    if top.startswith("shifted"):
        return True
    if top.startswith("unshifted"):
        return False

    shift_key_l = shift_key.lower()
    for folder in parts[:-1]:
        if shift_key_l in folder.lower():
            return True
    return False

def write_labels(rows: list[tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        csv.writer(fh).writerows([("file", "label")] + rows)
    print(f"Wrote {out_path} ({len(rows)} trials)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_root", default="data/processed")
    ap.add_argument("--out_dir", default="data/logs")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--shift_key", default="shift")

    # NEW:
    # If set, ONLY include files whose *folder path* contains this substring.
    # Example: --only_tag shift1p5cm
    ap.add_argument("--only_tag", default=None)

    # NEW:
    # Suffix used in output filename so it’s unique.
    # Example: --out_tag shift1p5cm_fix  -> labels_S01_shift1p5cm_fix.csv
    ap.add_argument("--out_tag", default=None)

    args = ap.parse_args()

    subj_dir = Path(args.processed_root) / args.subject
    if not subj_dir.exists():
        raise RuntimeError(f"Missing processed subject dir: {subj_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unshifted_rows: list[tuple[str, str]] = []
    shifted_rows: list[tuple[str, str]] = []

    only_tag_l = args.only_tag.lower() if args.only_tag else None

    for f in sorted(subj_dir.rglob("*.csv")):
        stem = f.stem
        gesture = stem.split("_")[0]
        if gesture not in LABELS:
            continue

        rel = f.relative_to(subj_dir)  # e.g. shifted_fix/20260207_shift1p5cm/rest_001.csv

        # filter to ONLY a specific tag if requested
        if only_tag_l:
            folder_path = "/".join(rel.parts[:-1]).lower()
            if only_tag_l not in folder_path:
                continue

        row = (str(rel), LABELS[gesture])

        if is_shifted(rel, args.shift_key):
            shifted_rows.append(row)
        else:
            unshifted_rows.append(row)

    # output naming
    if args.out_tag:
        # single combined output if you're using only_tag (recommended)
        out_path = out_dir / f"labels_{args.subject}_{args.out_tag}.csv"
        # If you filtered to shifted-only, unshifted_rows will be 0; that's fine.
        # We'll write whichever is non-empty (prefer shifted if exists).
        rows = shifted_rows if len(shifted_rows) else unshifted_rows
        write_labels(rows, out_path)
    else:
        # old behavior (two outputs)
        write_labels(unshifted_rows, out_dir / f"labels_{args.subject}_final_unshifted_fix.csv")
        write_labels(shifted_rows, out_dir / f"labels_{args.subject}_final_shifted_fix.csv")

if __name__ == "__main__":
    main()
