# scripts/process_raw.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src.filters import preprocess_emg
from src.config import StreamConfig

MIN_SAMPLES = 50


def process_file(in_path: Path, out_path: Path, cfg: StreamConfig) -> bool:
    df = pd.read_csv(in_path)

    if "ch0" not in df.columns:
        print(f"[skip] {in_path} (no ch0)")
        return False

    df_out = df.copy()

    # ---- CH0 ----
    x0 = df["ch0"].to_numpy(dtype=float)
    if len(x0) < MIN_SAMPLES:
        print(f"[skip] {in_path} (too short)")
        return False

    df_out["ch0_f"] = preprocess_emg(
        x0,
        fs=cfg.fs_hz,
        lo=cfg.bandpass_lo,
        hi=cfg.bandpass_hi,
        notch_hz=cfg.notch_hz,
        notch_q=cfg.notch_q,
    )

    # ---- CH1 (optional) ----
    if "ch1" in df.columns:
        x1 = df["ch1"].to_numpy(dtype=float)
        if len(x1) >= MIN_SAMPLES:
            df_out["ch1_f"] = preprocess_emg(
                x1,
                fs=cfg.fs_hz,
                lo=cfg.bandpass_lo,
                hi=cfg.bandpass_hi,
                notch_hz=cfg.notch_hz,
                notch_q=cfg.notch_q,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", default="data/raw")
    ap.add_argument("--out_root", default="data/processed")
    ap.add_argument("--subject", required=True)
    ap.add_argument(
        "--sessions",
        nargs="*",
        help="optional list of session folders (e.g. 20260131 20260201)",
    )
    ap.add_argument("--subset", default="", help="e.g. shifted_fix or unshifted_fix")
    args = ap.parse_args()

    cfg = StreamConfig()

    in_subj = Path(args.in_root) / args.subject
    out_subj = Path(args.out_root) / args.subject
    if args.subset:
        out_subj = out_subj / args.subset

    if not in_subj.exists():
        raise RuntimeError(f"No such subject folder: {in_subj}")

    # ---- choose sessions ----
    if args.sessions:
        session_dirs = [in_subj / s for s in args.sessions]
    else:
        session_dirs = [p for p in in_subj.iterdir() if p.is_dir()]

    files = []
    for sd in session_dirs:
        files.extend(sd.glob("*.csv"))

    if not files:
        raise RuntimeError("No raw CSV files found")

    ok = 0
    for f in sorted(files):
        rel = f.relative_to(in_subj)      # session/file.csv
        out = out_subj / rel
        if process_file(f, out, cfg):
            ok += 1

    print(f"[done] processed {ok}/{len(files)} files → {out_subj}")


if __name__ == "__main__":
    main()
