# scripts/split_labels.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_labels", required=True, help="input labels csv")
    ap.add_argument("--out_calib", required=True, help="output calib labels csv")
    ap.add_argument("--out_eval", required=True, help="output eval labels csv")
    ap.add_argument("--n_per_class", type=int, default=5, help="calib samples per class")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_csv(args.in_labels)
    if not {"file", "label"}.issubset(df.columns):
        raise ValueError(f"Expected columns file,label. Got: {list(df.columns)}")

    rng = np.random.default_rng(args.seed)

    calib_idx = []
    for lab, g in df.groupby("label"):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        take = min(args.n_per_class, len(idx))
        calib_idx.extend(idx[:take].tolist())

    calib = df.loc[sorted(calib_idx)].reset_index(drop=True)
    eval_df = df.drop(index=calib_idx).reset_index(drop=True)

    Path(args.out_calib).parent.mkdir(parents=True, exist_ok=True)
    calib.to_csv(args.out_calib, index=False)
    eval_df.to_csv(args.out_eval, index=False)

    print(f"[split] in={len(df)}  calib={len(calib)}  eval={len(eval_df)}")
    print(calib["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
