# scripts/split_by_file.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def stratified_split_files(df: pd.DataFrame, val_size: float, test_size: float, seed: int):
    """
    Stratified split at the FILE level:
      - per label, randomly shuffle files
      - allocate counts to val/test/train
    Guarantees at least 1 file per class in val/test when feasible.
    """
    if not {"file", "label"}.issubset(df.columns):
        raise ValueError("labels csv must have columns: file,label")

    # de-dup just in case
    df = df.drop_duplicates(subset=["file", "label"]).reset_index(drop=True)

    groups = []
    rng = np.random.default_rng(seed)

    for label, sub in df.groupby("label"):
        files = sub["file"].tolist()
        files = list(rng.permutation(files))


        n = len(files)
        # base counts
        n_test = int(round(n * test_size))
        n_val  = int(round(n * val_size))

        # ensure at least 1 in val/test if possible (n>=3 lets you do 1/1/rest)
        if n >= 3:
            n_test = max(n_test, 1)
            n_val  = max(n_val, 1)
        # don't exceed
        if n_test + n_val > n - 1:
            # leave at least 1 for train if possible
            if n >= 3:
                # reduce the larger one first
                while n_test + n_val > n - 1:
                    if n_test >= n_val and n_test > 1:
                        n_test -= 1
                    elif n_val > 1:
                        n_val -= 1
                    else:
                        break
            else:
                # n=1 or 2: can't guarantee; just do best effort
                pass

        test_files = files[:n_test]
        val_files  = files[n_test:n_test + n_val]
        train_files= files[n_test + n_val:]

        groups.append(("train", label, train_files))
        groups.append(("val", label, val_files))
        groups.append(("test", label, test_files))

    # build dfs
    out = {}
    for split in ["train", "val", "test"]:
        rows = []
        for s, label, files in groups:
            if s != split:
                continue
            for f in files:
                rows.append({"file": f, "label": label})
        out[split] = pd.DataFrame(rows).sort_values(["label", "file"]).reset_index(drop=True)

    return out["train"], out["val"], out["test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_labels", required=True, help="Path to labels_baseline.csv")
    ap.add_argument("--out_dir", required=True, help="Output dir, e.g. data/splits")
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.in_labels)
    out_dir = Path(args.out_dir)

    df = pd.read_csv(in_path)

    train_df, val_df, test_df = stratified_split_files(
        df=df,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    # write
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train" / "labels.csv", index=False)
    val_df.to_csv(out_dir / "val" / "labels.csv", index=False)
    test_df.to_csv(out_dir / "test" / "labels.csv", index=False)

    # print summary
    def summarize(name: str, d: pd.DataFrame):
        print(f"\n{name}: {len(d)} files")
        if len(d) == 0:
            return
        print(d["label"].value_counts().to_string())

    print("Wrote:")
    print(out_dir / "train" / "labels.csv")
    print(out_dir / "val" / "labels.csv")
    print(out_dir / "test" / "labels.csv")

    summarize("TRAIN", train_df)
    summarize("VAL", val_df)
    summarize("TEST", test_df)

if __name__ == "__main__":
    main()
