# scripts/run_all_subjects.py
from __future__ import annotations
import argparse
import re
import subprocess
from pathlib import Path

def run(cmd: list[str]):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def subj_id_from_labels(path: Path) -> str:
    # labels_S02.csv -> S02
    m = re.search(r"labels_(S\d+)\.csv$", path.name)
    if not m:
        raise ValueError(f"Bad labels filename: {path.name} (expected labels_S##.csv)")
    return m.group(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", default="data/logs", help="where labels_S##.csv live")
    ap.add_argument("--data_dir", default="data/processed", help="processed CSV directory")
    ap.add_argument("--splits_dir", default="data/splits", help="output splits root")
    ap.add_argument("--models_dir", default="data/models", help="output models root")
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    label_files = sorted(labels_dir.glob("labels_S*.csv"))
    if not label_files:
        raise SystemExit(f"No labels found at {labels_dir}/labels_S*.csv")

    subjects = [subj_id_from_labels(p) for p in label_files]
    label_map = {subj_id_from_labels(p): p for p in label_files}

    print("Found subjects:", ", ".join(subjects))

    for train_subj in subjects:
        print("\n" + "=" * 70)
        print(f"TRAIN SUBJECT: {train_subj}")
        print("=" * 70)

        # 1) Split
        out_split = Path(args.splits_dir) / train_subj
        run([
            "python", "-m", "scripts.split_by_file",
            "--in_labels", str(label_map[train_subj]),
            "--out_dir", str(out_split),
            "--val_size", str(args.val_size),
            "--test_size", str(args.test_size),
            "--seed", str(args.seed),
        ])

        # 2) Train on train split
        model_path = Path(args.models_dir) / f"rf_{train_subj}.joblib"
        run([
            "python", "-m", "scripts.train",
            "--data_dir", args.data_dir,
            "--label_map", str(out_split / "train" / "labels.csv"),
            "--out_model", str(model_path),
        ])

        # 3) Eval val + test (within-subject)
        run([
            "python", "-m", "scripts.evaluate",
            "--label_map", str(out_split / "val" / "labels.csv"),
            "--model", str(model_path),
        ])
        run([
            "python", "-m", "scripts.evaluate",
            "--label_map", str(out_split / "test" / "labels.csv"),
            "--model", str(model_path),
        ])

        # 4) Cross-subject eval: evaluate this model on every OTHER subject’s full labels
        for test_subj in subjects:
            if test_subj == train_subj:
                continue
            print("\n" + "-" * 50)
            print(f"CROSS-SUBJECT: train={train_subj} -> test={test_subj}")
            print("-" * 50)
            run([
                "python", "-m", "scripts.evaluate",
                "--label_map", str(label_map[test_subj]),
                "--model", str(model_path),
            ])

if __name__ == "__main__":
    main()
