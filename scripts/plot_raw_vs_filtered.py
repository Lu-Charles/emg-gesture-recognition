# scripts/plot_raw_vs_filtered.py
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw csv path")
    ap.add_argument("--proc", required=True, help="processed csv path")
    ap.add_argument("--out", default="outputs/raw_vs_filtered.png")
    ap.add_argument("--seconds", type=float, default=3.0, help="plot first N seconds")
    ap.add_argument("--fs", type=float, default=800.0)
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    proc = pd.read_csv(args.proc)

    n = int(args.seconds * args.fs)
    raw = raw.iloc[:n].copy()
    proc = proc.iloc[:n].copy()

    t = np.arange(len(raw)) / args.fs

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # ch0
    ax[0].plot(t, raw["ch0"].to_numpy(), label="ch0 raw")
    ax[0].plot(t, proc["ch0_f"].to_numpy(), label="ch0 filtered")
    ax[0].set_title("Channel 0: raw vs filtered")
    ax[0].set_ylabel("ADC / filtered units")
    ax[0].legend()

    # ch1
    ax[1].plot(t, raw["ch1"].to_numpy(), label="ch1 raw")
    ax[1].plot(t, proc["ch1_f"].to_numpy(), label="ch1 filtered")
    ax[1].set_title("Channel 1: raw vs filtered")
    ax[1].set_ylabel("ADC / filtered units")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("Saved:", out)

if __name__ == "__main__":
    main()
