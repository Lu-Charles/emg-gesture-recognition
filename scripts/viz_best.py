# scripts/viz_best.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import StreamConfig
from src.filters import preprocess_emg


def welch_psd(x: np.ndarray, fs: float, nperseg: int = 2048, noverlap: int | None = None):
    """
    Welch PSD using numpy only.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)

    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("noverlap must be < nperseg")

    if len(x) < nperseg:
        x = np.pad(x, (0, nperseg - len(x)), mode="constant")

    win = np.hanning(nperseg)
    scale = fs * np.sum(win**2)

    psd_accum = None
    nseg = 0

    for start in range(0, len(x) - nperseg + 1, step):
        seg = x[start : start + nperseg] * win
        X = np.fft.rfft(seg)
        Pxx = (np.abs(X) ** 2) / scale
        psd_accum = Pxx if psd_accum is None else (psd_accum + Pxx)
        nseg += 1

    psd = psd_accum / max(1, nseg)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freqs, psd


def load_raw_and_filtered(csv_path: Path, cfg: StreamConfig):
    df = pd.read_csv(csv_path)

    if "t_us" in df.columns:
        t = (df["t_us"].to_numpy(dtype=np.float64) - df["t_us"].iloc[0]) / 1e6
    else:
        t = np.arange(len(df), dtype=np.float64) / cfg.fs_hz

    ch0 = df["ch0"].to_numpy(dtype=np.float64)
    ch1 = df["ch1"].to_numpy(dtype=np.float64)

    ch0_f = preprocess_emg(
        ch0,
        fs=cfg.fs_hz,
        lo=cfg.bandpass_lo,
        hi=cfg.bandpass_hi,
        notch_hz=cfg.notch_hz,
        notch_q=cfg.notch_q,
    )
    ch1_f = preprocess_emg(
        ch1,
        fs=cfg.fs_hz,
        lo=cfg.bandpass_lo,
        hi=cfg.bandpass_hi,
        notch_hz=cfg.notch_hz,
        notch_q=cfg.notch_q,
    )

    return t, ch0, ch1, ch0_f, ch1_f


def plot_time(ax, t, x_raw, x_f, title, ylim=(-200, 800)):
    ax.plot(t, x_raw, label="Raw EMG", linewidth=1.0, alpha=0.85)
    ax.plot(t, x_f,   label="Filtered EMG", linewidth=1.0, alpha=0.85)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ADC counts")
    ax.set_ylim(*ylim)
    ax.legend(loc="upper right", frameon=True)


def plot_psd(ax, x_raw, x_f, cfg: StreamConfig, title):
    freqs_r, psd_r = welch_psd(x_raw, fs=cfg.fs_hz, nperseg=2048)
    freqs_f, psd_f = welch_psd(x_f,   fs=cfg.fs_hz, nperseg=2048)

    eps = 1e-12
    psd_r = np.maximum(psd_r, eps)
    psd_f = np.maximum(psd_f, eps)

    ax.set_facecolor("white")

    ax.semilogy(
        freqs_r, psd_r,
        label="Raw EMG",
        linewidth=2.2,
        alpha=0.55,
        color="#4C72B0"
    )

    ax.semilogy(
        freqs_f, psd_f,
        label="Filtered EMG",
        linewidth=2.2,
        alpha=0.55,
        color="#C44E52"
    )

    # Filter markers
    ax.axvline(cfg.notch_hz, linestyle=":", linewidth=1.2, alpha=0.8)
    ax.axvline(cfg.bandpass_lo, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(cfg.bandpass_hi, linestyle="--", linewidth=1.2, alpha=0.8)

    ax.set_xlim(0, cfg.fs_hz / 2)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (relative)")

    y0, y1 = ax.get_ylim()
    y_txt = y0 * 2.2

    ax.text(cfg.bandpass_lo + 2, y_txt, "20 Hz", rotation=90, va="bottom", ha="left")
    ax.text(cfg.notch_hz + 2,    y_txt, "60 Hz", rotation=90, va="bottom", ha="left")
    ax.text(cfg.bandpass_hi - 2, y_txt, "200 Hz", rotation=90, va="bottom", ha="right")

    ax.legend(loc="upper right", frameon=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = StreamConfig()
    csv_path = Path(args.csv)

    t, ch0, ch1, ch0_f, ch1_f = load_raw_and_filtered(csv_path, cfg)

    keep = t <= args.seconds
    t2 = t[keep]

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    fig.suptitle(
        f"Example EMG trial: {csv_path.name}\n"
        f"Raw signal compared to band-pass filtered EMG (20–200 Hz, 60 Hz noise removed)",
        fontsize=14
    )

    plot_time(axs[0, 0], t2, ch0[keep], ch0_f[keep], "Channel 0 (time)")
    plot_time(axs[0, 1], t2, ch1[keep], ch1_f[keep], "Channel 1 (time)")

    plot_psd(axs[1, 0], ch0, ch0_f, cfg, "Channel 0 (PSD)")
    plot_psd(axs[1, 1], ch1, ch1_f, cfg, "Channel 1 (PSD)")

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out) if args.out else (out_dir / f"{csv_path.stem}_best.png")
    fig.savefig(out_path, dpi=200)
    print(f"[viz] wrote {out_path}")


if __name__ == "__main__":
    main()