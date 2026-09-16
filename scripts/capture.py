# scripts/capture.py
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt

from src.config import StreamConfig
from src.serial_reader import open_serial, read_csv_stream


def next_trial_index(out_dir: Path, gesture: str) -> int:
    # find max fist_###.csv in folder and increment
    mx = 0
    for f in out_dir.glob(f"{gesture}_*.csv"):
        stem = f.stem  # fist_001
        parts = stem.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            mx = max(mx, int(parts[1]))
    return mx + 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--seconds", type=float, default=15.0)

    # clearer than "tag"
    p.add_argument("--gesture", required=True, choices=["rest", "fist", "extend"])

    # subject/session organization
    p.add_argument("--subject", default="S01", help="subject id like S01, S02")
    p.add_argument("--session", default=time.strftime("%Y%m%d"), help="session folder like 20260131")

    p.add_argument("--out_root", default="data/raw")

    # preview defaults ON, BOTH channels
    p.add_argument("--no_preview", action="store_true", help="disable live plot")
    p.add_argument("--plot_window_s", type=float, default=3.0)
    p.add_argument("--timeout_s", type=float, default=2.0)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = StreamConfig()

    out_dir = Path(args.out_root) / args.subject / args.session
    out_dir.mkdir(parents=True, exist_ok=True)

    k = next_trial_index(out_dir, args.gesture)
    out_path = out_dir / f"{args.gesture}_{k:03d}.csv"

    max_lines = int(cfg.fs_hz * args.seconds) + 10

    print(f"[capture] subject={args.subject} session={args.session} gesture={args.gesture}")
    print(f"[capture] port={args.port} baud={cfg.baud} fs={cfg.fs_hz}Hz seconds={args.seconds}")
    print(f"[capture] saving -> {out_path}")

    ser = open_serial(args.port, cfg.baud, timeout=1.0)

    do_preview = not args.no_preview
    win_n = max(50, int(args.plot_window_s * cfg.fs_hz))

    buf_t = deque(maxlen=win_n)
    buf0 = deque(maxlen=win_n)
    buf1 = deque(maxlen=win_n)

    if do_preview:
        plt.ion()
        fig, ax = plt.subplots()
        (line0,) = ax.plot([], [], label="ch0 (FDS/flexor or whatever you wired)")
        (line1,) = ax.plot([], [], label="ch1 (EDC/extensor or whatever you wired)")
        ax.set_title(f"Live EMG — {args.subject} {args.gesture} #{k:03d}")
        ax.set_xlabel("time (s, relative)")
        ax.set_ylabel("ADC counts")
        ax.legend(loc="upper right")
        status = ax.text(0.02, 0.95, "Waiting for data...", transform=ax.transAxes, va="top")
        fig.canvas.manager.set_window_title("EMG Live Preview")
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.flush_events()

    start = time.time()
    got_any = False
    n = 0

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_us", "ch0", "ch1"])

        for t_us, ch0, ch1 in read_csv_stream(ser, max_lines=max_lines):
            now = time.time()
            got_any = True
            n += 1
            w.writerow([t_us, ch0, ch1])

            if n % max(1, int(cfg.fs_hz * 0.5)) == 0:
                print(f"[capture] {n} samples ({(now-start):.1f}s)")

            if do_preview and (n % max(1, int(cfg.fs_hz / 20))) == 0:
                t0 = (t_us / 1e6)
                buf_t.append(t0)
                buf0.append(ch0)
                buf1.append(ch1)

                if len(buf_t) > 5:
                    t_rel = [x - buf_t[0] for x in buf_t]
                    line0.set_data(t_rel, list(buf0))
                    line1.set_data(t_rel, list(buf1))
                    status.set_text(f"samples={n}")

                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                plt.pause(0.001)

            if (now - start) >= args.seconds:
                break

    ser.close()

    if not got_any:
        raise RuntimeError(
            f"No serial data received within {args.timeout_s}s.\n"
            f"Check: Arduino streaming, correct port, correct baud, nothing else using port."
        )

    print(f"[capture] DONE: wrote {n} samples -> {out_path}")


if __name__ == "__main__":
    main()
