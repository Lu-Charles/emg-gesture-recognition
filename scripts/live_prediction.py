from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from collections import deque

from src.config import StreamConfig
from src.serial_reader import open_serial, read_csv_stream
from src.filters import preprocess_emg
from src.dataset import build_feature_table
from src.features import extract_window_features
from src.model import load_model
from src.adapt import SelfCalibrator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--channel", choices=["ch0","ch1"], default="ch0")
    ap.add_argument("--self_cal", action="store_true")
    ap.add_argument("--seed_file", default="data/logs/seed_windows.csv")
    args = ap.parse_args()

    cfg = StreamConfig()
    model = load_model(args.model)

    calibrator = SelfCalibrator(max_buf=2000, conf_thresh=0.90) if args.self_cal else None

    # ring buffer for raw samples
    win_n = int(round(cfg.win_sec * cfg.fs_hz))
    hop_n = int(round(cfg.hop_sec * cfg.fs_hz))
    buf = deque(maxlen=win_n * 4)  # keep extra

    ser = open_serial(args.port, cfg.baud)

    print("Streaming... Ctrl+C to stop.")
    try:
        i = 0
        for t_us, ch0, ch1 in read_csv_stream(ser):
            v = ch0 if args.channel == "ch0" else ch1
            buf.append(float(v))
            i += 1
            if len(buf) < win_n:
                continue
            # run every hop
            if i % hop_n != 0:
                continue

            x = np.array(buf)[-win_n:]
            xf = preprocess_emg(x, cfg.fs_hz, cfg.bandpass_lo, cfg.bandpass_hi, cfg.notch_hz, cfg.notch_q)

            ft = build_feature_table(xf, cfg.fs_hz, cfg.win_sec, cfg.win_sec, extract_window_features)  # single window
            X = ft.drop(columns=["start_idx"]).to_numpy(dtype=np.float64)

            proba = model.predict_proba(X)[0]
            cls = model.classes_[int(np.argmax(proba))]
            conf = float(np.max(proba))
            print(f"{cls}  conf={conf:.2f}")

            if calibrator is not None:
                calibrator.ingest(model, X)
                # if you have seed windows (small supervised set), adapt occasionally
                if calibrator.has_enough(300) and (i % (cfg.fs_hz * 5) == 0):
                    if not pd.io.common.file_exists(args.seed_file):
                        # no seed; skip hard refit (keeps safe)
                        continue
                    seed = pd.read_csv(args.seed_file)
                    y_seed = seed["label"].to_numpy()
                    X_seed = seed.drop(columns=["label"]).to_numpy(dtype=np.float64)
                    model = calibrator.adapt(model, X_seed, y_seed)
                    print("[self-cal] model updated")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

if __name__ == "__main__":
    main()