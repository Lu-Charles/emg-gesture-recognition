# timeplot.py
# Generate a raw EMG time-domain plot using true timestamps

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================
CSV_PATH = "data/raw/S01/20260131/extend_002.csv"
EMG_COLUMN = "ch0"   # change if needed

# =========================
# LOAD DATA
# =========================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Could not find file: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

required_cols = ["t_us", EMG_COLUMN]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

# Extract signal
emg = df[EMG_COLUMN].to_numpy(dtype=np.float64)

# Use real timestamps instead of assuming 500 Hz
t = df["t_us"].to_numpy(dtype=np.float64) / 1e6  # convert microseconds → seconds
t = t - t[0]  # make time relative

# =========================
# Print diagnostics
# =========================
duration = t[-1] - t[0]
estimated_fs = len(emg) / duration if duration > 0 else 0

print(f"Total samples: {len(emg)}")
print(f"Duration (sec): {duration:.3f}")
print(f"Estimated sampling rate: {estimated_fs:.2f} Hz")

# =========================
# OUTPUT DIRECTORY
# =========================
out_dir = "graphs"
os.makedirs(out_dir, exist_ok=True)

# =========================
# PLOT
# =========================
plt.figure(figsize=(10, 4))
plt.plot(t, emg, linewidth=1)

plt.xlabel("Time (seconds)")
plt.ylabel("EMG amplitude (ADC units)")
plt.title("Raw surface EMG signal during hand gesture")

plt.tight_layout()

out_path = os.path.join(out_dir, "raw_emg_example.png")
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved raw EMG plot to: {out_path}")