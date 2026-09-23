"""SeNic trial parsing and a simple spatial-registration baseline.

No GRABMyo class mapping, sample rate, or normalization is reused.
Registration is an engineering baseline, not a reproduction or new method claim.
"""

import re
from pathlib import Path
import numpy as np
from src.grabmyo import amplitude_features

GESTURES = (
    "eversion",
    "fist",
    "open_hand",
    "pinch_forefinger",
    "pinch_middlefinger",
    "two",
    "varus",
)
DEVELOPMENT = (0, 1, 14, 18, 24, 25)
FS = 200
START, STOP, WINDOW, HOP = 600, 1000, 50, 25


def parse_trial(path):
    path = Path(path)
    match = re.fullmatch(r"emg_p(\d+)_r(\d+)_(.+)\.csv", path.name)
    if not match or match[3] not in GESTURES:
        raise ValueError(f"Unknown trial filename: {path}")
    subject = int(path.parent.parent.name.removeprefix("h"))
    position, repetition = int(match[1]), int(match[2])
    if position not in range(11) or repetition not in range(3):
        raise ValueError("Invalid position or repetition")
    return dict(
        subject=subject,
        session=int(path.parent.name),
        position=position,
        repetition=repetition,
        gesture=match[3],
        label=GESTURES.index(match[3]),
        path=str(path.resolve()),
    )


def trial_windows(signal):
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != 8 or len(signal) < STOP:
        raise ValueError(
            f"Expected >={STOP} samples and eight channels, got {signal.shape}"
        )
    if not np.isfinite(signal).all():
        raise ValueError("Nonfinite raw signal")
    starts = np.arange(START, STOP - WINDOW + 1, HOP)
    return np.stack([signal[s : s + WINDOW] for s in starts]), starts


def amplitude_windows(signal):
    windows, starts = trial_windows(signal)
    return amplitude_features(windows), starts


def rotate_features(features, shift):
    """Linear circular interpolation in channel coordinates, feature-major layout."""
    values = np.asarray(features, dtype=np.float64)
    if values.shape[-1] % 8:
        raise ValueError("Feature layout must contain complete eight-channel groups")
    channel_groups = values.reshape(*values.shape[:-1], -1, 8)
    integer_shift = int(np.floor(shift))
    fraction = shift - integer_shift
    return (
        (1 - fraction) * np.roll(channel_groups, integer_shift, axis=-1)
        + fraction * np.roll(channel_groups, integer_shift + 1, axis=-1)
    ).reshape(values.shape)


def estimate_rotation(source_rms, target_rms):
    """Equal-weight per-gesture cosine matching; no ruler angles or test labels."""
    a, b = np.asarray(source_rms), np.asarray(target_rms)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 8:
        raise ValueError("Matched gesture-by-channel templates required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite template")
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    shifts = sorted(np.arange(-4.0, 4.0, 0.125), key=lambda s: (abs(s), s))
    costs = []
    for shift in shifts:
        z = rotate_features(b, shift)
        z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        costs.append(float(np.mean(np.sum((a - z) ** 2, axis=1))))
    best = int(np.argmin(costs))
    return float(shifts[best]), float(costs[best])


def validate_split(source, calibration, score, rows):
    groups = [set(source), set(calibration), set(score)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Whole-trial leakage")
    if not source or not score:
        raise ValueError("Empty source or score set")
    for i in set.union(*groups):
        if rows[i]["subject"] not in DEVELOPMENT or rows[i]["session"] != 0:
            raise ValueError("Pilot accessed reserved participant/session")
    if len({rows[i]["subject"] for i in set.union(*groups)}) != 1:
        raise ValueError("Mixed participants")
    if any(
        rows[i]["position"] != 0 or rows[i]["repetition"] not in (0, 1) for i in source
    ):
        raise ValueError("Wrong enrollment role")
    if any(rows[i]["repetition"] == 2 for i in calibration):
        raise ValueError("Scoring repetition in calibration")
    if any(rows[i]["repetition"] != 2 for i in score):
        raise ValueError("Wrong scoring repetition")
    if calibration and {rows[i]["position"] for i in calibration} != {
        rows[i]["position"] for i in score
    }:
        raise ValueError("Calibration and scoring positions differ")
