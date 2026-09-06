"""Versioned GRABMyo retrieval, physical signals and trial-first allocations.

This public-data path is separate from the legacy two-channel pipeline.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import wfdb

VERSION = "1.1.0"
BASE_URL = f"https://physionet.org/files/grabmyo/{VERSION}/"
FS = 2048
SAMPLES = 10240
LABELS = (
    "Lateral Prehension", "Thumb Adduction",
    "Thumb and Little Finger Opposition", "Thumb and Index Finger Opposition",
    "Thumb and Index Finger Extension", "Thumb and Little Finger Extension",
    "Index and Middle Finger Extension", "Little Finger Extension",
    "Index Finger Extension", "Thumb Finger Extension", "Wrist Extension",
    "Wrist Flexion", "Forearm Supination", "Forearm Pronation", "Hand Open",
    "Hand Close", "Rest",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checksum_index(root: Path) -> dict[str, str]:
    entries = {}
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name.lstrip("*")] = digest
    return entries


def verify_file(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def fetch(root: Path, relative: str, checksums: dict[str, str]) -> dict:
    """Fetch only a published, checksum-listed file; never replace cached data."""
    if relative not in checksums or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Unlisted or unsafe dataset path: {relative}")
    dest = root / relative
    cached = dest.exists()
    if not cached:
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_suffix(dest.suffix + ".partial")
        try:
            subprocess.run(
                ["/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
                 "--max-time", "60", "--output", str(temporary), BASE_URL + relative],
                check=True, capture_output=True,
            )
            verify_file(temporary, checksums[relative])
            temporary.replace(dest)
        finally:
            temporary.unlink(missing_ok=True)
    verify_file(dest, checksums[relative])
    return {"path": relative, "url": BASE_URL + relative, "sha256": checksums[relative],
            "bytes": dest.stat().st_size, "cached": cached}


def record_path(participant: int, session: int, gesture: int, trial: int) -> str:
    if not (1 <= participant <= 43 and 1 <= session <= 3 and
            1 <= gesture <= 17 and 1 <= trial <= 7):
        raise ValueError("Invalid GRABMyo record identity")
    prefix = f"session{session}_participant{participant}"
    return f"Session{session}/{prefix}/{prefix}_gesture{gesture}_trial{trial}"


def ranked(values, seed: int, context: str):
    return sorted(values, key=lambda v: hashlib.sha256(
        f"{seed}:{context}:{v}".encode()).hexdigest())


def allocations(seed: int = 20260906) -> tuple[dict, list[dict]]:
    """Allocate people and complete trials before any performance inspection."""
    ids = ranked(range(1, 44), seed, "participants")
    groups = {"train": ids[:20], "development": ids[20:28], "final": ids[28:]}
    rows = []
    for group, participants in groups.items():
        for p in sorted(participants):
            for s in range(1, 4):
                for g in range(1, 18):
                    order = ranked(range(1, 8), seed, f"p{p}/s{s}/g{g}")
                    for t in range(1, 8):
                        rank = order.index(t) + 1
                        role = ("representation" if group == "train" else
                                "enrollment" if s == 1 else
                                "calibration" if rank <= 3 else "scoring")
                        rows.append({"participant": p, "group": group, "session": s,
                                     "gesture": g, "class_index": g - 1, "label": LABELS[g - 1],
                                     "trial": t, "role": role,
                                     "calibration_rank": rank if role == "calibration" else 0,
                                     "record": record_path(p, s, g, t)})
    return groups, rows


def read_forearm(root: Path, relative: str, checksums: dict[str, str]) -> tuple[np.ndarray, dict]:
    """Return samples x 16 physical mV channels, preserving amplitude."""
    for suffix in (".hea", ".dat"):
        verify_file(root / (relative + suffix), checksums[relative + suffix])
    rec = wfdb.rdrecord(str(root / relative), physical=True)
    if rec.fs != FS or rec.sig_len != SAMPLES or rec.n_sig != 32:
        raise ValueError("Unexpected sample rate, trial length or channel count")
    if rec.sig_name[:16] != [f"F{i}" for i in range(1, 17)] or rec.units[:16] != ["mV"] * 16:
        raise ValueError("Unexpected forearm channel order or physical units")
    if any(fmt != "16" for fmt in rec.fmt):
        raise ValueError("Unexpected WFDB storage format")
    x = rec.p_signal[:, :16]
    if not np.isfinite(x).all() or np.any(np.ptp(x, axis=0) == 0):
        raise ValueError("Nonfinite or constant forearm channel")
    # Independent format-16 decoding checks channel interleaving and ADC scaling.
    digital = np.fromfile(root / (relative + ".dat"), dtype="<i2")
    if digital.size != SAMPLES * 32:
        raise ValueError("Unexpected binary length")
    digital = digital.reshape(SAMPLES, 32)
    manual = (digital[:, :16].astype(np.float64) - np.asarray(rec.baseline[:16])) / rec.adc_gain[:16]
    error = float(np.max(np.abs(manual - x)))
    if error > 1e-12:
        raise ValueError("Physical-unit decoding mismatch")
    return x, {"fs": FS, "samples": SAMPLES, "channels": rec.sig_name[:16],
               "units": "mV", "manual_decode_max_abs_error": error,
               "rms_mv": np.sqrt(np.mean(x*x, axis=0)).tolist()}


def short_windows(x: np.ndarray, window: int = 512, hop: int = 256,
                  start: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Trial-local observations after fixed 0.5 s onset exclusion, no future filtering.

    The recording hardware already bandpasses at 10–500 Hz. This initial pipeline
    adds no software filter or whole-trial normalization. Every sample in a window
    is available when that window ends. It is an offline held-gesture benchmark.
    """
    if x.ndim != 2 or window <= 0 or hop <= 0 or start < 0 or start + window > len(x):
        raise ValueError("Invalid trial/window geometry")
    starts = np.arange(start, len(x) - window + 1, hop, dtype=np.int64)
    return np.stack([x[s:s + window] for s in starts]), starts


def amplitude_features(windows: np.ndarray) -> np.ndarray:
    """Three stable amplitude-inclusive features per channel for first baselines."""
    return np.concatenate([
        np.mean(np.abs(windows), axis=1),
        np.sqrt(np.mean(windows ** 2, axis=1)),
        np.mean(np.abs(np.diff(windows, axis=1)), axis=1),
    ], axis=1)
