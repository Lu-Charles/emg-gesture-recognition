"""Checksum-verified public corpus retrieval and bounded-memory window batches."""
from __future__ import annotations
import csv
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import requests
from src.grabmyo import verify_file

S3_URL = "https://physionet-open.s3.amazonaws.com/grabmyo/1.1.0/"
_local = threading.local()


def hash_stream(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_s3(root, relative, checksums):
    if relative not in checksums or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Unlisted or unsafe dataset path")
    target = Path(root) / relative
    cached = target.exists()
    attempts = []
    if not cached:
        if not hasattr(_local, "session"):
            _local.session = requests.Session()
        for attempt in range(1, 4):
            start = time.perf_counter()
            try:
                response = _local.session.get(S3_URL + relative, timeout=(10, 45))
                response.raise_for_status()
                data = response.content
                if hashlib.sha256(data).hexdigest() != checksums[relative]:
                    raise ValueError("Downloaded checksum mismatch")
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(target.suffix + ".partial")
                partial.write_bytes(data)
                partial.replace(target)
                attempts.append({"attempt": attempt, "status": response.status_code,
                                 "seconds": time.perf_counter() - start})
                break
            except requests.RequestException as exc:
                attempts.append({"attempt": attempt, "error": str(exc),
                                 "seconds": time.perf_counter() - start})
                if attempt == 3:
                    raise RuntimeError(json.dumps({"path": relative, "attempts": attempts})) from exc
                time.sleep(attempt)
    verify_file(target, checksums[relative])
    return {"path": relative, "url": S3_URL + relative, "sha256": checksums[relative],
            "bytes": target.stat().st_size, "cached": cached, "attempts": attempts}


def permitted_rows(preparation, groups=("train", "development")):
    if not groups or not set(groups) <= {"train", "development"}:
        raise ValueError("This development pipeline cannot access final participants")
    preparation = Path(preparation)
    summary = json.loads((preparation / "summary.json").read_text())
    for file, key in (("trial_manifest.csv", "manifest_sha256"), ("config.json", "config_sha256")):
        if hash_stream(preparation / file) != summary[key]:
            raise ValueError("Frozen allocation artifact changed")
    with (preparation / "trial_manifest.csv").open() as f:
        return [r for r in csv.DictReader(f) if r["group"] in groups]


class WindowCorpus:
    """Raw trial cache with indexed 512-sample windows, no cross-trial windows."""
    def __init__(self, path, group):
        if group not in ("train", "development"):
            raise ValueError("Final participants are not permitted in development")
        self.path = Path(path)
        self.config = json.loads((self.path / "config.json").read_text())
        completion = json.loads((self.path / "summary.json").read_text())
        with (self.path / f"{group}_manifest.csv").open() as f:
            self.rows = list(csv.DictReader(f))
        self.signals = np.load(self.path / f"{group}_signals.npy", mmap_mode="r")
        self.features = np.load(self.path / f"{group}_features.npy", mmap_mode="r")
        if any(r["group"] != group for r in self.rows) or len({r["record"] for r in self.rows}) != len(self.rows):
            raise ValueError("Wrong or duplicate corpus identities")
        if self.signals.shape != (len(self.rows), 16, 10240) or self.features.shape != (len(self.rows), 35, 48):
            raise ValueError("Incomplete corpus")
        if completion["groups"][group]["trials"] != len(self.rows):
            raise ValueError("Cache completion mismatch")
        self.starts = 1024 + np.arange(35, dtype=np.int64) * 256
        self.labels = np.asarray([int(r["class_index"]) for r in self.rows], dtype=np.int64)

    def window_ids(self, trial_ids):
        trial_ids = np.asarray(trial_ids, dtype=np.int64)
        return (trial_ids[:, None] * 35 + np.arange(35)[None, :]).reshape(-1)

    def batch(self, ids, mean=None, scale=None):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1 or not len(ids) or ids.min() < 0 or ids.max() >= len(self.rows) * 35:
            raise ValueError("Window index outside corpus")
        trials, positions = np.divmod(ids, 35)
        x = np.stack([self.signals[t, :, self.starts[w]:self.starts[w] + 512] for t, w in zip(trials, positions)])
        if mean is not None:
            x = ((x - mean) / scale).astype(np.float32)
        return x, self.labels[trials]


def training_moments(corpus):
    if any(r["group"] != "train" for r in corpus.rows):
        raise ValueError("Shared scaling may only fit representation-training participants")
    total = np.zeros(16, dtype=np.float64)
    squares = np.zeros(16, dtype=np.float64)
    count = 0
    for trial in range(len(corpus.rows)):
        x, _ = corpus.batch(corpus.window_ids([trial]))
        x = x.astype(np.float64)
        total += x.sum(axis=(0, 2))
        squares += (x * x).sum(axis=(0, 2))
        count += x.shape[0] * x.shape[2]
    mean = total / count
    scale = np.sqrt(np.maximum(squares / count - mean * mean, 1e-16))
    return mean.reshape(1, 16, 1), scale.reshape(1, 16, 1), count


def epoch_blocks(trials, seed, epoch, buffer_trials=128):
    """Shuffle whole-trial buffers, then all windows within each buffer exactly once."""
    if trials <= 0 or buffer_trials <= 0:
        raise ValueError("Positive trial/buffer counts required")
    rng = np.random.default_rng(seed + epoch)
    order = rng.permutation(trials)
    for offset in range(0, trials, buffer_trials):
        selected = order[offset:offset + buffer_trials]
        yield selected, rng.permutation(len(selected) * 35)
