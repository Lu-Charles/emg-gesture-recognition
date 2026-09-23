"""Final evaluation access explicitly tied to the immutable September16 protocol."""

import csv
import json
from pathlib import Path
import numpy as np
from src.grabmyo_corpus import WindowCorpus, hash_stream
from scripts.cache_final_coverage import frozen

R = Path(__file__).resolve().parents[1]


class FinalCorpus(WindowCorpus):
    def __init__(self, path, group="final"):
        if group != "final":
            raise ValueError("Final-only corpus")
        protocol = frozen()
        self.path = Path(path)
        self.config = protocol["grabmyo"]
        summary = json.loads((self.path / "summary.json").read_text())
        for name, expected_hash in summary["hashes"].items():
            if hash_stream(self.path / name) != expected_hash:
                raise ValueError("Changed final cache " + name)
        self.rows = list(csv.DictReader((self.path / "final_manifest.csv").open()))
        self.signals = np.load(self.path / "final_signals.npy", mmap_mode="r")
        self.features = np.load(self.path / "final_features.npy", mmap_mode="r")
        assert (
            len(self.rows) == 5355
            and sorted({int(row["participant"]) for row in self.rows})
            == protocol["grabmyo"]["participants"]
        )
        assert (
            all(row["group"] == "final" for row in self.rows)
            and len({row["record"] for row in self.rows}) == 5355
        )
        assert self.signals.shape == (5355, 16, 10240) and self.features.shape == (
            5355,
            35,
            48,
        )
        self.starts = 1024 + np.arange(35, dtype=np.int64) * 256
        self.labels = np.array([int(row["class_index"]) for row in self.rows])


def validate_roles(rows, source, cal, score):
    """Check trial separation, the frozen cohort, and four scoring trials per class."""
    protocol = frozen()
    source_trials, calibration_trials, scoring_trials = map(set, [source, cal, score])
    assert not (
        source_trials & calibration_trials
        or source_trials & scoring_trials
        or calibration_trials & scoring_trials
    )
    # 17 classes: seven enrollment trials and four scoring trials per class.
    assert len(source) == 119 and len(cal) == 2 and len(score) == 68
    used_rows = [rows[i] for i in source + cal + score]
    assert len({int(row["participant"]) for row in used_rows}) == 1
    assert all(
        int(row["participant"]) in protocol["grabmyo"]["participants"]
        and row["group"] == "final"
        for row in used_rows
    )
    assert all(
        rows[i]["role"] == "enrollment" and int(rows[i]["session"]) == 1 for i in source
    )
    assert all(rows[i]["role"] == "calibration" for i in cal) and all(
        rows[i]["role"] == "scoring" for i in score
    )
    assert len({int(rows[i]["session"]) for i in cal + score}) == 1
    assert np.all(
        np.bincount([int(rows[i]["class_index"]) for i in score], minlength=17) == 4
    )


def validate_senic(source, cal, score, rows):
    """Check subject, position, and repetition roles against the frozen protocol."""
    protocol = frozen()
    source_trials, calibration_trials, scoring_trials = map(set, [source, cal, score])
    assert not (
        source_trials & calibration_trials
        or source_trials & scoring_trials
        or calibration_trials & scoring_trials
    )
    used_rows = [rows[i] for i in source + cal + score]
    assert len({row["subject"] for row in used_rows}) == 1
    assert all(
        row["subject"] in protocol["senic"]["participants"] and row["session"] == 0
        for row in used_rows
    )
    assert len(source) == 14 and len(score) == 7
    assert all(rows[i]["position"] == 0 and rows[i]["repetition"] < 2 for i in source)
    assert all(rows[i]["repetition"] == 2 for i in score) and all(
        rows[i]["repetition"] < 2 for i in cal
    )
    assert len({rows[i]["position"] for i in cal + score}) == 1
