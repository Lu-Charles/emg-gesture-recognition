"""Risks: sampling errors, degenerate inputs, duplicate identities and split leakage."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.prepare_icbbe import (ROOT, checked_features, make_splits,
                                   quality_reasons, source_manifest_assignments)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/"research/icbbe_protocol_v1.json").read_text())

    def signal(self, fs=500):
        t = np.arange(500)/fs
        return np.column_stack([np.arange(500)*1e6/fs,
                                np.sin(2*np.pi*35*t), np.cos(2*np.pi*70*t)])

    def test_sampling_and_gap_rejection(self):
        self.assertEqual(quality_reasons(self.signal(), self.cfg["quality"]), [])
        self.assertIn("wrong_median_sampling_interval", quality_reasons(self.signal(800), self.cfg["quality"]))
        gap = self.signal()
        gap[1:, 0] += 120e6
        self.assertIn("timestamp_discontinuity", quality_reasons(gap, self.cfg["quality"]))
        reversed_time = self.signal()
        reversed_time[5, 0] = reversed_time[4, 0]
        self.assertIn("timestamp_discontinuity", quality_reasons(reversed_time, self.cfg["quality"]))

    def test_degenerate_and_nonfinite_rejected(self):
        signal = self.signal()
        signal[:, 1] = 0
        self.assertIn("constant_raw_channel", quality_reasons(signal, self.cfg["quality"]))
        with self.assertRaisesRegex(ValueError, "constant_filtered_window"):
            checked_features(signal, self.cfg)
        signal[0, 1] = np.nan
        self.assertEqual(quality_reasons(signal, self.cfg["quality"]), ["nonfinite_raw"])
        self.assertEqual(quality_reasons(self.signal()[:100], self.cfg["quality"]), ["too_short"])

    def test_features_finite_and_amplitude_retained(self):
        features = checked_features(self.signal(), self.cfg)
        self.assertEqual(features.shape, (16, 18))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue((features[:, [0, 1, 2, 9, 10, 11]] > 0).all())

    def records(self):
        records, assignments = [], {}
        def add(session, label, i):
            ident = f"S01/{session}/{label}_{i:03}.csv"
            r = dict(trial_id=ident, session=session, label=label,
                     signal_sha256=ident, channels_sha256="channels/"+ident, exclusion_reason="")
            records.append(r)
            return ident
        for label in self.cfg["labels"]:
            for i, role in enumerate(["source_train", "source_val", "source_test"]):
                assignments[add("20260131", label, i)] = role
            for session in self.cfg["target_sessions"]:
                for i in range(10):
                    add(session, label, i)
        records[-1]["exclusion_reason"] = "timestamp_discontinuity"
        return records, assignments

    def test_disjoint_fixed_scoring_nested_budgets_and_determinism(self):
        records, assignments = self.records()
        split = make_splits(records, self.cfg, assignments)
        self.assertEqual(split, make_splits(records, self.cfg, assignments))
        shuffled = make_splits(list(reversed(records)), self.cfg, assignments)
        self.assertEqual({r["trial_id"]: (r["role"], r["calibration_rank"]) for r in split},
                         {r["trial_id"]: (r["role"], r["calibration_rank"]) for r in shuffled})
        scoring = {r["trial_id"] for r in split if r["role"] == "target_test"}
        previous = set()
        for budget in self.cfg["calibration_budgets_per_class"]:
            calibration = {r["trial_id"] for r in split if r["role"] == "target_calibration" and r["calibration_rank"] <= budget}
            self.assertFalse(calibration & scoring)
            self.assertTrue(previous <= calibration)
            self.assertEqual(len(calibration), budget*9)
            previous = calibration
        self.assertEqual(split[-1]["role"], "excluded")

    def test_duplicate_identity_conflicting_label_or_coverage_fails(self):
        records, assignments = self.records()
        bad = copy.deepcopy(records)
        bad[1]["channels_sha256"] = bad[0]["channels_sha256"]
        with self.assertRaisesRegex(ValueError, "Duplicate channels_sha256"):
            make_splits(bad, self.cfg, assignments)
        with self.assertRaisesRegex(ValueError, "Duplicate trial_id"):
            make_splits(records+[records[0]], self.cfg, assignments)
        bad = copy.deepcopy(records)
        bad[0]["label"] = "fist"
        with self.assertRaisesRegex(ValueError, "Conflicting trial label"):
            make_splits(bad, self.cfg, assignments)
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            make_splits(records, self.cfg, {})

    def test_inadequate_target_pool_fails(self):
        records, assignments = self.records()
        records = [r for r in records if r["session"] in self.cfg["source_sessions"] or
                   int(Path(r["trial_id"]).stem.rsplit("_", 1)[1]) < 5]
        with self.assertRaisesRegex(ValueError, "Insufficient trials"):
            make_splits(records, self.cfg, assignments)

    def test_source_manifest_cross_partition_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(self.cfg, source_split_root="splits")
            for part in ["train", "val", "test"]:
                folder = Path(tmp)/"splits"/part
                folder.mkdir(parents=True)
                pd.DataFrame([{"file": "unshifted_fix/20260131/fist_001.csv", "label": "fist"}]).to_csv(folder/"labels.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate or conflicting"):
                source_manifest_assignments(Path(tmp), cfg)


if __name__ == "__main__":
    unittest.main()
