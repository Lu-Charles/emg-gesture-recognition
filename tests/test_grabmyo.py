"""Checks for public-data split leakage, ADC scaling, and future access."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from src.grabmyo import (allocations, record_path, short_windows, amplitude_features,
                         sha256, verify_file, fetch, read_forearm)


class PublicDataTests(unittest.TestCase):
    def test_pilot_fitting_never_includes_scoring_or_other_people(self):
        from scripts.pilot_grabmyo import fitting_rows
        groups, rows = allocations()
        p = groups["development"][0]
        rows = [r for r in rows if r["session"] in (1, 2)]
        for method in ("source_only", "source_plus_calibration", "calibration_only"):
            for budget in (1, 2, 3):
                fit = fitting_rows(rows, p, method, budget)
                self.assertTrue(all(r["participant"] == p for r in fit))
                self.assertTrue(all(r["role"] in ("enrollment", "calibration") for r in fit))
                self.assertTrue(all(r["calibration_rank"] <= budget for r in fit))
                expected = (119 if method == "source_only" else
                            119 + 17 * budget if method == "source_plus_calibration" else 17 * budget)
                self.assertEqual(len(fit), expected)

    def test_participants_and_trial_roles_are_disjoint(self):
        groups, rows = allocations()
        self.assertEqual([len(groups[g]) for g in ("train", "development", "final")], [20, 8, 15])
        self.assertEqual(len(set(sum(groups.values(), []))), 43)
        self.assertEqual(len(rows), 43 * 3 * 17 * 7)
        self.assertEqual(len({r["record"] for r in rows}), len(rows))
        for g in ("development", "final"):
            for p in groups[g]:
                for s in (2, 3):
                    subset = [r for r in rows if r["participant"] == p and r["session"] == s]
                    score = {r["record"] for r in subset if r["role"] == "scoring"}
                    self.assertEqual(len(score), 68)
                    previous = set()
                    for budget in (1, 2, 3):
                        cal = {r["record"] for r in subset if r["role"] == "calibration" and r["calibration_rank"] <= budget}
                        self.assertEqual(len(cal), budget * 17)
                        self.assertTrue(previous <= cal)
                        self.assertFalse(cal & score)
                        previous = cal

    def test_allocations_reproduce(self):
        self.assertEqual(allocations(), allocations())

    def test_invalid_identity_and_download_path(self):
        with self.assertRaises(ValueError):
            record_path(44, 1, 1, 1)
        with self.assertRaises(ValueError):
            fetch(Path("."), "../file", {})

    def test_corrupt_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a"
            path.write_bytes(b"correct")
            checksum = sha256(path)
            path.write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                verify_file(path, checksum)

    def test_past_predictions_do_not_use_future_samples(self):
        x = np.random.default_rng(1).normal(size=(10240, 16))
        w, starts = short_windows(x)
        changed = x.copy()
        changed[2048:] *= 100
        after, _ = short_windows(changed)
        past = starts + 512 <= 2048
        np.testing.assert_array_equal(amplitude_features(w[past]), amplitude_features(after[past]))
        self.assertEqual(w.shape, (35, 512, 16))
        self.assertEqual(starts[-1] + 512, 10240)

    def test_amplitude_retained_and_constant_features_finite(self):
        w = np.ones((2, 512, 16))
        self.assertTrue(np.isfinite(amplitude_features(w * 0)).all())
        np.testing.assert_allclose(amplitude_features(w * 2), 2 * amplitude_features(w))

    def test_decoding_checks_units_scaling_and_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = np.tile(np.arange(10240, dtype=np.int16)[:, None], (1, 32))
            data.astype("<i2").tofile(root / "r.dat")
            (root / "r.hea").write_text("fixture")
            checksums = {name: sha256(root / name) for name in ("r.dat", "r.hea")}
            rec = SimpleNamespace(fs=2048, sig_len=10240, n_sig=32,
                                  sig_name=[f"F{i}" for i in range(1, 17)] + ["unused"] * 16,
                                  units=["mV"] * 32, fmt=["16"] * 32,
                                  baseline=[10] * 32, adc_gain=[100.] * 32,
                                  p_signal=(data.astype(float) - 10) / 100.)
            with patch("src.grabmyo.wfdb.rdrecord", return_value=rec):
                x, info = read_forearm(root, "r", checksums)
                self.assertEqual(x.shape, (10240, 16))
                self.assertEqual(info["manual_decode_max_abs_error"], 0)
                rec.units[0] = "uV"
                with self.assertRaises(ValueError):
                    read_forearm(root, "r", checksums)
                rec.units[0] = "mV"
                rec.p_signal = rec.p_signal * 2
                with self.assertRaises(ValueError):
                    read_forearm(root, "r", checksums)


if __name__ == "__main__":
    unittest.main()
