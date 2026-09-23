"""Check numerical alignment, source-only fitting, voting and trial-level metrics."""
import copy
import json
import unittest

import numpy as np
import pandas as pd

from scripts.prepare_icbbe import ROOT
from scripts.evaluate_coral import coral_apply
from scripts.run_icbbe import fit_source, fit_alignment, vote, metrics, calibration_rows


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/"research/icbbe_protocol_v1.json").read_text())

    def test_target_to_source_mean_and_covariance(self):
        rng = np.random.default_rng(123)
        z = rng.normal(size=(2000, 3))
        source = z @ np.array([[3., .5, 0.], [0., 1., .3], [.2, 0., 2.]]) + [3, -4, 8]
        target = z @ np.array([[.7, 0., .4], [.4, 4., 0.], [0., .3, 2.]]) + [-8, 3, 1]
        transform, diagnostic = fit_alignment(source, target)
        aligned = coral_apply(target, *transform)
        np.testing.assert_allclose(aligned.mean(0), source.mean(0), atol=1e-10)
        relative_error = np.linalg.norm(np.cov(aligned, rowvar=False)-np.cov(source, rowvar=False))/np.linalg.norm(np.cov(source, rowvar=False))
        self.assertLess(relative_error, 1e-5)
        self.assertTrue(np.isfinite(diagnostic["transform_spectral_norm"]))

    def test_invalid_alignment_aborts_and_degenerate_condition_is_visible(self):
        rng = np.random.default_rng(1)
        source = rng.normal(size=(50, 3))
        for bad in [source[:1], source[:19], np.full((20, 3), np.nan), np.ones((20, 4))]:
            with self.assertRaises(ValueError):
                fit_alignment(source, bad)
        transform, diagnostic = fit_alignment(source, np.ones((20, 3)))
        self.assertGreater(diagnostic["transform_spectral_norm"], 100)
        self.assertTrue(np.isfinite(transform[0]).all())

    def test_scoring_transform_is_independent_of_other_scoring_trials(self):
        rng = np.random.default_rng(9)
        source = rng.normal(size=(80, 3))
        calibration = rng.normal(size=(60, 3))*[2, .5, 3]+5
        transform, _ = fit_alignment(source, calibration)
        before = [x.copy() for x in transform]
        scoring = rng.normal(size=(10, 3))
        expected = coral_apply(scoring, *transform)
        combined = coral_apply(np.vstack([scoring, np.full((10, 3), 1e6)]), *transform)
        np.testing.assert_allclose(combined[:10], expected)
        for a, b in zip(transform, before):
            np.testing.assert_array_equal(a, b)

    def test_held_out_signals_and_labels_cannot_change_source_fit(self):
        rng = np.random.default_rng(42)
        rows, features = [], {}
        for i, label in enumerate(self.cfg["labels"]):
            for role in ["source_train", "source_test", "source_val", "target_calibration", "target_test"]:
                key = f"{role}_{label}"
                rows.append(dict(trial_id=key, feature_key=key, label=label, role=role))
                features[key] = rng.normal(size=(25, 18))+i*3
        m = pd.DataFrame(rows)
        first, xs, fit = fit_source(m, features, self.cfg)
        changed_features = copy.deepcopy(features)
        altered = m.copy()
        for row in m[m.role != "source_train"].itertuples():
            changed_features[row.feature_key][:] = np.nan
        altered.loc[altered.role != "source_train", "label"] = "invalid_test_label"
        second, xs_again, fit_again = fit_source(altered, changed_features, self.cfg)
        np.testing.assert_array_equal(xs, xs_again)
        self.assertEqual(set(fit.trial_id), set(fit_again.trial_id))
        self.assertEqual(first.named_steps["scaler"].n_samples_seen_, 75)
        for a, b in zip(first.named_steps["rf"].estimators_, second.named_steps["rf"].estimators_):
            np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)
            np.testing.assert_array_equal(a.tree_.value, b.tree_.value)

    def test_vote_ties_and_known_trial_metrics(self):
        self.assertEqual(vote(np.array(["rest", "fist"]), self.cfg["labels"]), ("fist", True))
        self.assertEqual(vote(np.array(["rest", "rest", "fist"]), self.cfg["labels"]), ("rest", False))
        for predictions in [np.array([]), np.array(["unknown"])]:
            with self.assertRaises(ValueError):
                vote(predictions, self.cfg["labels"])
        trials = pd.DataFrame(dict(label=["extend", "fist", "rest"], prediction=["extend", "fist", "fist"], vote_tied=[False]*3))
        result = metrics(trials, self.cfg["labels"])
        self.assertEqual(result["correct"], 2)
        self.assertAlmostEqual(result["macro_f1"], 5/9)
        self.assertAlmostEqual(result["balanced_accuracy"], 2/3)

    def test_unplanned_or_incomplete_calibration_budget_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unplanned budget"):
            calibration_rows(pd.DataFrame(), self.cfg, 4)
        m = pd.DataFrame(dict(role=["target_calibration"], calibration_rank=[1], session=[self.cfg["target_sessions"][0]], label=["fist"]))
        with self.assertRaisesRegex(ValueError, "Unbalanced/incomplete"):
            calibration_rows(m, self.cfg, 1)


if __name__ == "__main__":
    unittest.main()
