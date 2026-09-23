import unittest
import numpy as np
from scripts.audit_csl_prediction_bias import diagnose
from scripts.pilot_gesture_coverage import selections, validate_roles


class CoverageDiagnostics(unittest.TestCase):
    def test_collapse_has_perfect_calibrated_recall_but_low_precision(self):
        y = np.repeat(np.arange(8), 9)
        d = diagnose(y, np.zeros_like(y), 0)
        self.assertEqual(d["calibrated_recall"], 1)
        self.assertEqual(d["calibrated_precision"], 1 / 8)
        self.assertEqual(d["other_to_calibrated"], 1)
        self.assertEqual(d["oracle_case_permutation_accuracy"], 1 / 8)

    def test_permutation_is_distinct_from_collapse(self):
        y = np.repeat(np.arange(8), 9)
        d = diagnose(y, (y + 1) % 8, 0)
        self.assertEqual(d["accuracy"], 0)
        self.assertEqual(d["oracle_case_permutation_accuracy"], 1)
        self.assertFalse(d["all_predictions_calibrated"])

    def test_budget_and_nonoverlapping_windows(self):
        for k in [1, 2, 4, 8, 16]:
            for offset in range(16):
                plan = selections(list(range(16)), k, offset)
                self.assertEqual(len(plan), k)
                self.assertEqual(sum(len(indices) for _, indices in plan), 16)
                for _, indices in plan:
                    self.assertEqual(len(indices), len(set(indices)))
                    starts = np.array(sorted(indices)) * 256 + 1024
                    self.assertTrue(np.all(np.diff(starts) >= 512))
                    self.assertLessEqual(starts[-1] + 512, 9216)

    def test_score_leakage_and_final_participant_blocked(self):
        rows = [
            dict(participant=3, group="development", session=s, role=r)
            for s, r in [(1, "enrollment"), (2, "calibration"), (2, "scoring")]
        ]
        validate_roles(rows, [0], [1], [2])
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_roles(rows, [0], [1, 2], [2])
        rows[1]["group"] = "final"
        with self.assertRaisesRegex(ValueError, "cohort"):
            validate_roles(rows, [0], [1], [2])


if __name__ == "__main__":
    unittest.main()
