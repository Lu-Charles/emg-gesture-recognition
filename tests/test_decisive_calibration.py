import unittest
import numpy as np
import torch
from scripts.decisive_calibration import schedules, fit_head, measures


class DecisiveCalibration(unittest.TestCase):
    def test_matched_rank_recording_budget_and_gesture_balance(self):
        rows = [
            dict(
                participant=3,
                session=2,
                role="calibration",
                class_index=g,
                calibration_rank=r,
            )
            for g in range(17)
            for r in [1, 2, 3]
        ]
        cases = list(schedules(rows, 3, 2))
        self.assertEqual(len(cases), 32)
        for c in cases:
            self.assertEqual(len(set(c["ids"])), 2)
            self.assertEqual({rows[i]["calibration_rank"] for i in c["ids"]}, {1, 2})
            self.assertEqual(len({rows[i]["class_index"] for i in c["ids"]}), c["k"])
            self.assertNotIn(16, c["gestures"])
        for k in [1, 2]:
            counts = np.bincount(
                [g for c in cases if c["k"] == k for g in c["gestures"]], minlength=16
            )
            np.testing.assert_array_equal(counts, np.full(16, k))
        for a, b in zip(cases[::2], cases[1::2]):
            self.assertEqual(a["ids"][0], b["ids"][0])

    def test_base_unchanged_and_seed_reproducibility(self):
        torch.manual_seed(7)
        head = torch.nn.Linear(4, 17)
        state = {k: v.clone() for k, v in head.state_dict().items()}
        sx = torch.randn(34, 4)
        sy = torch.arange(17).repeat(2)
        cx = sx[:2]
        cy = torch.zeros(2, dtype=torch.long)
        for method in ["naive", "replay", "l2"]:
            a, _ = fit_head(head, cx, cy, sx, sy, method, 4, steps=(2,))
            b, _ = fit_head(head, cx, cy, sx, sy, method, 4, steps=(2,))
            for key in state:
                torch.testing.assert_close(
                    head.state_dict()[key], state[key], rtol=0, atol=0
                )
                torch.testing.assert_close(a[2][key], b[2][key], rtol=0, atol=0)

    def test_collapse_metrics(self):
        y = np.repeat(np.arange(17), 4)
        m = measures(np.zeros_like(y), y, [0])
        self.assertEqual(m["calibrated_recall"], 1)
        self.assertEqual(m["other_active_accuracy"], 0)
        self.assertEqual(m["other_into_calibration"], 1)


if __name__ == "__main__":
    unittest.main()
