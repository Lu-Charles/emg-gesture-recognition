import unittest
import numpy as np
from scripts.rotation_coverage import profile_rotation, selection, synthetic


class RotationCoverage(unittest.TestCase):
    def test_known_shared_gain_rotation(self):
        self.assertEqual(synthetic()["two_template_shift"], 2)

    def test_proportional_templates_ambiguous(self):
        u = np.array([[1, 2, 3, 4, 5, 6, 7, 8.0], [2, 4, 6, 8, 10, 12, 14, 16.0]])
        s, g, c = profile_rotation(u, u, False)
        self.assertEqual(s, 0)
        self.assertLess(c, 1e-25)

    def test_selection_invariant_to_gesture_scale(self):
        rng = np.random.default_rng(33)
        u = np.exp(rng.normal(size=(7, 8)))
        a = selection(u)
        b = selection(u * np.arange(1, 8)[:, None])
        self.assertEqual(a["identifiable"], b["identifiable"])
        self.assertEqual(a["diverse"], b["diverse"])


if __name__ == "__main__":
    unittest.main()
