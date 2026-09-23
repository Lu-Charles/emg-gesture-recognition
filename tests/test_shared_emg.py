import unittest
from src.grabmyo import allocations
from scripts.train_shared_emg import roles, validate_separation


class SharedAccessTests(unittest.TestCase):
    def test_final_group_and_shared_person_are_rejected(self):
        train = [{"group": "train", "participant": "1"}]
        with self.assertRaises(ValueError):
            validate_separation(train, [{"group": "final", "participant": "2"}])
        with self.assertRaises(ValueError):
            validate_separation(train, [{"group": "development", "participant": "1"}])

    def test_source_calibration_scoring_and_other_session_stay_separate(self):
        groups, rows = allocations()
        p = groups["development"][0]
        source = set(roles(rows, p, 1, "enrollment"))
        self.assertEqual(len(source), 119)
        for session in (2, 3):
            score = set(roles(rows, p, session, "scoring"))
            self.assertEqual(len(score), 68)
            for budget in (1, 2, 3):
                calibration = set(roles(rows, p, session, "calibration", budget))
                self.assertEqual(len(calibration), 17 * budget)
                self.assertFalse((source | calibration) & score)
                self.assertTrue(all(rows[i]["session"] == session for i in calibration | score))


if __name__ == "__main__":
    unittest.main()
