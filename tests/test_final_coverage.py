"""Check the frozen trial rules with synthetic metadata and an optional real manifest."""

import copy
import csv
import unittest
from pathlib import Path

from scripts.cache_final_coverage import frozen
from scripts.decisive_calibration import schedules
from src.final_coverage import validate_roles


class FinalCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        person = frozen()["grabmyo"]["participants"][0]
        cls.rows = []
        for role, session, repetitions in [("enrollment", 1, 7), ("scoring", 2, 4)]:
            for gesture in range(17):
                for trial in range(repetitions):
                    cls.rows.append(
                        dict(
                            participant=str(person),
                            group="final",
                            session=str(session),
                            role=role,
                            class_index=str(gesture),
                            trial=str(trial),
                        )
                    )
        cls.src = list(range(119))
        cls.score = list(range(119, 187))
        cls.cal = [187, 188]
        for trial in range(2):
            cls.rows.append(
                dict(
                    participant=str(person),
                    group="final",
                    session="2",
                    role="calibration",
                    class_index="0",
                    trial=str(trial),
                )
            )

    def test_valid_frozen_roles(self):
        validate_roles(self.rows, self.src, self.cal, self.score)

    def test_scoring_leakage_rejected(self):
        with self.assertRaises(AssertionError):
            validate_roles(
                self.rows, self.src, [self.score[0], self.cal[1]], self.score
            )

    def test_wrong_cohort_rejected(self):
        rows = copy.deepcopy(self.rows)
        for index in self.src + self.cal + self.score:
            rows[index]["group"] = "development"
        with self.assertRaises(AssertionError):
            validate_roles(rows, self.src, self.cal, self.score)

    def test_scoring_class_imbalance_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[self.score[0]]["class_index"] = "1"
        with self.assertRaises(AssertionError):
            validate_roles(rows, self.src, self.cal, self.score)

    def test_source_session_mismatch_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[self.src[0]]["session"] = "2"
        with self.assertRaises(AssertionError):
            validate_roles(rows, self.src, self.cal, self.score)

    def test_calibration_session_mismatch_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[self.cal[0]]["session"] = "3"
        with self.assertRaises(AssertionError):
            validate_roles(rows, self.src, self.cal, self.score)

    def test_cached_manifest_when_available(self):
        root = Path(__file__).resolve().parents[1]
        manifest = root / "data/public/grabmyo/cache_final_20260916/final_manifest.csv"
        if not manifest.exists():
            self.skipTest("Optional integration check requires the GRABMyo final cache")
        with manifest.open() as stream:
            rows = list(csv.DictReader(stream))
        person = int(rows[0]["participant"])
        source = [
            i
            for i, row in enumerate(rows)
            if int(row["participant"]) == person and row["role"] == "enrollment"
        ]
        scoring = [
            i
            for i, row in enumerate(rows)
            if int(row["participant"]) == person
            and int(row["session"]) == 2
            and row["role"] == "scoring"
        ]
        calibration = next(schedules(rows, person, 2))["ids"]
        validate_roles(rows, source, calibration, scoring)


if __name__ == "__main__":
    unittest.main()
