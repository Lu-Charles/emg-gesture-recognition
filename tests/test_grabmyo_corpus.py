import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.grabmyo_corpus import WindowCorpus, hash_stream, permitted_rows, training_moments, epoch_blocks


class CorpusRiskTests(unittest.TestCase):
    def test_buffer_shuffle_visits_every_window_once_including_partial_buffer(self):
        visited = []
        for trials, order in epoch_blocks(259, 42, 0):
            self.assertLessEqual(len(trials), 128)
            global_ids = (trials[:, None] * 35 + np.arange(35)[None, :]).reshape(-1)
            visited.extend(global_ids[order])
        np.testing.assert_array_equal(np.sort(visited), np.arange(259 * 35))
        repeated = [((trials[:, None] * 35 + np.arange(35)).reshape(-1))[order]
                    for trials, order in epoch_blocks(259, 42, 0)]
        np.testing.assert_array_equal(visited, np.concatenate(repeated))

    def test_final_group_cannot_be_requested(self):
        with self.assertRaises(ValueError):
            permitted_rows(Path("nonexistent"), ("final",))
        with self.assertRaises(ValueError):
            WindowCorpus(Path("nonexistent"), "final")

    def test_chunk_hash_matches_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input"; p.write_bytes(b"sample" * 100)
            self.assertEqual(hash_stream(p), hashlib.sha256(p.read_bytes()).hexdigest())

    def fixture(self, path):
        (path / "config.json").write_text("{}")
        (path / "summary.json").write_text(json.dumps({"groups": {"train": {"trials": 2}}}))
        (path / "train_manifest.csv").write_text("record,group,class_index\na,train,2\nb,train,9\n")
        data = np.random.default_rng(2).normal(size=(2, 16, 10240)).astype(np.float32)
        data[1] += 50
        np.save(path / "train_signals.npy", data)
        np.save(path / "train_features.npy", np.zeros((2, 35, 48)))
        return data

    def test_window_edges_and_trial_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); original = self.fixture(path)
            corpus = WindowCorpus(path, "train")
            x, y = corpus.batch([0, 34, 35, 69])
            np.testing.assert_array_equal(y, [2, 2, 9, 9])
            for actual, trial, start in zip(x, [0, 0, 1, 1], [1024, 9728, 1024, 9728]):
                np.testing.assert_array_equal(actual, original[trial, :, start:start + 512])
            with self.assertRaises(ValueError):
                corpus.batch([70])

    def test_streaming_statistics_equal_expanded_training_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); self.fixture(path)
            corpus = WindowCorpus(path, "train")
            mean, scale, count = training_moments(corpus)
            x, _ = corpus.batch(np.arange(70)); x = x.astype(np.float64)
            np.testing.assert_allclose(mean, x.mean(axis=(0, 2), keepdims=True), rtol=1e-12)
            np.testing.assert_allclose(scale, x.std(axis=(0, 2), keepdims=True), rtol=1e-12)
            self.assertEqual(count, 70 * 512)
            corpus.rows[1]["group"] = "development"
            with self.assertRaises(ValueError):
                training_moments(corpus)


if __name__ == "__main__":
    unittest.main()
