"""Small synthetic check of pretraining initialization/order reproducibility."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scripts.train_shared_emg import pretrain


class TinyCorpus:
    def __init__(self):
        self.rows = [{}, {}]
        self.x = np.random.default_rng(9).normal(size=(70, 16, 512)).astype(np.float32)
        self.y = np.repeat([0, 1], 35)

    def window_ids(self, trials):
        return np.concatenate([np.arange(t * 35, (t + 1) * 35) for t in trials])

    def batch(self, ids, mean, scale):
        return self.x[ids], self.y[ids]


class TrainingSeedTests(unittest.TestCase):
    def test_same_seed_repeats_and_other_seed_changes_trained_weights(self):
        torch.set_num_threads(2)
        corpus = TinyCorpus()
        states = []
        for seed in (0, 0, 1):
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                model, _ = pretrain(corpus, None, None, 'cpu', Path(tmp), epochs=2, seed=seed)
                states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        self.assertTrue(all(torch.equal(states[0][k], states[1][k]) for k in states[0]))
        self.assertTrue(any(not torch.equal(states[0][k], states[2][k]) for k in states[0]))


if __name__ == '__main__':
    unittest.main()
