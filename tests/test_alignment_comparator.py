import copy
import unittest

import numpy as np
import torch
from torch import nn

from src.alignment_comparator import coral_loss, source_alignment_objective
from scripts.pilot_gesture_coverage import validate_roles


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 4), nn.Tanh())
        self.head = nn.Linear(4, 2)


class AlignmentComparatorTests(unittest.TestCase):
    def test_covariance_formula_and_unequal_batch_sizes(self):
        rng = np.random.default_rng(5)
        s, t = rng.normal(size=(7, 4)), rng.normal(size=(9, 4))
        expected = np.square(np.cov(s, rowvar=False) - np.cov(t, rowvar=False)).sum() / 64
        self.assertAlmostEqual(float(coral_loss(torch.tensor(s), torch.tensor(t))), expected, places=12)

    def test_identical_covariance_translation_and_permutation(self):
        torch.manual_seed(3)
        s = torch.randn(8, 4, dtype=torch.double)
        self.assertLess(float(coral_loss(s, s.flip(0) + 7)), 1e-25)

    def test_finite_nonzero_gradients_both_domains(self):
        torch.manual_seed(4)
        s = torch.randn(6, 4, dtype=torch.double, requires_grad=True)
        t = torch.randn(8, 4, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(coral_loss, (s, t)))
        coral_loss(s, t).backward()
        for x in (s, t):
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(x.grad.abs().sum()), 0)

    def test_source_only_has_no_target_dependency(self):
        torch.manual_seed(0)
        m = SmallModel()
        sx, sy, tx = torch.randn(8, 3), torch.arange(8) % 2, torch.randn(5, 3)
        a = source_alignment_objective(m, sx, sy, tx, 0)[0]
        b = source_alignment_objective(m, sx, sy, tx * 100, 0)[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.backward()
        self.assertGreater(float(m.head.weight.grad.abs().sum()), 0)

    def test_alignment_update_preserves_original_model(self):
        torch.manual_seed(2)
        base = SmallModel()
        state = copy.deepcopy(base.state_dict())
        m = copy.deepcopy(base)
        sx, sy, tx = torch.randn(8, 3), torch.arange(8) % 2, torch.randn(5, 3)
        opt = torch.optim.Adam(m.parameters(), lr=.001)
        source_alignment_objective(m, sx, sy, tx, 1)[0].backward()
        opt.step()
        for k, v in state.items():
            torch.testing.assert_close(base.state_dict()[k], v, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(state[k], v) for k, v in m.state_dict().items()))

    def test_refuses_degenerate_covariance(self):
        with self.assertRaises(ValueError):
            coral_loss(torch.zeros(1, 4), torch.zeros(3, 4))

    def test_whole_trial_overlap_and_final_access_rejected(self):
        rows = [dict(participant=3, group='development', session=s, role=r)
                for s, r in [(1, 'enrollment'), (2, 'calibration'), (2, 'scoring')]]
        validate_roles(rows, [0], [1], [2])
        with self.assertRaises(ValueError):
            validate_roles(rows, [0], [1], [1])
        rows[2]['group'] = 'final'
        with self.assertRaises(ValueError):
            validate_roles(rows, [0], [1], [2])


if __name__ == '__main__':
    unittest.main()
