import unittest
import numpy as np
from scipy.signal import lfilter
from src.open_set_emg import tdar_rms, cutoff, measures, command_curve, validate_allocation, feature_arrays


class OpenSetRisks(unittest.TestCase):
    def test_tuple_trial_indexing_and_label_alignment(self):
        features=np.arange(4*3*2,dtype=float).reshape(4,3,2)
        x,y=feature_arrays(features,np.array([10,11,12,13]),(3,1))
        np.testing.assert_array_equal(x,np.concatenate([features[3],features[1]]))
        np.testing.assert_array_equal(y,[13,13,13,11,11,11])
        features[2]=np.nan
        with self.assertRaises(ValueError): feature_arrays(features,np.arange(4),(2,))

    def test_feature_hand_examples_and_degeneracy(self):
        x = np.array([[[0, 1, -1, 0, 2]], [[0, 0, 0, 0, 0]], [[3, 3, 3, 3, 3]]], dtype=float)
        f = tdar_rms(x)
        np.testing.assert_allclose(f[:, 0], [.8, 0, 3])
        np.testing.assert_equal(f[:, 1], [1, 0, 0])
        np.testing.assert_equal(f[:, 2], [2, 3, 3])
        np.testing.assert_allclose(f[:, 3], [6, 0, 0])
        np.testing.assert_allclose(f[:, 8], [np.sqrt(1.2), 0, 3])
        np.testing.assert_array_equal(f[1, 4:8], 0)
        self.assertTrue(np.isfinite(f).all())

    def test_ar_sign_and_recovery(self):
        rng = np.random.default_rng(483)
        x = lfilter([1], [1, -.65, .2], rng.normal(size=120000))[2000:]
        f = tdar_rms(x.reshape(1, 1, -1))
        np.testing.assert_allclose(f[0, 4:8], [-.65, .2, 0, 0], atol=.012)

    def test_reject_all_does_not_fake_success(self):
        y = np.array([10, 11, 0, 16]); pred = np.array([10, 10, 11, 10]); s = np.ones(4)
        m = measures(y, pred, s, 2)
        self.assertEqual(m['known_correct_acceptance'], 0)
        self.assertEqual(m['unknown_false_acceptance'], 0)
        self.assertEqual(m['known_rejection'], 1)
        self.assertEqual(measures(y, pred, s, 0)['unknown_false_acceptance'], 1)

    def test_ties_and_rest_are_not_unknown_commands(self):
        y = np.array([10, 11, 0, 1, 16]); pred = np.array([10, 11, 16, 10, 16]); s = np.ones(5)
        t, c, u = command_curve(y, pred, s)
        np.testing.assert_array_equal(c, [0, 1]); np.testing.assert_array_equal(u, [0, .5])
        self.assertEqual(cutoff(np.ones(10)), 1)

    def test_leakage_and_budget_guards(self):
        rows = [dict(corpus_index=i, participant='3', session='1', class_index='10') for i in range(4)]
        a = dict(fit=[0], threshold=[1], score=[2], budget=0, target_fit=[])
        validate_allocation(a, rows)
        for change in [dict(threshold=[0]), dict(budget=1)]:
            with self.assertRaises(ValueError): validate_allocation(dict(a, **change), rows)
        rows[0]['class_index'] = '1'
        with self.assertRaises(ValueError): validate_allocation(a, rows)


if __name__ == '__main__': unittest.main()
