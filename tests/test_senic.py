import unittest
import numpy as np
from src.senic import trial_windows, rotate_features, estimate_rotation, validate_split


class SeNicRisks(unittest.TestCase):
    def test_onset_exclusion_and_window_bounds(self):
        x = np.repeat(np.arange(1500)[:,None], 8, axis=1)
        w, starts = trial_windows(x)
        self.assertEqual(w.shape, (15,50,8))
        self.assertEqual(w[0,0,0],600)
        self.assertEqual(w[-1,-1,0],999)
        self.assertTrue(np.all(starts >= 600))
        with self.assertRaises(ValueError): trial_windows(x[:999])

    def test_rotation_sign_and_feature_groups(self):
        a = np.array([[0,0,1,7,2,0,0,0.]],dtype=float)
        b = np.roll(a,2,axis=1)
        shift,_ = estimate_rotation(a,b)
        self.assertEqual(shift,-2)
        x = np.concatenate([a,2*a,3*a],axis=1)
        np.testing.assert_allclose(rotate_features(rotate_features(x,2),-2),x)
        np.testing.assert_allclose(rotate_features(x,.5),.5*(x+rotate_features(x,1)))
        self.assertEqual(estimate_rotation(np.ones((1,8)),np.ones((1,8)))[0],0)

    def test_leakage_and_final_subject_rejected(self):
        rows=[dict(subject=0,session=0,position=0,repetition=0),
              dict(subject=0,session=0,position=1,repetition=0),
              dict(subject=0,session=0,position=1,repetition=2)]
        validate_split([0],[1],[2],rows)
        with self.assertRaises(ValueError):validate_split([0],[2],[2],rows)
        rows[2]['subject']=2
        with self.assertRaises(ValueError):validate_split([0],[1],[2],rows)


if __name__ == '__main__': unittest.main()
