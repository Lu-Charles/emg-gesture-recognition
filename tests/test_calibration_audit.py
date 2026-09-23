import unittest
import numpy as np
from scripts.pilot_calibration_audit import accept_update, validate_roles


class CalibrationAudit(unittest.TestCase):
    def test_ties_retain_frozen(self):
        self.assertFalse(accept_update([1,0],[0,1],1))
        self.assertTrue(accept_update([0,0],[1,0],1))
        self.assertFalse(accept_update([1,1],[1,0],1))

    def test_trial_roles_and_vocabulary(self):
        rows=[dict(subject=0,session=0,position=0,repetition=0,label=0),
              dict(subject=0,session=0,position=1,repetition=0,label=0),
              dict(subject=0,session=0,position=1,repetition=1,label=1),
              dict(subject=0,session=0,position=1,repetition=2,label=2)]
        validate_roles([0],[1],2,[3],rows,True)
        with self.assertRaises(ValueError): validate_roles([0],[1],1,[3],rows,True)
        with self.assertRaises(ValueError): validate_roles([0],[1],3,[3],rows,True)
        rows[2]['label']=0
        with self.assertRaises(ValueError): validate_roles([0],[1],2,[3],rows,True)
        rows[2]['subject']=2
        with self.assertRaises(ValueError): validate_roles([0],[1],2,[3],rows,False)


if __name__=='__main__': unittest.main()
