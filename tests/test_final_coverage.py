import csv,copy,unittest
from pathlib import Path
from src.final_coverage import validate_roles
from scripts.decisive_calibration import schedules
class FinalCoverage(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  root=Path(__file__).resolve().parents[1];cls.rows=list(csv.DictReader((root/'data/public/grabmyo/cache_final_20260916/final_manifest.csv').open()));p=int(cls.rows[0]['participant']);cls.src=[i for i,r in enumerate(cls.rows) if int(r['participant'])==p and r['role']=='enrollment'];cls.score=[i for i,r in enumerate(cls.rows) if int(r['participant'])==p and int(r['session'])==2 and r['role']=='scoring'];cls.cal=next(schedules(cls.rows,p,2))['ids']
 def test_valid_frozen_roles(self):validate_roles(self.rows,self.src,self.cal,self.score)
 def test_scoring_leakage_rejected(self):
  with self.assertRaises(AssertionError):validate_roles(self.rows,self.src,[self.score[0],self.cal[1]],self.score)
 def test_wrong_cohort_rejected(self):
  rows=copy.deepcopy(self.rows)
  for i in self.src+self.cal+self.score:rows[i]['group']='development'
  with self.assertRaises(AssertionError):validate_roles(rows,self.src,self.cal,self.score)
if __name__=='__main__':unittest.main()
