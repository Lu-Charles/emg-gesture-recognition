"""Final evaluation access explicitly tied to the immutable September16 protocol."""
import csv,json
from pathlib import Path
import numpy as np
from src.grabmyo_corpus import WindowCorpus,hash_stream
from scripts.cache_final_coverage import frozen
R=Path(__file__).resolve().parents[1]
class FinalCorpus(WindowCorpus):
 def __init__(self,path,group='final'):
  if group!='final':raise ValueError('Final-only corpus')
  cfg=frozen();self.path=Path(path);self.config=cfg['grabmyo'];summary=json.loads((self.path/'summary.json').read_text())
  for name,h in summary['hashes'].items():
   if hash_stream(self.path/name)!=h:raise ValueError('Changed final cache '+name)
  self.rows=list(csv.DictReader((self.path/'final_manifest.csv').open()));self.signals=np.load(self.path/'final_signals.npy',mmap_mode='r');self.features=np.load(self.path/'final_features.npy',mmap_mode='r')
  assert len(self.rows)==5355 and sorted({int(r['participant']) for r in self.rows})==cfg['grabmyo']['participants']
  assert all(r['group']=='final' for r in self.rows) and len({r['record'] for r in self.rows})==5355
  assert self.signals.shape==(5355,16,10240) and self.features.shape==(5355,35,48)
  self.starts=1024+np.arange(35,dtype=np.int64)*256;self.labels=np.array([int(r['class_index']) for r in self.rows])
def validate_roles(rows,source,cal,score):
 cfg=frozen();a,b,c=map(set,[source,cal,score]);assert not(a&b or a&c or b&c)
 assert len(source)==119 and len(cal)==2 and len(score)==68
 rr=[rows[i] for i in source+cal+score];assert len({int(r['participant']) for r in rr})==1
 assert all(int(r['participant']) in cfg['grabmyo']['participants'] and r['group']=='final' for r in rr)
 assert all(rows[i]['role']=='enrollment' and int(rows[i]['session'])==1 for i in source)
 assert all(rows[i]['role']=='calibration' for i in cal) and all(rows[i]['role']=='scoring' for i in score)
 assert len({int(rows[i]['session']) for i in cal+score})==1
 assert np.all(np.bincount([int(rows[i]['class_index']) for i in score],minlength=17)==4)
def validate_senic(source,cal,score,rows):
 cfg=frozen();a,b,c=map(set,[source,cal,score]);assert not(a&b or a&c or b&c)
 rr=[rows[i] for i in source+cal+score];assert len({r['subject'] for r in rr})==1
 assert all(r['subject'] in cfg['senic']['participants'] and r['session']==0 for r in rr)
 assert len(source)==14 and len(score)==7;assert all(rows[i]['position']==0 and rows[i]['repetition']<2 for i in source)
 assert all(rows[i]['repetition']==2 for i in score) and all(rows[i]['repetition']<2 for i in cal)
 assert len({rows[i]['position'] for i in cal+score})==1
