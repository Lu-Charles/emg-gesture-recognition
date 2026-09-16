"""Plot verified calibrated-class and other-class outcomes for the manuscript."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'research/runs/20260907_csl_components_confirmatory'

def main():
    assert json.loads((RUN/'verification.json').read_text())['passed']
    a=json.loads((RUN/'analysis.json').read_text())
    methods=['frozen','gain_only','spatial_only','spatial_gain','classifier_fine','classifier_fine_gain']
    names=['Frozen','Gain','Spatial','Spatial\n+ gain','Fine-tuning','Fine-tuning\n+ gain']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,(left,right)=plt.subplots(1,2,figsize=(11.5,4.7),gridspec_kw={'width_ratios':[1.65,1]},layout='constrained')
    x=np.arange(len(methods))
    for offset,field,color,label in [(-.19,'seen','#376c99','Calibrated class: held-out trials'),(.19,'unseen','#d88842','Other seven classes')]:
        values=[100*a['methods'][m][field]['mean'] for m in methods]
        bars=left.bar(x+offset,values,.36,color=color,label=label)
        left.bar_label(bars,labels=[f'{v:.1f}' for v in values],padding=3,fontsize=8)
    left.set_xticks(x,names,rotation=20,ha='right');left.set_ylim(0,112);left.set_yticks(range(0,101,20));left.set_ylabel('Trial accuracy (%)');left.set_title('A. Success on one gesture does not establish transfer',loc='left',fontsize=11)
    left.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.25),fontsize=9)
    palette=['#376c99','#d88842','#648b5b','#966197']
    for i,p in enumerate([2,3,4,5]):
        values=[100*a['participants'][m][i]['unseen'] for m in ['frozen','spatial_gain']]
        right.plot([0,1],values,'o-',color=palette[i],lw=1.8,label=f'Participant {p}')
    right.set_xlim(-.2,1.2);right.set_ylim(0,100);right.set_xticks([0,1],['Frozen','Spatial + gain']);right.set_ylabel('Other-gesture trial accuracy (%)');right.set_title('B. All four participant means decline',loc='left',fontsize=11)
    right.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,fontsize=9)
    fig.savefig(RUN/'transfer_gap.png',dpi=200);fig.savefig(RUN/'transfer_gap.svg');plt.close(fig)
    print('Saved transfer_gap.png and transfer_gap.svg')
if __name__=='__main__':main()
