"""Scientific figure for the complete, verified SeNic comparator screen."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();r=a.run
    read=lambda f:json.loads((r/f).read_text())
    assert read('independent_audit.json')['passed']
    s=read('summary.json');c=read('config.json');prior=json.loads((Path(c['classical'])/'summary.json').read_text())
    groups={(g['family'],g['method'],g['trials']):g for g in s['groups']}
    lda={g['trials']:g for g in prior['groups'] if g['feature']=='tdar_rms' and g['method'] in ('frozen','target_only')}
    rng=np.random.default_rng(42);samples=rng.integers(0,6,(10000,6))
    def points(gg):
        x=[g['recorded_seconds'] for g in gg];y=np.array([100*g['accuracy'] for g in gg])
        bounds=np.array([np.quantile(100*np.array([p['accuracy'] for p in g['participants']])[samples].mean(1),[.025,.975]) for g in gg]).T
        return x,y,np.maximum(np.vstack([y-bounds[0],bounds[1]-y]),0)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.6),sharey=True)
    methods=[('fine_target','Fine-tune on target','#1b7193'),('fine_pooled','Fine-tune on source + target','#a05516'),('scratch_target','Train from scratch on target','#765398')]
    for ax,fam,title in zip(axes,['cnn','cnn_roll45'],['Plain CNN','CNN with nominal ±45° augmentation']):
        x,y,e=points([lda[b] for b in [0,7,14]])
        ax.errorbar(x,y,yerr=e,color='#444444',marker='s',ls='--',label='TDAR + RMS LDA',capsize=3,lw=1.6)
        for method,label,color in methods:
            gg=([groups[fam,'frozen',0]] if method!='scratch_target' else [])+[groups[fam,method,b] for b in [7,14]]
            x,y,e=points(gg)
            ax.errorbar(x,y,yerr=e,color=color,marker='o',label=label,capsize=3,lw=1.6)
        ax.set_title(title,pad=12,fontweight='bold');ax.set_xlabel('Average target recording time (seconds)')
        ax.set_xticks([0,52.3,104.8],['0\n0 trials','52.3\n7 trials','104.8\n14 trials'])
        ax.set_ylim(0,100);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].set_ylabel('Gesture classification accuracy (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.055))
    fig.suptitle('Calibration after real electrode rotation',fontsize=16,fontweight='bold',y=.98)
    fig.text(.5,.92,'Six development participants · session 0 · seed 42 · identical scoring trials',ha='center',color='#444444')
    fig.text(.5,.018,'Bars: exploratory 95% participant-bootstrap intervals. Recording cost includes complete trials.',ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.07,right=.98,top=.83,bottom=.28,wspace=.16)
    fig.savefig(r/'calibration_comparison.png',dpi=180);fig.savefig(r/'calibration_comparison.pdf');plt.close(fig)
    print('Saved calibration_comparison.png and .pdf')


if __name__=='__main__':main()
