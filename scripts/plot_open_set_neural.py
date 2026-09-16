"""Scientific pilot figure; subject points and all prespecified budgets."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.grabmyo_corpus import hash_stream
import json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);run=p.parse_args().run
    assert json.loads((run/'report_validation.json').read_text())['passed']
    data=pd.read_csv(run/'aggregate.csv');per=pd.read_csv(run/'per_participant.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.9),layout='constrained')
    colors={'cnn':'#2069a8','pl':'#bf681e','predin':'#7553a2'};labels={'cnn':'CNN / probability','pl':'PL / prototype (fit limitation)','predin':'PredIN / prototype'}
    for family in colors:
        rejector='probability' if family=='cnn' else 'prototype'
        common=data.family.eq(family)&data.rejector.eq(rejector)&data.update_strategy.eq('pooled_replay')
        for method,style in [('model_only','--'),('both','-')]:
            d=data[common&data.method.eq(method)].sort_values('seconds')
            axes[0].plot(100*d.known_correct_acceptance,100*d.unknown_false_acceptance,style,color=colors[family],label=labels[family] if method=='both' else None)
            for r in d.itertuples():axes[0].scatter(100*r.known_correct_acceptance,100*r.unknown_false_acceptance,marker={35:'^',70:'s',105:'o'}[r.seconds],color=colors[family],s=45)
        d=per[per.family.eq(family)&per.rejector.eq(rejector)&per.method.eq('both')&per.budget.eq(3)&per.update_strategy.eq('pooled_replay')]
        axes[1].scatter(100*d.known_correct_acceptance,100*d.unknown_false_acceptance,color=colors[family],alpha=.45,s=30)
        axes[1].scatter(100*d.known_correct_acceptance.mean(),100*d.unknown_false_acceptance.mean(),color=colors[family],s=140,marker='X',edgecolor='white',linewidth=.8,label=labels[family])
    for ax in axes:
        ax.set_xlim(0,100);ax.set_ylim(0,100);ax.set_xlabel('Correct intended commands accepted (%)');ax.set_ylabel('Unfamiliar gestures accepted as commands (%)');ax.grid(alpha=.18)
    axes[0].set_title('Recording budget tradeoff\nSolid: joint update; dashed: model only',loc='left',fontsize=11)
    axes[0].set_xlim(74,96);axes[0].set_ylim(35,65)
    axes[0].legend(handles=[Line2D([],[],color='#555555',linestyle='none',marker=m,label=f'{b} seconds') for b,m in [(35,'^'),(70,'s'),(105,'o')]],frameon=False,fontsize=9,loc='upper left')
    axes[1].set_xlim(55,100);axes[1].set_ylim(0,85)
    axes[1].set_title('105-second joint update\nDots: eight people; X: equal-person mean',loc='left',fontsize=11)
    axes[1].legend(loc='upper left',frameon=False,fontsize=9)
    fig.suptitle('Open-set EMG development screen — toward the bottom right is better',fontsize=12)
    fig.supxlabel('Fixed 95% calibration-score retention; day1 replay included. One seed, two later sessions; final15 untouched.',fontsize=9)
    fig.savefig(run/'pilot_tradeoff.png',dpi=180);fig.savefig(run/'pilot_tradeoff.pdf');plt.close(fig)
    (run/'code_snapshot'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (run/'figure_artifact_sha256.json').write_text(json.dumps({f:hash_stream(run/f) for f in ['pilot_tradeoff.png','pilot_tradeoff.pdf','code_snapshot/plot_open_set_neural.py']},indent=2)+'\n')


if __name__=='__main__':main()
