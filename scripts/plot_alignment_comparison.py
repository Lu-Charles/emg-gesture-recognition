"""Supplementary participant-level figure from the verified combined table."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--summary',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    data=json.loads(args.summary.read_text());args.out.mkdir(parents=True,exist_ok=True)
    assert data['bridge']['identical_trial_metrics']
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,
        'axes.titlesize':9,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
        'xtick.labelsize':8,'ytick.labelsize':8,'ytick.major.size':0,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white'})
    methods=['target_ce','replay','source_ce']+[f'coral_{w}' for w in ['0.0001','0.001','0.01','0.1','1','10']]
    labels=['Target CE','Replay','Source CE']+['CORAL '+x for x in ['10$^{-4}$','10$^{-3}$','10$^{-2}$','10$^{-1}$','1','10']]
    all_values=[v for r in data['contrasts'] if r['control']=='frozen'
                for v in r['accuracy']['participants']+r['accuracy']['ci95']]
    xmin=5*np.floor(min(min(all_values),-5)/5);xmax=5*np.ceil(max(max(all_values),5)/5)
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.5),sharex=True)
    fig.subplots_adjust(left=.14,right=.98,bottom=.105,top=.93,wspace=.48,hspace=.28)
    plotted=[]
    for ax,(k,step),letter in zip(axes.flat,[(1,25),(2,25),(1,100),(2,100)],'abcd'):
        ax.axvline(0,color='#b8b8b8',linewidth=.7,zorder=0)
        ax.axhline(5.5,color='#dddddd',linewidth=.6,zorder=0)
        ax.spines['left'].set_visible(False)
        for i,method in enumerate(methods):
            r=next(x for x in data['contrasts'] if x['k']==k and x['steps']==step and x['method']==method and x['control']=='frozen')
            z=r['accuracy'];vals=np.array(z['participants']);mean=vals.mean()
            half=t.ppf(.975,7)*vals.std(ddof=1)/np.sqrt(8)
            assert len(vals)==8 and abs(mean-z['mean'])<1e-10
            assert np.allclose([mean-half,mean+half],z['ci95'],rtol=0,atol=1e-10)
            color='#0072B2' if method=='replay' else '#222222';y=8-i
            offsets=np.linspace(-.10,.10,8)
            ax.scatter(vals,y+offsets,s=7,color='#aaaaaa',linewidths=0,zorder=2)
            ax.errorbar(mean,y,xerr=half,fmt='D',color=color,markersize=3.2,
                        capsize=2,elinewidth=.9,mew=.7,zorder=3)
            plotted.append(dict(k=k,steps=step,method=method,values_pp=vals.tolist(),mean_pp=mean,ci95_pp=[mean-half,mean+half]))
        ax.set_yticks(range(8,-1,-1),labels)
        ax.set_ylim(-.65,8.6);ax.set_xlim(xmin-1,xmax+1)
        ax.set_xticks(np.arange(10*np.ceil(xmin/10),xmax+1,10))
        ax.set_title(f'{letter}   {k} gesture'+('s' if k==2 else '')+f' · {step} steps',loc='left',pad=8)
        ax.tick_params(axis='y',pad=6)
    fig.supxlabel('Change in overall accuracy versus frozen (percentage points)',y=.035,fontsize=8)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():continue
        b=text.get_window_extent(renderer)
        assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1,text.get_text()
    for ext in ['png','svg','pdf']:
        fig.savefig(args.out/f'published_objective_comparison.{ext}',dpi=600)
    plt.close(fig)
    report=dict(summary_sha256=hashlib.sha256(args.summary.read_bytes()).hexdigest(),
                participants=data['participants'],conditions=plotted,xlimits=[xmin-1,xmax+1],
                scope='Exploratory development only; all six tested weights retained',text_within_canvas=True)
    (args.out/'figure_data.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.out/'CAPTION.md').write_text('Supplementary development comparison of update objectives. '
        'Panels show one or two prompted gestures at 25 or 100 update steps, always using two recordings. '
        'Gray points are eight participant differences after averaging sessions and the four fixed gesture choices; '
        'diamonds and whiskers show means and descriptive, unadjusted 95% participant t intervals. '
        'Zero denotes each participant’s matched frozen accuracy. CORAL labels give the alignment coefficient. '
        'The three smaller coefficients were added after the initial loss-scale diagnostic; all settings are retained. '
        'CUDA and MPS bridge controls had identical trial confusion matrices. '
        'No final participants, confirmatory superiority test, or new-method claim.\n')
    print(json.dumps({'plotted_conditions':len(plotted),'text_within_canvas':True}))


if __name__=='__main__':main()
