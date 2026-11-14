"""Matched 2x3 Adam comparison: standard versus constructed initialization."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,LogLocator,NullFormatter,NullLocator
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD55_qi_adam_drift.plot import smooth_error,SMOOTHING,TITLES
from experiments.expD55_qi_adam_drift.run import OUT,write_json
COMPARISON_SMOOTHING={**SMOOTHING,'window_span_steps':150,
                     'sample_offsets_steps':[-75,-50,-25,0,25,50,75]}


def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=OUT/'expanded_lr_1e-12_10k')
 p.add_argument('--max-step',type=int,default=10000);args=p.parse_args();output=args.results.resolve()
 manifest=json.loads((output/'manifest.json').read_text());steps=args.max_step
 if not 0<steps<=manifest['steps']:raise ValueError('Plot window is outside the saved histories')
 rates=manifest['learning_rates'];targets=manifest['targets'];data={};summaries=[];sources={};displayed={}
 color_positions=.08+.92*np.linspace(0.,1.,len(rates))**.65
 colors=dict(zip(rates,plt.colormaps['viridis'](color_positions)))
 for initialization in manifest['rows']:
  for target in targets:
   key=f'{initialization}/{target}';data[key]={}
   for lr in rates:
    path=ROOT/manifest['runs'][key][f'{lr:.0e}'];record=json.loads(path.read_text());h=record['history']
    assert h[0]['step']==0 and h[-1]['step']==manifest['steps']
    assert record['summary']['learning_rate']==lr and record['summary']['target']==target
    sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    h=[r for r in h if r['step']<=steps];x=np.array([r['step'] for r in h]);y=np.array([r['eval_rel_l2'] for r in h]);smoothed=smooth_error(x,y,window_steps=150)
    data[key][lr]=(x,y,smoothed)
    displayed[f'{key}/{lr:.0e}']={'step':x.tolist(),'raw_rel_l2':y.tolist(),'displayed_rel_l2':smoothed.tolist()}
    post=np.array([r['eval_rel_l2'] for r in h if r['step']>=manifest['warmup_steps']])
    summaries.append({'initialization':initialization,**record['summary'],
                      'post_warmup_median_rel_l2':float(np.median(post)) if len(post) else None,
                      'post_warmup_min_logged_rel_l2':float(post.min()) if len(post) else None})
 plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','font.size':7,
                      'mathtext.rm':'DejaVu Sans:bold','mathtext.it':'DejaVu Sans:italic:bold',
                      'mathtext.bf':'DejaVu Sans:bold'})
 suffix='' if steps==manifest['steps'] else f'_{steps}steps'
 for smooth in [True,False]:
  fig,axes=plt.subplots(2,3,figsize=(9.4,5.2),sharex=True,sharey=True)
  fig.subplots_adjust(left=.135,right=.96,bottom=.205,top=.85,hspace=.15,wspace=.17)
  fig.suptitle('Adam machine epsilon failure',x=.5475,y=.95,fontsize=12,fontweight='bold')
  for row,initialization in enumerate(manifest['rows']):
   for col,target in enumerate(targets):
    ax=axes[row,col];key=f'{initialization}/{target}'
    baseline_path=ROOT/manifest['initials'][f'qi/{target}'];baseline=json.loads(baseline_path.read_text())['eval_rel_l2']
    initial=json.loads((ROOT/manifest['initials'][key]).read_text())['eval_rel_l2']
    for lr in rates:
     x,y,smoothed=data[key][lr];assert y[0]==initial
     exponent=int(round(np.log10(lr)))
     ax.plot(x,smoothed if smooth else y,color=colors[lr],lw=1.3,marker=None,label=rf'$\eta=10^{{{exponent}}}$',zorder=2)
    ax.axhline(baseline,color='#333333',lw=1.25,ls='--',label='QUILL',zorder=3)
    ax.axvspan(0,manifest['warmup_steps'],color='#E6E6E6',alpha=.34,lw=0,zorder=0)
    ax.axvspan(manifest['warmup_steps'],manifest['warmup_steps'],edgecolor='#888888',lw=.65,ls=':')
    ax.set(xlim=(0,steps),ylim=(3e-17,10),yscale='log',xscale='linear')
    ticks=np.linspace(0,steps,6);ax.set_xticks(ticks,[f'{v/1000:g}k' if v>=1000 else f'{v:g}' for v in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1.,),numticks=100))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:rf'$10^{{{int(round(np.log10(y)))}}}$' if int(round(np.log10(y)))%4==0 else ''))
    ax.yaxis.set_minor_locator(LogLocator(base=10,subs=(2.,5.),numticks=100));ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which='major',labelsize=7,length=3,width=.85,pad=2,labelleft=True,labelbottom=True)
    ax.tick_params(which='minor',length=1.3,width=.35)
    if col==0:ax.set_ylabel(r'Relative $L^2$ error',fontsize=8,labelpad=1.5)
    if row==1:ax.set_xlabel('Adam step',fontsize=8,labelpad=2)
    for side in ['top','right']:ax.spines[side].set_visible(False)
    for side in ['bottom','left']:ax.spines[side].set_linewidth(1.)
    ax.set_axisbelow(True)
    for value,tick in zip(ax.yaxis.get_majorticklocs(),ax.yaxis.get_major_ticks()):
     tick.gridline.set_color('#D8D8D8');tick.gridline.set_linewidth(.45)
     tick.gridline.set_visible(int(round(np.log10(value)))%2==0)
    label_offset,label_alignment=6,'bottom'
    if row==1:
     label_offset,label_alignment=(2,'bottom') if col==1 else (-2.5,'top')
    elif col==2:
     label_offset,label_alignment=-2.5,'top'
    ax.annotate(f'QI: {baseline:.1e}',xy=(.97,baseline),xycoords=('axes fraction','data'),
                xytext=(0,label_offset),textcoords='offset points',ha='right',va=label_alignment,
                fontsize=6.3,color='#444444',zorder=5)
    assert len(ax.lines)==len(rates)+1 and all(line.get_marker() in (None,'','None') for line in ax.lines)
  for col,title in enumerate(TITLES):
   axes[0,col].set_title(title,pad=7,fontsize=9.2,fontweight='bold',math_fontfamily='custom')
  for row,label in enumerate(['Pytorch\nInitialization','QUILL Solution\nInitialization']):
   pos=axes[row,0].get_position();fig.text(.059,(pos.y0+pos.y1)/2,label,rotation=90,ha='center',va='center',fontsize=10,fontweight='bold',linespacing=1.05)
  handles,labels=axes[0,0].get_legend_handles_labels()
  if len(handles)==12:
   order=[0,6,1,7,2,8,3,9,4,10,5,11]
   handles=[handles[i] for i in order];labels=[labels[i] for i in order]
  fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5475,.055),ncol=6,
             frameon=False,fontsize=7,handlelength=1.6,handletextpad=.45,columnspacing=1.35,labelspacing=.55)
  name='comparison_2x3'+suffix+('' if smooth else '_raw')
  fig.savefig(output/f'{name}.png',dpi=600,bbox_inches='tight',pad_inches=.13)
  fig.savefig(output/f'{name}.pdf',bbox_inches='tight',pad_inches=.13);plt.close(fig)
 with (output/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
 write_json(output/f'displayed_curves{suffix}.json',displayed)
 write_json(output/f'figure_provenance{suffix}.json',{'source_sha256':sources,'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'smoothing':COMPARISON_SMOOTHING,'plotted_steps':steps,'rows':manifest['rows'],'targets':targets,'learning_rates':rates,
    'title':'Adam machine epsilon failure','title_fontsize_pt':12,'legend_position':'bottom',
    'colormap':'viridis','colormap_range':[.08,1.],'colormap_sampling_power':.65,
    'colormap_positions':color_positions.tolist(),'colors':{f'{lr:.0e}':[float(v) for v in colors[lr]] for lr in rates},
    'export_bbox':'tight','export_padding_inches':.13,
    'bottom_row_baseline_annotation_offsets_pt':[-2.5,2,-2.5],
    'top_row_baseline_annotation_offsets_pt':[6,6,-2.5],
    'figsize_inches':[9.4,5.2],'x_axis':'linear','axis_label_fontsize_pt':8,'tick_label_fontsize_pt':7,
    'subplot_margins':{'left':.135,'right':.96,'bottom':.205,'top':.85,'hspace':.15,'wspace':.17},
    'row_title_fontsize_pt':10,'column_title_fontsize_pt':9.2,'column_math_bold':True,'baseline_legend_label':'QUILL'})
 print(output/f'comparison_2x3{suffix}.png')

if __name__=='__main__':main()
