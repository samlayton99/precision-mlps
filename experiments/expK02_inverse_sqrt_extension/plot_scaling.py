"""Standalone 15-block QI scaling figure, 1k--5k neurons."""
import os,json,hashlib
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,NullFormatter,FixedLocator
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_K_practical_recipes/expK02_inverse_sqrt_extension'

def main():
 source=OUT/'scaling_1k_5k.json';rows=json.loads(source.read_text())
 assert all(r['blocks']==15 and r['halo']==32 for r in rows)
 plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','font.size':7,
                      'mathtext.rm':'DejaVu Sans:bold','mathtext.it':'DejaVu Sans:italic:bold',
                      'mathtext.bf':'DejaVu Sans:bold','mathtext.cal':'DejaVu Sans'})
 fig,ax=plt.subplots(figsize=(4.2,3.3));fig.subplots_adjust(left=.17,right=.97,bottom=.22,top=.79)
 fig.suptitle('QUILL approximation',x=.57,y=.945,fontsize=12,fontweight='bold')
 color=plt.colormaps['viridis'](.55)
 visible_indices=[i for i,r in enumerate(rows) if 1000<=r['width']<=5000]
 ax.semilogy([r['width'] for r in rows],[r['relative_l2'] for r in rows],
             color=color,lw=1.3,label='QUILL',marker='o',markersize=3.8,
             markerfacecolor='white',markeredgecolor=color,markeredgewidth=.9,
             markevery=visible_indices)
 ax.set(xscale='log',xlim=(1000,5000),ylim=(1e-15,1e4),xlabel='Hidden width (neurons)',ylabel=r'Relative $L^2$ error')
 ax.set_xticks([1000,2000,4000],['1k','2k','4k'])
 ax.xaxis.set_minor_locator(FixedLocator([1500,3000,5000]))
 ax.xaxis.set_minor_formatter(NullFormatter())
 ax.set_yticks([1e3,1,1e-3,1e-6,1e-9,1e-12,1e-15])
 ax.yaxis.set_minor_locator(LogLocator(base=10,subs=(2.,5.),numticks=100));ax.yaxis.set_minor_formatter(NullFormatter())
 ax.set_title(r'$f(x)=1/\sqrt{x},\quad x\in[10^{-4},1]$',
              fontweight='bold',math_fontfamily='custom',fontsize=9.2,pad=8)
 fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.57,.035),
            frameon=False,fontsize=7,handlelength=1.8,handletextpad=.5)
 last_visible=max((r for r in rows if r['width']<=5000),key=lambda r:r['width'])
 ax.annotate(r'$\approx 5\times10^{-14}$',xy=(last_visible['width'],last_visible['relative_l2']),
             xytext=(0,5),textcoords='offset points',ha='right',va='bottom',fontsize=7,color='#444444')
 ax.xaxis.label.set_fontsize(8);ax.yaxis.label.set_fontsize(8)
 ax.xaxis.labelpad=2;ax.yaxis.labelpad=1.5
 for side in ['top','right']:ax.spines[side].set_visible(False)
 for side in ['left','bottom']:ax.spines[side].set_linewidth(1.)
 ax.tick_params(which='major',labelsize=7,length=3,width=.85,pad=2)
 ax.tick_params(which='minor',length=1.3,width=.35)
 ax.grid(axis='y',which='major',color='#D8D8D8',lw=.45);ax.set_axisbelow(True)
 assert len(ax.lines)==1
 fig.savefig(OUT/'inverse_sqrt_scaling.png',dpi=600,bbox_inches='tight',pad_inches=.13)
 fig.savefig(OUT/'inverse_sqrt_scaling.pdf',bbox_inches='tight',pad_inches=.13);plt.close(fig)
 (OUT/'scaling_figure_provenance.json').write_text(json.dumps({
  'data':source.name,'data_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
  'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'blocks':15,'x_limits':[1000,5000],'x_scale':'log','y_scale':'log',
  'curves':['QUILL'],'smoothing':None,'markers':'white-centered circles at sampled widths',
  'marked_widths':[rows[i]['width'] for i in visible_indices],
  'x_major_ticks':[1000,2000,4000],'x_minor_ticks':[1500,3000,5000],
  'colormap':'viridis','colormap_position':.55,'color_rgba':[float(c) for c in color],
  'title':'QUILL approximation','function':r'f(x)=1/sqrt(x), x in [1e-4, 1]',
  'title_fontsize_pt':12,'function_title_fontsize_pt':9.2,'axis_label_fontsize_pt':8,
  'tick_label_fontsize_pt':7,'legend_position':'bottom','export_padding_inches':.13,
  'annotated_sample':last_visible},indent=2)+'\n')
 print(OUT/'inverse_sqrt_scaling.png')

if __name__=='__main__':main()
