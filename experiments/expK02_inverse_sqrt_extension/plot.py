"""Scientific summary in the typography of the paper's three-panel scaling figure."""
import os,sys,json
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,NullFormatter
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expK02_inverse_sqrt_extension.construction import Extension,Network
OUT=ROOT/'results/checkpoint_K_practical_recipes/expK02_inverse_sqrt_extension'

def main():
 info=json.loads((OUT/'verification.json').read_text());rows=json.loads((OUT/'results.json').read_text());compact=json.loads((OUT/'scaling_2k_6k.json').read_text())
 ext=Extension(order=16);net=Network.load(OUT/info['selected_model'])
 plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans','font.size':7})
 fig,axes=plt.subplots(1,3,figsize=(9.6,3.2))
 fig.subplots_adjust(left=.065,right=.985,bottom=.18,top=.84,wspace=.32)
 ax=axes[0];t=np.linspace(-.5,4,1201);x=t*ext.eps;extension=ext.value(x)
 ax.axvspan(-.5,1,color='#E6E6E6',alpha=.6,lw=0)
 ax.axvline(1,color='#888888',lw=.7,ls=':')
 ax.plot(t,extension,color='#009E73',lw=1.45,label=r'$C^{17}$ extension')
 on=t>=1
 ax.plot(t[on],x[on]**-.5,color='#0072B2',lw=1.45,label=r'Target $x^{-1/2}$')
 ax.plot(t,net.predict(x),color='#444444',lw=1.2,ls='--',label='QI network')
 ax.set(xlim=(-.5,4),ylim=(40,135),xticks=[0,1,2,3,4],yticks=[50,75,100,125],xlabel=r'$x/\varepsilon$',ylabel='Function value')
 ax.set_title('a) Smooth halo extension',fontweight='bold',fontsize=8.5,pad=8)
 ax.legend(loc='upper right',frameon=False,fontsize=6.5,handlelength=1.8)
 ax.text(.05,.08,'Halo side',transform=ax.transAxes,color='#666666',fontsize=6.3)
 ax=axes[1];uniform=[r for r in rows if r['method']=='uniform']
 ax.semilogy([r['width'] for r in uniform],[r['relative_l2'] for r in uniform],color='#E69F00',lw=1.45,label='One uniform QI grid')
 ax.semilogy([r['width'] for r in compact],[r['relative_l2'] for r in compact],color='#0072B2',lw=1.45,label='15 scaled QI blocks')
 ax.set(xlim=(2000,6000),ylim=(1e-15,1e3),xlabel='Total hidden width',ylabel=r'Relative $L^2$ error')
 ax.set_xticks([2000,3000,4000,5000,6000],['2k','3k','4k','5k','6k']);ax.set_yticks([1e2,1e-2,1e-6,1e-10,1e-14])
 ax.yaxis.set_minor_locator(LogLocator(base=10,subs=(2.,5.),numticks=100));ax.yaxis.set_minor_formatter(NullFormatter())
 ax.set_title('b) Accuracy versus width',fontweight='bold',fontsize=8.5,pad=8)
 ax.legend(loc='center right',bbox_to_anchor=(1.,.68),frameon=False,fontsize=6.3,handlelength=1.8)
 ax=axes[2];xx=np.geomspace(ext.eps,1,4097);truth=xx**-.5;err=abs((net.predict(xx)-truth)/truth)
 ax.loglog(xx,np.where(err>0,err,np.nan),color='#0072B2',lw=.7)
 ax.set(xlim=(ext.eps,1),ylim=(1e-16,1e-12),xlabel=r'$x$',ylabel=r'$|q(x)-f(x)|/|f(x)|$')
 ax.set_xticks([1e-4,1e-3,1e-2,1e-1,1]);ax.set_yticks([1e-16,1e-14,1e-12])
 ax.set_title(f"c) Exported tanh MLP: {net.width:,} neurons",fontweight='bold',fontsize=8.5,pad=8)
 ax.text(.97,.08,rf"Relative $L^2$: {info['relative_l2_quadrature']['512']:.1e}",transform=ax.transAxes,ha='right',fontsize=6.5,
         bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
 for ax in axes:
  ax.xaxis.label.set_fontsize(8);ax.yaxis.label.set_fontsize(8)
  ax.xaxis.labelpad=ax.yaxis.labelpad=3
  for side in ['top','right']:ax.spines[side].set_visible(False)
  for side in ['left','bottom']:ax.spines[side].set_linewidth(1.)
  ax.tick_params(which='major',labelsize=7,length=3,width=.85,pad=3)
  ax.tick_params(which='minor',length=1.3,width=.35)
  ax.grid(axis='y',which='major',color='#D8D8D8',lw=.45)
  ax.set_axisbelow(True)
 fig.savefig(OUT/'inverse_sqrt_qi.png',dpi=600);fig.savefig(OUT/'inverse_sqrt_qi.pdf');plt.close(fig)
 print(OUT/'inverse_sqrt_qi.png')

if __name__=='__main__':main()
