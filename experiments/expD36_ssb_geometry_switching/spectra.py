"""Frequency-resolved function changes and physical parameter trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.expD06_fixed_center_scales import core, difference_training as maps


def physical(z,g,coord):
    weights=maps.decode(z[:g.width+1],g,'parameter_scale' if coord=='individual' else 'parameter_differences',xp=np)
    return weights,z[g.width+1:]/g.h


def transform(value):return np.fft.rfft(value)/np.sqrt(len(value))


def bands(size):
    ranges=[(0,1),(1,2)];lower=2
    while lower<size:
        upper=min(2*lower,size);ranges.append((lower,upper));lower=upper
    return ranges


def summarize(r,readout,geometry):
    rf,wf,gf=map(transform,(r,readout,geometry));weights=np.full(len(rf),2.);weights[0]=1.
    if len(r)%2==0:weights[-1]=1.
    ranges=bands(len(rf));output=[]
    for lo,hi in ranges:
        q=slice(lo,hi);rw=weights[q]
        output.append(dict(indices=[lo,hi-1],residual_mse=float(np.sum(rw*np.abs(rf[q])**2)),
            readout_energy=float(np.sum(rw*np.abs(wf[q])**2)),geometry_energy=float(np.sum(rw*np.abs(gf[q])**2)),
            readout_descent=float(-2*np.sum(rw*np.real(np.conj(rf[q])*wf[q]))),
            geometry_descent=float(-2*np.sum(rw*np.real(np.conj(rf[q])*gf[q])))))
    total=sum(q['residual_mse'] for q in output)
    for q in output:q['residual_percent']=100*q['residual_mse']/total
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);records=[]
    for path in sorted(args.root.glob('*/case.json')):
        c=json.loads(path.read_text());d=path.parent
        if c.get('implementation')!='primed_guard_v2' or c['n']!=128 or c['seed']!=0 or c['policy'] not in ('baseline','adaptive'):continue
        if not (d/'diagnostic_000020000.npz').exists():continue
        g=core.geometry(c['n']);x=np.linspace(-1,1,16*c['n']+1);m=len(x)
        z0=np.load(d/'snapshot_000019500.npz')['z'];z1=np.load(d/'snapshot_000020000.npz')['z'];w0,ga0=physical(z0,g,c['coordinates']);w1,ga1=physical(z1,g,c['coordinates'])
        phi0=np.tanh((x[:,None]-g.centers)*ga0);phi1=np.tanh((x[:,None]-g.centers)*ga1)
        dw=w1-w0
        readout=(dw[0]+.5*(phi0+phi1)@dw[1:])/np.sqrt(m)
        geometry=.5*(phi1-phi0)@(w0[1:]+w1[1:])/np.sqrt(m)
        actual=((w1[0]+phi1@w1[1:])-(w0[0]+phi0@w0[1:]))/np.sqrt(m)
        data=np.load(d/'diagnostic_000020000.npz');r=data['residual']
        # Window change uses an independently evaluated residual at its start.
        r0=(w0[0]+phi0@w0[1:]-core.target(x,c['target'],np))/np.sqrt(m)
        direction_rows=[]
        for direction in data['directions']:
            wd,gd=physical(direction,g,c['coordinates'])
            fw=(wd[0]+phi1@wd[1:])/np.sqrt(m)
            exponent=np.exp(-2*np.abs((x[:,None]-g.centers)*ga1))
            sech2=4*exponent/(1+exponent)**2
            fg=(sech2*(x[:,None]-g.centers))@(w1[1:]*gd)/np.sqrt(m)
            direction_rows.append(summarize(r,fw,fg))
        record=dict(id=d.name,config=c,window=[19500,20000],window_decomposition=summarize(r0,readout,geometry),
            direction_decompositions=direction_rows,closure_rms=float(np.linalg.norm(readout+geometry-actual)),
            function_change_rms=float(np.linalg.norm(actual)),final_mse=float(r@r))
        initial_w,_=physical(np.load(d/'snapshot_000000000.npz')['z'],g,c['coordinates'])
        record['weight_to_envelope_quantiles']=np.quantile(np.abs(w1[1:]/g.alpha[1:]),[.5,.9,1]).tolist()
        record['native_weight_norm_growth']=float(np.linalg.norm(w1[1:]/g.alpha[1:])/np.linalg.norm(initial_w[1:]/g.alpha[1:]))
        exact=-2*data['directions']@data['gradient']
        spectral=np.array([sum(q['readout_descent']+q['geometry_descent'] for q in rr) for rr in direction_rows])
        record['descent_accounting_relative_error']=(np.abs(spectral-exact)/np.maximum(np.abs(exact),1e-300)).tolist()
        records.append(record)
        labels=['DC' if q['indices']==[0,0] else str(q['indices'][0]) if q['indices'][0]==q['indices'][1] else f"{q['indices'][0]}–{q['indices'][1]}" for q in direction_rows[0]]
        fig,axes=plt.subplots(1,2,figsize=(12,4));xx=np.arange(len(labels))
        axes[0].bar(xx,[q['residual_percent'] for q in direction_rows[0]],color='#555555')
        axes[0].set_ylabel('Percent of residual MSE');axes[0].set_title('Residual at update 20,000')
        for k,(name,color) in enumerate((('SSB direction','#0072b2'),('Scale-aware direction','#d55e00'))):
            total=np.array([q['readout_descent']+q['geometry_descent'] for q in direction_rows[k]])
            axes[1].plot(xx,total,marker='o',label=name,color=color)
        axes[1].set_yscale('symlog',linthresh=1e-15);axes[1].axhline(0,color='gray',lw=.6)
        axes[1].set_ylabel('MSE decrease / unit native step');axes[1].set_title('Direction × residual, by frequency');axes[1].legend(fontsize=8)
        for ax in axes:ax.set_xticks(xx,labels,rotation=65);ax.set_xlabel('DFT index (paired positive/negative frequencies)');ax.grid(axis='y',alpha=.2)
        fig.suptitle(f"{c['target']}; {c['coordinates']}; {c['policy']}; N=128, seed 0")
        fig.tight_layout();fig.savefig(args.output/f"spectrum_{c['coordinates']}_{c['target']}_{c['policy']}.png",dpi=180);plt.close(fig)
        if c['policy']=='baseline':
            fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
            for step in (0,5000,20000):
                z=np.load(d/f'snapshot_{step:09d}.npz')['z'];w,ga=physical(z,g,c['coordinates'])
                axes[0].plot(g.centers,w[1:],'.-',ms=2,lw=.8,label=f'Update {step:,}')
                axes[1].plot(g.centers,g.h*ga,'.-',ms=2,lw=.8)
            axes[0].set_yscale('symlog',linthresh=.05);axes[1].set_yscale('symlog',linthresh=.01)
            axes[0].set_ylabel('Physical readout w');axes[1].set_ylabel('Signed bandwidth λ = hγ');axes[1].set_xlabel('Fixed physical center')
            axes[0].legend();axes[0].set_title(f"{c['target']}; {c['coordinates']}; N=128, seed 0")
            for ax in axes:ax.axvspan(-1,1,color='gray',alpha=.07);ax.grid(alpha=.2)
            fig.tight_layout();fig.savefig(args.output/f"parameters_{c['coordinates']}_{c['target']}.png",dpi=180);plt.close(fig)
    (args.output/'spectra.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
