"""Reproduce the complete scalar construction, including its target readout."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import torch
from experiments.expD39_qi_init_theory.analyze import plt, save
from experiments.expD39_qi_init_theory.run import OUT, base
from src.construction.qi_mpmath import construct_qi, evaluate_qi
from src.data.targets import get_target


def main():
    torch.set_num_threads(1)
    target=get_target('sine')
    x=np.linspace(-1,1,2001)
    y=target.fn_numpy(x)
    records=[]
    fig,ax=plt.subplots(figsize=(10,4.5))
    for precision,color in [('fp64','tab:blue'),('mpmath','tab:orange')]:
        qi=construct_qi(target.fn_numpy,target.deriv_numpy,N=64,precision=precision)
        width=len(qi.centers)
        model=torch.nn.Sequential(torch.nn.Linear(1,width),torch.nn.Tanh(),torch.nn.Linear(width,1)).double()
        with torch.no_grad():
            model[0].weight.fill_(qi.gamma)
            model[0].bias.copy_(torch.from_numpy(-qi.gamma*qi.centers))
            model[2].weight.copy_(torch.from_numpy(qi.a_coeffs)[None,:])
            model[2].bias.fill_(qi.c0)
            pred=model(torch.from_numpy(x[:,None])).numpy().ravel()
        direct=evaluate_qi(qi,x)
        error=np.abs(pred-y)
        assert np.max(np.abs(pred-direct))<2e-14
        threshold=1e-10 if precision=='fp64' else 1e-14
        assert np.max(error)<threshold,(precision,np.max(error))
        records.append(dict(construction_arithmetic=precision,lambda_value=qi.lambda_val,width=width,
                            intervals=64,halo=qi.halo,torch_linf=float(np.max(error)),
                            direct_linf=float(np.max(np.abs(direct-y))),
                            error=error.tolist(),model_evaluation_dtype='float64'))
        ax.plot(x,np.maximum(error,1e-17),color=color,alpha=.7,
                label=f'{precision} coefficients · λ={qi.lambda_val} · width {width}')
    ax.set(yscale='log',ylim=(1e-17,1e-10),xlim=(-1,1),xlabel='x',ylabel='Absolute error',
           title='Complete 1-D QI construction → ordinary tanh MLP\nTarget-specific readout · float64 model and evaluation')
    ax.grid(which='both',alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.04),ncol=2)
    fig.subplots_adjust(top=.76)
    save(fig,'constructor_reference.png')
    base.save_json(OUT/'constructor_reference.json',dict(target='sine',x=x.tolist(),records=records,
          note='Two documented construction operating points, not an isolated arithmetic ablation. Exact zero errors plotted at1e-17.'))
    print([(r['construction_arithmetic'],r['torch_linf']) for r in records])


if __name__=='__main__':
    main()
