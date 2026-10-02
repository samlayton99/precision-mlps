"""Retained row directions versus retained alignment within original QI banks."""
from __future__ import annotations
import json
import numpy as np
from scipy.linalg import svdvals
import torch
from experiments.expD39_qi_init_theory.run import base, make_model, OUT, BASE_OUT
from experiments.expD39_qi_init_theory.analyze import LABELS, save, plt


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    candidate = json.loads((OUT/'selection.json').read_text())['candidate']
    if candidate in ('soft24','sharp64'):
        from experiments.expD39_qi_init_theory.run import VARIANTS
        from experiments.expD39_qi_init_theory.followup import EXTRA
        VARIANTS.update(EXTRA)
    recipe = json.loads((BASE_OUT/'selected_recipe.json').read_text())['airfoil']
    records = []
    for scheme in ['qi',candidate]:
        root = BASE_OUT if scheme=='qi' else OUT
        path = root/'data/compare'/f"airfoil_{scheme}_w512_lr{recipe['lr']:g}_seed0_steps20000.json"
        result = json.loads(path.read_text())
        identity = result['identity']
        arrays, metadata = base.load_data('airfoil',identity['config'])
        assert metadata==result['data']
        inputs = {k:torch.from_numpy(v[0]) for k,v in arrays.items()}
        model, info = make_model(5,512,0,scheme,inputs['train'],identity['config'])
        pred,_ = base.predictions(model,inputs)
        for split in inputs:
            np.testing.assert_allclose(base.mse(pred[split],arrays[split][1]),result['trace'][0]['learned'][split],rtol=1e-10,atol=1e-12)
        ckpt = torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=True)
        assert base.same_identity(identity,ckpt['identity'])
        for name, key in [('fc1','first'),('fc2','second')]:
            before = getattr(model,name).weight.detach().numpy()
            after = ckpt['model'][name+'.weight'].numpy()
            gb,ga = [np.linalg.norm(w,axis=1) for w in [before,after]]
            ub,ua = before/gb[:,None],after/ga[:,None]
            self_cos = np.clip(np.sum(ub*ua,axis=1),-1,1)
            sb,sa=svdvals(before),svdvals(after)
            if scheme=='qi':
                sizes = [22]*23+[6]
            else:
                sizes = [b['size'] for b in info.get(key,{}).get('banks',[])]
            within = []
            start = 0
            for n in sizes:
                u = ua[start:start+n]
                within.append(float(np.clip((np.sum(u.sum(0)**2)-n)/(n*(n-1)),-1,1)))
                start += n
            records.append(dict(scheme=scheme,layer=name,initial_gamma=gb.tolist(),final_gamma=ga.tolist(),
                                row_cosine=self_cos.tolist(),within_bank_mean_pairwise_cosine=within,
                                initial_weight_rank=int(np.sum(sb>sb[0]*1e-12)),
                                final_weight_rank=int(np.sum(sa>sa[0]*1e-12)),
                                median_gamma_ratio=float(np.median(ga/gb)),median_row_cosine=float(np.median(self_cos)),
                                median_bank_cosine=float(np.median(within)) if within else None))
    base.save_json(OUT/'airfoil_movement.json',dict(rows=records,seed=0,steps=20000,
         weight_rank_relative_cutoff=1e-12,comparison='original QI vs global sharp64 candidate',
         caveat='Second-layer row directions are measured in hidden-neuron coordinates. Final gamma*h is not assigned after directions untie.'))
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for i,layer in enumerate(['fc1','fc2']):
        for row in [r for r in records if r['layer']==layer]:
            color='tab:orange' if row['scheme']=='qi' else 'tab:green'
            for j,key in enumerate(['row_cosine','within_bank_mean_pairwise_cosine']):
                values=row[key]
                if values:
                    axes[i,j].hist(values,bins=np.linspace(-1,1,81),weights=np.full(len(values),100/len(values)),alpha=.4,color=color,label=LABELS[row['scheme']])
        axes[i,0].set(title=f'Layer {i+1}: each row vs its initial direction',xlabel='Cosine similarity',ylabel='Rows (%)')
        axes[i,1].set(title=f'Layer {i+1}: alignment within original banks',xlabel='Mean pairwise cosine in bank',ylabel='Banks (%)')
    for ax in axes.flat:
        ax.set(xlim=(-1,1),ylim=(0,105))
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Airfoil · original QI vs global candidate · 20k steps\nWidth 512 · seed 0',y=1.04)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.97),ncol=2)
    fig.subplots_adjust(top=.84,hspace=.35,wspace=.2)
    save(fig,'airfoil_bank_alignment.png')


if __name__=='__main__':
    main()
