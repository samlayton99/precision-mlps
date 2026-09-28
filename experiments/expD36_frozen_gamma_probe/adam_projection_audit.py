"""Independent full gesvd check of archived Adam modal projections."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scipy.linalg import svd
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path('results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep'))
root=parser.parse_args().root
out=root/'refinements/gamma_optimizer_access'
hashes={}
def read_array(p):
 p=root/p;hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
 with np.load(p) as z:return dict(z)
def read_json(p):
 p=root/p;hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
common=read_array('common/N512/arrays.npz');saved=read_array('refinements/gamma_optimizer_access/projections.npz')
analysis=read_json('refinements/gamma_optimizer_access/analysis.json');selection=read_json('training/N512_raw_selection.json')
case=read_json('training/N512_raw_adam_continue/case.json')
steps=list(saved['checkpoint_steps']);targets=analysis['targets'];cutoffs=[2e-8,2e-7,2e-6,2e-5,2e-4,.02,1.]
checkpoints={}
for n in steps:
 stage='pilot' if n<=50000 else 'continue'
 checkpoints[n]=read_array(f'training/N512_raw_adam_{stage}/checkpoint_{n:06d}.npz')
y=common['y_train'];norm=np.sum(y*y,axis=0);rows=[]
for gamma in [8,12,16,64]:
 gi=case['gammas'].index(gamma)
 j=np.column_stack((np.ones(len(y)),np.tanh(gamma*(common['x_train'][:,None]-common['centers']))))/np.sqrt(len(y))
 u,s,vh=svd(j,full_matrices=False,lapack_driver='gesvd',check_finite=True)
 rho=(s/s[0])**2; oldrho=saved[f'g{gamma}_rho']
 # Treat numerical singular values below archive cutoff as unresolved, never exact zero.
 resolved=s>1e-14*s[0]
 spec=read_array(f'dictionaries/N512_raw_g{gamma}/spectrum.npz')
 bandrows=[];maximum_closure=0.;maximum_projection=0.
 for view in ['common','selected']:
  ci=[next(i for i,c in enumerate(case['columns']) if c['view']==view and c['target']==t)for t in targets]
  pi=np.asarray(selection['indices'])[gi,ci]
  theta=np.concatenate([checkpoints[n]['theta'][gi][:,pi if n<=50000 else ci] for n in steps],axis=1)
  r=j@theta-np.tile(y,(1,len(steps)))
  direct=u.T@r
  residual_perp=r-u@direct
  total=np.sum(r*r,axis=0)/np.tile(norm,len(steps))
  closure=np.abs((np.sum(direct*direct,axis=0)+np.sum(residual_perp*residual_perp,axis=0))/np.tile(norm,len(steps))-total)
  np.testing.assert_allclose(total.reshape(len(steps),len(targets)),saved[f'g{gamma}_{view}_total_energy'],rtol=2e-10,atol=1e-12)
  assert float(closure.max()) < 1e-10
  maximum_closure=max(maximum_closure,float(closure.max()))
  oldcoeff=spec['singular'][:,None]*(spec['Vh']@theta)-np.tile(spec['loadings'],(1,len(steps)))
  # Full physical sample-space projection comparison at each cutoff.
  for cutoff in cutoffs:
   mask=resolved&(rho<=cutoff);omask=oldrho<=cutoff
   fresh=np.sum(direct[mask]**2,axis=0)/np.tile(norm,len(steps))
   archived=saved[f'g{gamma}_{view}_residual_energy'][:,omask,:].sum(axis=1).reshape(-1)
   initial=np.sum((u[:,mask].T@y)**2,axis=0)/norm
   oldinitial=saved[f'g{gamma}_initial_energy'][omask].sum(axis=0)
   np.testing.assert_allclose(fresh,archived,rtol=0,atol=1e-10)
   np.testing.assert_allclose(initial,oldinitial,rtol=0,atol=1e-10)
   # Compare band-projector actions without forming m by m matrices. Saved U
   # is unavailable: reconstruct ONLY these resolved modes from J V/s.
   oldmask=(spec['singular']/spec['singular'][0])**2<=cutoff
   old_u=j@spec['Vh'][oldmask].T/spec['singular'][oldmask]
   # J V/s is ill-conditioned at extremely small sigma. Keep this as a
   # diagnostic, not the accuracy criterion for saved modal coefficients.
   projerr=np.linalg.norm(u[:,mask]@direct[mask]-old_u@oldcoeff[oldmask],axis=0)/np.sqrt(np.tile(norm,len(steps)))
   maximum_projection=max(maximum_projection,float(projerr.max()))
   bandrows.append(dict(view=view,cutoff=cutoff,fresh_resolved_count=int(mask.sum()),saved_count=int(omask.sum()),
    maximum_absolute_energy_gap=float(np.max(np.abs(fresh-archived))),maximum_initial_energy_gap=float(np.max(abs(initial-oldinitial))),
    maximum_reconstructed_saved_projector_action_gap=float(projerr.max()),
    fresh_energy=fresh.reshape(len(steps),len(targets)).tolist(),
    nearest_resolved_ratio_below=float(rho[mask].max())if mask.any()else None,
    nearest_ratio_above=float(rho[rho>cutoff].min())if np.any(rho>cutoff)else None))
 rows.append(dict(gamma=gamma,full_svd_modes=len(s),resolved_modes=int(resolved.sum()),
  maximum_energy_closure_gap=maximum_closure,maximum_reconstructed_saved_projector_action_gap=maximum_projection,bands=bandrows))
 print(gamma,'energy gap',max(b['maximum_absolute_energy_gap']for b in bandrows),'closure',maximum_closure,flush=True)
result=dict(method='Fresh full rectangular scipy gesvd; direct U.T@(Jtheta-y); CPU single BLAS thread.',cutoffs=cutoffs,
 checkpoint_steps=[int(n)for n in steps],targets=targets,cases=rows,source_sha256=hashes,
 checks=dict(checkpoint_target_comparisons=480,band_checkpoint_target_comparisons=3360,absolute_energy_tolerance=1e-10,status='passed'),
 verification_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 limitations=['FP64 independent numerical comparison, not interval certification.','Resolved modes require singular>1e-14*sigma1; smaller singular modes remain unresolved.','Reconstructed saved U=JV/s can lose accuracy near numerical cutoff; direct fresh-SVD energy is the primary check.'])
(out/'projection_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
