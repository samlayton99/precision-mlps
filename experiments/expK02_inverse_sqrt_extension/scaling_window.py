"""Measure fixed-15-block width sweeps for the requested plotting windows."""
import sys,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expK02_inverse_sqrt_extension.construction import Extension,build,metrics
OUT=ROOT/'results/checkpoint_K_practical_recipes/expK02_inverse_sqrt_extension'

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--window',choices=['2k-6k','1k-5k'],default='2k-6k');args=parser.parse_args()
 ext=Extension(order=16);rows=[]
 sizes=list(range(64,337,16)) if args.window=='2k-6k' else [1,2,4,8,16,24,32,48,*range(64,273,16)]
 existing={}
 if args.window=='1k-5k':
  for name in ['compact_sweep.json','scaling_2k_6k.json']:
   if (OUT/name).exists():existing.update({r['N']:r for r in json.loads((OUT/name).read_text())})
 destination=OUT/('scaling_'+args.window.replace('-','_')+'.json')
 for n in sizes:
  if n in existing:
   rows.append(dict(existing[n],blocks=len(ext.scales)));destination.write_text(json.dumps(rows,indent=2)+'\n');continue
  start=time.monotonic();net=build(ext,n,OUT/'cache',halo=32)
  model=f'models/multiscale_p16_N{n}_R32_window.npz';net.save(OUT/model)
  rows.append(dict(N=n,halo=32,blocks=len(ext.scales),width=net.width,
                   relative_l2=metrics(net,ext.eps,256),model=model,seconds=time.monotonic()-start))
  destination.write_text(json.dumps(rows,indent=2)+'\n')
  print(n,net.width,rows[-1]['relative_l2'],flush=True)

if __name__=='__main__':main()
