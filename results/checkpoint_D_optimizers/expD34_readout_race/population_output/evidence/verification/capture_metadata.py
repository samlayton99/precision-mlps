import hashlib, importlib.metadata, json, platform
from pathlib import Path
base=Path('/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923')
out=base/'evidence/population_output_cb89e72'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p): return dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p))
code=base/'code'
sources=sorted(set(list((code/'experiments/expD34_readout_race').glob('population*.py'))+list((code/'experiments/expD34_readout_race').glob('adam_output*.py'))+list((code/'experiments/expD34_readout_race').glob('mechanism_dilation*.py'))+list((code/'tests').glob('*population*.py'))+list((code/'tests').glob('*adam_output*.py'))))
versions={}
for package in ['numpy','scipy','jax','jaxlib','optax','matplotlib','python-flint','pytest']:
 try: versions[package]=importlib.metadata.version(package)
 except importlib.metadata.PackageNotFoundError: versions[package]=None
manifests=sorted(out.rglob('manifest.json'))
inputs=list(out.glob('*.npz'))
for m in manifests:
 try:
  data=json.loads(m.read_text())
  for key in ('input','input_path','source'):
   val=data.get(key)
   if isinstance(val,str) and val.endswith('.npz') and Path(val).is_file(): inputs.append(Path(val))
 except (ValueError,AttributeError): pass
inventory=[]
for name in ['diagnostic_batch_check','frozen_witness','archive_summary','archive_summary_refined','dilation_main_four_summary','dilation_main_controls_summary','force_concentration_summary']:
 p=out/name
 files=[q for q in p.rglob('*') if q.is_file()]
 item=dict(path=str(p),files=[record(q) for q in files],total_bytes=sum(q.stat().st_size for q in files))
 if name=='frozen_witness':
  item['matches_final']={str(q.relative_to(p)): (out/'frozen_witness_final'/q.relative_to(p)).is_file() and digest(q)==digest(out/'frozen_witness_final'/q.relative_to(p)) for q in files}
 inventory.append(item)
result=dict(python=platform.python_version(),libraries=versions,sources=[record(p) for p in sources],prepared_inputs=[record(p) for p in sorted(set(inputs))],manifests=[record(p) for p in manifests],cleanup_inventory=inventory)
(out/'reproducibility_metadata.json').write_text(json.dumps(result,indent=2)+'\n')
