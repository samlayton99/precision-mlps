"""Check prepared branch coverage and geometry before allocating a GPU."""
import json
from pathlib import Path

import numpy as np

root = Path('/workspace/junmiaoh/experiments/precision-mlps/mechanism-0365cd3-20260923/evidence/dilation_b52df33')
panels = {}
for name, expected in [('late_development', 138), ('late_confirmation', 138),
                       ('wide_N512', 72), ('wide_N1024', 72)]:
    with np.load(root/f'{name}.npz') as archive:
        cases = json.loads(str(archive['cases']))
        p = archive['p']
        assert len(cases) == len(p) == expected
        assert np.all(np.isfinite(p))
        w = (p.shape[1]-1)//3
        base = {c['source_index']: p[i] for i, c in enumerate(cases) if c['arm'] == 'original'}
        for i, case in enumerate(cases):
            np.testing.assert_array_equal(p[i, :2*w], case['scale']*base[case['source_index']][:2*w])
        repairs = [c['repair'] for c in cases if c['arm'] != 'original' and c['valid']]
        panels[name] = dict(attempts=len(cases), width=w, samples=len(archive['x']),
            originals=len(base), targets=sorted({c['target'] for c in cases}),
            starts=sorted({c['start'] for c in cases}), seeds=sorted({c['seed'] for c in cases}),
            accepted=sum(c['valid'] for c in cases),
            failures=[c for c in cases if not c['valid']],
            normalized_balance_max=max(r['normalized_balance'] for r in repairs),
            normalized_stationarity_max=max(r['normalized_stationarity'] for r in repairs),
            tracking_to_effective_scale_max=max(r['tracking_to_effective_scale'] for r in repairs),
            geometry_locked_exactly=True,
            input_sources=json.loads(str(archive['sources'])),
            code_sources=json.loads(str(archive['code_sources'])))
result = dict(panels=panels, attempts=sum(p['attempts'] for p in panels.values()),
              accepted=sum(p['accepted'] for p in panels.values()))
(root/'preparation_summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
