"""Redraw saved observations and force mechanisms on bounded Modal CPU.

No training or report generation. The spectrum checkout supplies the existing
five-million-update comparison; the current checkout supplies force audits.
Only figures and a numerical provenance record are returned to the laptop.
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/work')
SPECTRUM = Path(os.environ.get('D34_SPECTRUM_ROOT', str(ROOT.parent/'precision-mlps-frozen-gamma-probe')))
ANALYSIS = Path('output/diagnostics/section34_long_horizon/main_5m')
BASE = Path('results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/section34_long_horizon_20260924/main')
FORCES = Path('results/checkpoint_D_optimizers/expD34_readout_race/adam_force_extension/curated')
FLOW = Path('results/checkpoint_D_optimizers/expD34_readout_race/population_output/evidence/feedback_flow_100k/states.csv')
EXTERNAL = (ANALYSIS/'joint_analysis/summary.json', ANALYSIS/'joint_analysis/selected_traces.npz',
            ANALYSIS/'uniform_access/summary.json', ANALYSIS/'uniform_access/assay_traces.npz',
            BASE/'base/joint_input.npz', BASE/'base/manifest.json', BASE/'h5m/frozen_gd/raw_error.npy')
LOCAL = (FLOW, *(FORCES/f'primary_{seed}'/name for seed in range(5)
                 for name in ('trace.npz', 'manifest.json')))
app = modal.App('d34-narrative-figures')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('numpy==2.2.6', 'matplotlib==3.10.3')
         .env({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}))
if modal.is_local():
    for root, prefix, paths in ((ROOT, '/work', LOCAL), (SPECTRUM, '/spectrum', EXTERNAL)):
        for path in paths:
            image = image.add_local_file(root/path, str(Path(prefix)/path))


@app.function(image=image, cpu=2, memory=(1024, 4096), timeout=300, max_containers=1, retries=0)
def render() -> bytes:
    import resource
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    output = Path('/tmp/narrative')
    output.mkdir()
    source = Path('/spectrum')
    summary = json.loads((source/ANALYSIS/'joint_analysis/summary.json').read_text())
    access = json.loads((source/ANALYSIS/'uniform_access/summary.json').read_text())
    manifest = json.loads((source/BASE/'base/manifest.json').read_text())
    assert summary['horizon'] == access['assay_steps'] == 5000000
    assert summary['width'] == 512
    assert manifest['input_sha256'] == access['input_sha256']
    horizon, width = summary['horizon'], summary['width']
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                         'legend.fontsize': 7, 'legend.frameon': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})

    def save(fig, name):
        for suffix in ('png', 'svg', 'pdf'):
            fig.savefig(output/f'{name}.{suffix}', dpi=300)
        svg = output/f'{name}.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.05), layout='constrained')
    record = {'observation': {}, 'tracking': {}, 'reinforcement': {}}
    with np.load(source/BASE/'base/joint_input.npz', allow_pickle=False) as initial:
        x, y = initial['x'], initial['target']
        initial_error = []
        for p in initial['initial_parameters']:
            a, b, c = p[:-1].reshape(3, width)
            initial_error.append(np.linalg.norm(np.tanh(x[:, None]*a+b)@c+p[-1]-y)/np.linalg.norm(y))
    with np.load(source/ANALYSIS/'joint_analysis/selected_traces.npz', allow_pickle=False) as trace:
        for optimizer, label, color in (('adam', 'Adam', '#0072B2'), ('gd', 'GD', '#D55E00')):
            selected = summary['selected'][optimizer]
            endpoint = trace[f'{optimizer}_error_endpoint']
            np.testing.assert_allclose(endpoint, selected['train_errors'], rtol=1e-10)
            assert len(endpoint) == 5
            t = np.r_[0, trace[f'{optimizer}_error_steps'], horizon]/1e6
            mid, low, high = [np.vstack((initial_error, trace[f'{optimizer}_error_{key}'], endpoint))
                              for key in ('median', 'low', 'high')]
            axes[0].plot(t, np.median(mid, axis=1), color=color, label=f'Joint {label}')
            axes[0].fill_between(t, low.min(axis=1), high.max(axis=1), color=color, alpha=.07, lw=0)
            ts = np.r_[0, trace[f'{optimizer}_rms_steps'], horizon]/1e6
            start, end = trace[f'{optimizer}_lambda_rms_checkpoints'][0], trace[f'{optimizer}_rms_endpoint']
            mid, low, high = [np.vstack((start, trace[f'{optimizer}_rms_{key}'], end))
                              for key in ('median', 'low', 'high')]
            assert np.all(low <= mid) and np.all(mid <= high)
            axes[1].plot(ts, np.median(mid, axis=1), color=color, label=f'{label} RMS')
            axes[1].fill_between(ts, low.min(axis=1), high.max(axis=1), color=color, alpha=.1, lw=0)
            record['observation'][optimizer] = dict(final_rms=float(np.median(end)),
                final_relative_training_error=float(np.median(endpoint)),
                final_relative_dense_error=float(np.median(selected['eval_errors'])),
                schedule=selected['schedule'], learning_rate=selected['learning_rate'])
    gi = next(i for i, row in enumerate(access['geometries'])
              if row['family'] == 'uniform' and np.isclose(row['lambda_rms'], .25))
    selected = access['geometries'][gi]['selected']
    with np.load(source/ANALYSIS/'uniform_access/assay_traces.npz', allow_pickle=False) as curves:
        steps = curves[f'g{gi}_steps']
        groups = np.minimum((steps/steps[-1]*200).astype(int), 199)
        groups[0], groups[-1] = -1, 200
        display = [(0., 1., 1., 1.)]
        for group in np.unique(groups):
            mask = groups == group
            display.append((np.median(steps[mask]), np.median(curves[f'g{gi}_median'][mask]),
                            np.min(curves[f'g{gi}_low'][mask]), np.max(curves[f'g{gi}_high'][mask])))
        endpoint = float(curves[f'g{gi}_endpoint'])
        display.append((horizon, endpoint, endpoint, endpoint))
        t, mid, low, high = np.asarray(display).T
        axes[0].plot(t/1e6, mid, color='#333333', ls='--', lw=.9, label='Fixed Adam')
        axes[0].fill_between(t/1e6, low, high, color='#333333', alpha=.07, lw=0)
    # Map the 153 MiB scalar file remotely; never load a parameter archive.
    actual = np.load(source/BASE/'h5m/frozen_gd/raw_error.npy', mmap_mode='r', allow_pickle=False)
    assert actual.shape == (horizon+1, 4)
    indices = np.unique(np.r_[0, np.geomspace(1, horizon, 2300).astype(int),
                              np.linspace(0, horizon, 1800).astype(int)])
    axes[0].plot(indices/1e6, actual[indices, -1], color='#777777', ls=':', label='Fixed GD')
    axes[1].axhline(.25, color='#333333', ls=':', lw=.9, label='Supplied geometry: 1/4')
    axes[0].set(title='(a) Output accuracy', ylabel='Relative training error')
    axes[1].set(title='(b) Population slope scale', ylabel=r'RMS relative slope $\lambda_{\rm RMS}$')
    for ax in axes:
        ax.set(xlabel='Updates (millions)', xlim=(0, 5), yscale='log')
        ax.grid(axis='y', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(loc='upper right', ncol=2, columnspacing=.8, handlelength=1.5)
    axes[1].legend(loc='lower right', fontsize=6.5)
    record['observation'].update(width=width, spacing=summary['spacing'], seeds=5, updates=horizon,
        fixed_gd_final_training_error=float(actual[-1, -1]), fixed_adam_final_training_error=endpoint,
        fixed_adam_selection=selected)
    save(fig, 'joint_acquisition_rms')

    transition, transition_ax = plt.subplots(figsize=(5.5, 2.05), layout='constrained')
    names = ['raw_total_norm', 'raw_effective_norm', 'raw_tracking_norm']
    channels = []
    for seed in range(5):
        folder = Path('/work')/FORCES/f'primary_{seed}'
        meta = json.loads((folder/'manifest.json').read_text())
        assert (meta['m'], meta['width'], meta['max_updates']) == (2048, 177, 600000)
        case_index = next(i for i, case in enumerate(meta['cases'])
                          if case['optimizer'] == 'gd' and case['target'] == 'mixed_sine')
        assert meta['cases'][case_index]['eta'] == .002
        columns = [meta['metrics'].index(name) for name in names]
        with np.load(folder/'trace.npz', allow_pickle=False) as data:
            values = data['values'][case_index]
            assert np.all(values[:, meta['metrics'].index('coarse_resolved')] == 1)
            assert np.all(values[:, meta['metrics'].index('raw_unresolved_norm')] == 0)
            if seed == 0:
                updates = data['ends']-1
            else:
                np.testing.assert_array_equal(updates, data['ends']-1)
            channels.append(values[:, columns])
    channels = np.asarray(channels)
    assert np.all(channels > 0)
    for j, (label, color) in enumerate((('Full slope gradient', '#111827'),
                                      ('Effective fine', '#0072B2'), ('Coarse tracking', '#D55E00'))):
        values = channels[:, :, j]
        transition_ax.plot(updates, np.median(values, axis=0), color=color, lw=1.2, label=label)
        transition_ax.fill_between(updates, values.min(axis=0), values.max(axis=0), color=color, alpha=.12, lw=0)
    transition_ax.set(xscale='log', yscale='log', xlabel='Total GD updates', ylabel='Slope-gradient norm')
    transition_ax.legend(loc='lower left')
    transition_ax.grid(alpha=.15)
    transition_ax.spines[['top', 'right']].set_visible(False)
    record['tracking'] = dict(target='mixed_sine', width=177, seeds=5, final_update=int(updates[-1]),
                              final_tracking_to_fine_ratio=(channels[:, -1, 2]/channels[:, -1, 1]).tolist())

    save(transition, 'tracking_transition')
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15), layout='constrained')
    with (Path('/work')/FLOW).open(newline='') as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [row for row in all_rows
            if row['target'] == 'step_right' and row['kind'] == 'effective' and float(row['dt']) == .01]
    gd = sorted([row for row in all_rows if row['target'] == 'step_right'
                 and row['kind'] == 'gd' and float(row['dt']) == .002], key=lambda row: float(row['time']))
    rows.sort(key=lambda row: float(row['time']))
    get = lambda name: np.array([float(row[name]) for row in rows])
    t, force = get('time'), get('f')
    assert len(t) == 201 and t[-1] == 200
    gd_force = np.array([float(row['f']) for row in gd])
    gd_tracking = np.array([float(row['R_norm']) for row in gd])
    np.testing.assert_array_equal(t, [float(row['time']) for row in gd])
    np.testing.assert_allclose(gd_force[0], force[0], rtol=1e-12)
    age = 20+t/.002/1000
    axes[0].plot(age, gd_force, color='#0072B2', label='Effective fine F')
    axes[0].plot(age, gd_tracking, color='#D55E00', label='Coarse tracking R')
    axes[0].set(yscale='log', ylabel='Full parameter-gradient norm', title='(a) After the tracking transient')
    axes[0].legend(loc='center right', fontsize=6.5)
    integral = lambda values: np.r_[0., np.cumsum(np.diff(t)*(values[1:]+values[:-1])/2)]
    feedback = integral(get('geometry')+get('compensation'))
    depletion = -integral(get('relaxation'))
    growth = np.log(force/force[0])
    defect = float(np.max(np.abs(feedback+depletion-growth)))
    assert defect < 1e-4, defect
    axes[1].plot(age, feedback, color='#009E73', label='Geometry + compensation')
    axes[1].plot(age, depletion, color='#CC6677', label='Residual relaxation')
    axes[1].plot(age, growth, color='#111827', ls='--', label='Net log force change')
    axes[1].axhline(0, color='#999999', lw=.5)
    axes[1].set(ylabel='Contribution to log force change', title='(b) Limited force reinforcement')
    axes[1].legend(loc='upper left', fontsize=6.1)
    for ax in axes:
        ax.set(xlabel='Total GD updates (thousands)', xlim=(20, 120), xticks=[20, 60, 100, 120])
        ax.grid(alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    record['reinforcement'] = dict(target='step_right', width=705, seed=30, restart=20000,
        additional_updates=100000, initial_force=float(force[0]), final_force_ratio=float(force[-1]/force[0]),
        feedback=float(feedback[-1]), relaxation=float(depletion[-1]), quadrature_identity_defect=defect,
        final_relative_error=float(get('relative_error')[-1]),
        gd_max_tracking_to_fine_ratio=float(np.max(gd_tracking/gd_force)),
        gd_effective_max_relative_force_difference=float(np.max(np.abs(gd_force-force)/force)))
    save(fig, 'tracking_and_reinforcement')
    record['sources'] = []
    for prefix, paths in (('/work', LOCAL), ('/spectrum', EXTERNAL)):
        for path in paths:
            with (Path(prefix)/path).open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            record['sources'].append(dict(path=str(Path(prefix)/path), sha256=digest))
    record.update(platform='Modal CPU', memory_hard_limit_mib=4096,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (output/'facts.json').write_text(json.dumps(record, indent=2)+'\n')
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output.iterdir():
            archive.write(path, path.name)
    return packed.getvalue()


@app.local_entrypoint()
def main(output: str):
    destination = Path(output)
    if destination.exists():
        raise ValueError('Use a new output directory')
    packed = render.remote()
    with zipfile.ZipFile(io.BytesIO(packed)) as archive:
        if sum(item.file_size for item in archive.infolist()) > 8*1024**2:
            raise ValueError('Download cap exceeded')
        if any(Path(item.filename).name != item.filename for item in archive.infolist()):
            raise ValueError('Flat artifacts only')
        destination.mkdir(parents=True)
        archive.extractall(destination)
    print(f'Downloaded {len(packed)} bytes to {destination}')
