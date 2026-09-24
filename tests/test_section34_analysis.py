"""Prevent plotted trace reduction from hiding optimizer excursions."""
import numpy as np
from experiments.expD36_frozen_gamma_probe.section34_analyze import binned_traces


def test_trace_reduction_retains_one_update_excursions_and_endpoint():
    traces = np.ones((101, 5, 2))
    traces[37, 2, 1] = 777.
    traces[83, 1, 1] = 1e-4
    traces[-1, :, 1] = np.arange(5)+2.
    reduced = binned_traces(traces, recipe=1, bins=10)
    assert reduced['high'].max() == 777.
    assert reduced['low'].min() == 1e-4
    np.testing.assert_array_equal(reduced['endpoint'], traces[-1, :, 1])
    assert reduced['steps'][-1] == 100
    assert reduced['median'].shape == (10, 5)
    assert np.all(reduced['low'] <= reduced['median'])
    assert np.all(reduced['median'] <= reduced['high'])


def test_selection_uses_one_recipe_across_seeds(tmp_path, monkeypatch):
    """Per-seed winners differ; the published comparison must remain one recipe."""
    import json
    from experiments.expD36_frozen_gamma_probe import section34_analyze as module
    base, run, out = [tmp_path/name for name in ['base','run','out']]
    base.mkdir(); run.mkdir()
    (base/'manifest.json').write_text(json.dumps(dict(width=2, spacing=.5)))
    x=np.linspace(-1,1,9);y=np.ones(9)
    np.savez(base/'input.npz',train_x=x,target=y,validation_x=x,validation_target=y,eval_x=x,eval_target=y)
    endpoint_errors=np.array([[.1,.3],[.1,.3],[.9,.3],[.9,.3],[.9,.3]])
    params=np.zeros((5,2,7));params[:,:,:2]=.2;params[:,:,-1]=1+endpoint_errors
    np.savez(run/'state.npz',p=params,count=10)
    np.save(run/'relative_error.npy',np.broadcast_to(endpoint_errors,(11,5,2)))
    np.save(run/'slope_rms.npy',np.full((11,5,2),.2))
    np.save(run/'parameter_checkpoints.npy',np.stack([params,params]))
    np.save(run/'checkpoint_steps.npy',[0,10])
    (run/'metadata.json').write_text(json.dumps(dict(width=2,recipes=[dict(schedule='constant',learning_rate=.1),dict(schedule='constant',learning_rate=.2)])))
    monkeypatch.setattr(module,'plot_joint',lambda *args:None)
    monkeypatch.setattr(module,'plot_candidates',lambda *args:None)
    module.analyze(base,[('adam',run)],out)
    result=json.loads((out/'summary.json').read_text())
    assert result['selected']['adam']['recipe_index']==1
    curves=np.load(out/'selected_traces.npz')
    np.testing.assert_allclose(curves['adam_error_endpoint'],.3)
    # An invalid per-seed selection would instead contain [.1,.1,.3,.3,.3].
    assert not np.allclose(curves['adam_error_endpoint'],np.min(endpoint_errors,axis=1))
