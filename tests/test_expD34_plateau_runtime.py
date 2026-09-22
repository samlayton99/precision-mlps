import sys
from types import SimpleNamespace

import pytest

from experiments.expD34_readout_race import plateau_runtime as runtime
from experiments.expD34_readout_race import run


def test_slurm_dispatch_keeps_allocation_checks(monkeypatch, tmp_path):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='Slurm step'):
        runtime.verify_gpu(tmp_path)
    monkeypatch.setenv('SLURM_JOB_ID', '12')
    monkeypatch.setenv('SLURM_STEP_ID', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(run.subprocess, 'check_output', lambda *a, **k: 'JobState=PENDING')
    with pytest.raises(RuntimeError, match='running GPU allocation'):
        runtime.verify_gpu(tmp_path)
    monkeypatch.setattr(run.subprocess, 'check_output', lambda *a, **k: 'JobState=RUNNING State=RUNNING gpu:1')
    monkeypatch.setattr(runtime.jax, 'devices', lambda: [SimpleNamespace(platform='gpu')])
    runtime.verify_gpu(tmp_path)
    assert (tmp_path / 'environment_12_0.json').exists()


def test_runtime_rejects_disabled_fp64(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.jax, 'config', SimpleNamespace(x64_enabled=False))
    with pytest.raises(RuntimeError, match='FP64'):
        runtime.verify_gpu(tmp_path, 'modal')


def test_modal_requires_remote_input_gpu_and_provenance(monkeypatch, tmp_path):
    modal = SimpleNamespace(is_local=lambda: True, current_function_call_id=lambda: 'fc-test',
                            current_input_id=lambda: 'in-test')
    monkeypatch.setitem(sys.modules, 'modal', modal)
    monkeypatch.setenv('MODAL_TASK_ID', 'ta-test')
    with pytest.raises(RuntimeError, match='genuine remote'):
        runtime.verify_gpu(tmp_path, 'modal')
    modal.is_local = lambda: False
    monkeypatch.setattr(runtime.jax, 'devices', lambda: [SimpleNamespace(platform='cpu')])
    with pytest.raises(RuntimeError, match='exactly one Modal GPU'):
        runtime.verify_gpu(tmp_path, 'modal')
    gpu = SimpleNamespace(platform='gpu', device_kind='test GPU')
    monkeypatch.setattr(runtime.jax, 'devices', lambda: [gpu, gpu])
    with pytest.raises(RuntimeError, match='exactly one Modal GPU'):
        runtime.verify_gpu(tmp_path, 'modal')
    monkeypatch.setattr(runtime.jax, 'devices', lambda: [gpu])
    monkeypatch.delenv('RACE_MODAL_IMAGE_ID', raising=False)
    with pytest.raises(RuntimeError, match='provenance'):
        runtime.verify_gpu(tmp_path, 'modal')
    monkeypatch.setenv('RACE_MODAL_IMAGE_ID', 'im-test')
    monkeypatch.setenv('RACE_SOURCE_COMMIT', 'source-test')
    runtime.verify_gpu(tmp_path, 'modal')
    assert (tmp_path / 'environment_ta-test_in-test.json').exists()


def test_modal_checkpoint_requires_mounted_volume(monkeypatch):
    monkeypatch.setattr(runtime, 'checkpoint_volume', None)
    runtime.commit_checkpoint('slurm')
    with pytest.raises(RuntimeError, match='Volume'):
        runtime.commit_checkpoint('modal')
    committed = []
    monkeypatch.setattr(runtime, 'checkpoint_volume', SimpleNamespace(commit=lambda: committed.append(True)))
    runtime.commit_checkpoint('modal')
    assert committed == [True]
