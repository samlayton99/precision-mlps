import pytest
from types import SimpleNamespace

from experiments.expD34_readout_race import modal_campaign as campaign, plateau_gate
from experiments.expD34_readout_race.modal_campaign import protocol_jobs, reserve


def test_matrix_preserves_queued_allocations_and_cases():
    jobs = protocol_jobs()
    assert len(jobs) == 22
    assert len({j['id'] for j in jobs}) == len(jobs)
    assert sum(j['seconds'] for j in jobs) == 30000
    assert {j['seed'] for j in jobs if j['stage'] == 'confirm'} == set(range(20, 25))
    assert {(j['seed'], j['start']) for j in jobs if j['stage'] == 'discovery'} == {
        (s, t) for s in range(3) for t in (100000, 600000)}


def test_reservations_include_startup_and_do_not_extend_deadlines():
    state = dict(reserved_gpu_seconds=900, deadline=20000)
    attempt = reserve(state, dict(id='long-gd', seconds=3600), 100)
    assert attempt['deadline'] == 3670
    assert attempt['reservation_seconds'] == 3600
    assert state['reserved_gpu_seconds'] == 4500
    assert reserve(state, dict(id='long-adam', seconds=3600), 105)['deadline'] == 3675
    state['reserved_gpu_seconds'] = 35999
    with pytest.raises(RuntimeError, match='stop'):
        reserve(state, dict(seconds=10), 100)
    state['reserved_gpu_seconds'] = 0
    with pytest.raises(RuntimeError, match='independent campaign stop'):
        reserve(state, dict(seconds=100), 19901)


@pytest.mark.parametrize('incomplete', [False, True])
def test_scheduler_pool_and_completion_dependencies(monkeypatch, incomplete):
    running = set()
    completed = []
    gates = []
    peak = 0
    clock = [0.]

    class Call:
        def __init__(self, job):
            self.job = job
            self.object_id = job['id']
            self.polled = False

        def get(self, timeout):
            if not self.polled:
                self.polled = True
                raise TimeoutError()
            running.remove(self.object_id)
            completed.append(self.job)
            return dict(passed=True)

    def spawn(name, job):
        nonlocal peak
        if job['stage'] == 'confirm':
            assert 'discovery' in gates
        if job['stage'] == 'confirm-probes':
            assert 'confirm' in gates
        running.add(job['id'])
        peak = max(peak, len(running))
        return Call(job)

    def gate(root, stage):
        assert sum(j['stage'] == stage for j in completed) == (6 if stage == 'discovery' else 5)
        if incomplete:
            raise ValueError('Incomplete source bundle')
        gates.append(stage)

    monkeypatch.setattr(campaign, 'gpu_worker', SimpleNamespace(spawn=spawn))
    monkeypatch.setattr(campaign, 'ledger', {})
    monkeypatch.setattr(campaign, 'outputs', SimpleNamespace(reload=lambda: None))
    monkeypatch.setattr(campaign, 'persist', lambda *args: None)
    monkeypatch.setattr(plateau_gate, 'check', gate)
    monkeypatch.setattr(campaign.time, 'time', lambda: clock[0])
    monkeypatch.setattr(campaign.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    state = dict(deadline=20000, attempts=[], reserved_gpu_seconds=900)
    campaign.execute('test', 'campaign', state, protocol_jobs())
    assert peak == 2 and not running
    assert len(completed) == (12 if incomplete else 22)
    assert state['reserved_gpu_seconds'] == (23700 if incomplete else 30900)
    assert gates == ([] if incomplete else ['discovery', 'confirm'])
