"""Bounded two-H100 execution of the queued 0f3385d plateau protocol.

Use Modal 1.5.5 and the kinematic-pretrain profile. See modal_campaign.md.
All JAX imports and numerical work occur inside remote functions.
"""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import modal

APP_NAME = 'd34-force-plateaus'
INPUT_VOLUME = 'd34-plateau-inputs-0f3385d'
OUTPUT_VOLUME = 'd34-plateau-outputs'
SOURCE = Path(os.environ.get('D34_MODAL_SOURCE_DIR', '/code'))
if modal.is_local() and not (SOURCE / 'source_lock.json').exists():
    raise RuntimeError('Run scripts/prepare_d34_modal.py and set D34_MODAL_SOURCE_DIR first')
INPUT = Path('/inputs')
OUTPUT = Path('/outputs')
NUMERICAL_COMMIT = '0f3385d'
# Upper bounds, at the published 2026-09-22 rates. CPU/memory have hard limits.
GPU_SECOND_USD = .001097 + 4 * .0000131 + 48 * .00000222
CPU_RESERVE_USD = 4.0
GPU_CEILING_SECONDS = 36000
SPENDING_STOP_USD = 50.0

app = modal.App(APP_NAME)
inputs = modal.Volume.from_name(INPUT_VOLUME, create_if_missing=True)
outputs = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)
ledger = modal.Dict.from_name('d34-plateau-attempts', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.4.1+cpu', index_url='https://download.pytorch.org/whl/cpu')
    .pip_install('jax[cuda12]==0.10.2', 'numpy==2.4.6', 'scipy==1.17.1',
                 'ml_dtypes==0.6.0', 'opt_einsum==3.4.0', 'optax==0.2.8',
                 'mpmath==1.3.0', 'PyYAML==6.0.3', 'threadpoolctl==3.7.0',
                 'matplotlib==3.11.2', 'pytest==9.1.1')
    .pip_install('nvidia-nccl-cu12==2.31.2', 'pyparsing==3.3.2')
    .env(dict(JAX_ENABLE_X64='true', XLA_PYTHON_CLIENT_PREALLOCATE='false',
              OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONPATH='/code',
              MPLBACKEND='Agg', RACE_SOURCE_COMMIT=NUMERICAL_COMMIT))
    .add_local_dir(SOURCE, '/code', copy=True))
mounts = {'/inputs': inputs.with_mount_options(read_only=True), '/outputs': outputs}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_inputs():
    hashes = json.loads((INPUT / 'hashes.json').read_text())
    for name, expected in hashes.items():
        if sha256(INPUT / name) != expected:
            raise ValueError(f'Input hash mismatch: {name}')
    return hashes


def verify_source():
    lock = json.loads(Path('/code/source_lock.json').read_text())
    for name, expected in lock['files'].items():
        if sha256(Path('/code') / name) != expected:
            raise ValueError(f'Source hash mismatch: {name}')
    return lock


def protocol_jobs():
    """Exactly the queued run matrix and allocation caps, in GPU seconds."""
    jobs = [dict(id=f'long-{opt}', stage='long', optimizer=opt, seconds=3600)
            for opt in ('gd', 'adam')]
    jobs += [dict(id=f'discovery-{seed}-{start}', stage='discovery', seed=seed,
                  start=start, seconds=1800) for seed in range(3) for start in (100000, 600000)]
    jobs += [dict(id=f'checks-{kind}-{start}', stage='checks', seed=0, start=start,
                  kind=kind, seconds=1200) for kind in ('half', 'grid') for start in (100000, 600000)]
    jobs += [dict(id=f'confirm-{seed}', stage='confirm', seed=seed, seconds=720)
             for seed in range(20, 25)]
    jobs += [dict(id=f'confirm-probes-{seed}', stage='confirm-probes', seed=seed,
                  start=600000, seconds=720) for seed in range(20, 25)]
    return jobs


def reserve(state, job, now):
    """Charge the full cap, including startup, failure, restart and teardown."""
    seconds = job['seconds']
    total = state['reserved_gpu_seconds'] + seconds
    if total > GPU_CEILING_SECONDS or total * GPU_SECOND_USD + CPU_RESERVE_USD > SPENDING_STOP_USD:
        raise RuntimeError('Campaign GPU-hour or spending stop reached')
    if now + seconds > state['deadline']:
        raise RuntimeError('Insufficient time before independent campaign stop')
    state['reserved_gpu_seconds'] = total
    return dict(job, submitted=now, deadline=now + seconds - 30,
                reservation_seconds=seconds, status='reserved')


def run_training(root, job, deadline):
    from experiments.expD34_readout_race import plateau_run as pr, plateau_probes as pp
    remaining = deadline - time.time() - 60
    if remaining <= 0:
        raise RuntimeError('No training time remains after startup')
    if job['stage'] in ('long', 'confirm'):
        confirm = job['stage'] == 'confirm'
        args = SimpleNamespace(runtime='modal', root=INPUT,
            output=root / ('confirm' if confirm else 'long'),
            optimizer='confirm' if confirm else job['optimizer'],
            confirm_seed=job.get('seed'), end_step=1100000 if confirm else 6000000,
            max_seconds=min(620 if confirm else 3300, remaining))
        pr.run(args)
        return args.output / (f"primary_{job['seed']}" if confirm else job['optimizer'])
    stage = job['stage']
    name = (f"{job['kind']}_{job['start']}" if stage == 'checks'
            else f"seed{job['seed']}_{job['start']}")
    args = SimpleNamespace(runtime='modal',
        source=root / 'confirm' if stage == 'confirm-probes' else INPUT / 'raw',
        output=root / ('probes' if stage == 'discovery' else stage) / name,
        seed=job['seed'], start=job['start'], horizon=500000,
        half=job.get('kind') == 'half', samples=4096 if job.get('kind') == 'grid' else 2048,
        max_seconds=min({'discovery': 1650, 'confirm-probes': 620, 'checks': 1100}[stage], remaining))
    pp.run(args)
    return args.output


@app.function(image=image, gpu='H100', max_containers=2, min_containers=0,
              cpu=(4, 4), memory=(49152, 49152), retries=0, timeout=3600,
              startup_timeout=120, single_use_containers=True, volumes=mounts)
def gpu_worker(campaign, attempt):
    """The sole GPU function; inputs are serial and own disjoint output bundles."""
    os.chdir('/code')
    os.environ['JAX_PLATFORMS'] = 'cuda'
    os.environ['RACE_MODAL_IMAGE_ID'] = image.object_id
    key = f"{campaign}/attempt/{attempt['id']}"
    # A platform restart may rerun an input despite retries=0. Fail closed:
    # never issue a new deadline or execute a second copy of an output owner.
    owner = dict(container=os.environ.get('MODAL_TASK_ID'), input=modal.current_input_id(),
                 call=modal.current_function_call_id(), started=time.time())
    if not ledger.put(key + '/owner', owner, skip_if_exists=True):
        raise RuntimeError('Interrupted or duplicate attempt; accounting requires review')
    if ledger.get(campaign + '/stop', False) or time.time() >= attempt['deadline']:
        raise RuntimeError('Original attempt deadline expired or campaign stopped')
    inputs.reload()
    outputs.reload()
    verify_inputs()
    lock = verify_source()
    root = OUTPUT / campaign
    logdir = root / 'attempts' / attempt['id']
    logdir.mkdir(parents=True, exist_ok=True)
    from experiments.expD34_readout_race import plateau_runtime
    plateau_runtime.checkpoint_volume = outputs
    plateau_runtime.verify_gpu(logdir, 'modal')
    write_json(logdir / 'attempt.json', dict(attempt, **owner, source=lock,
               image=image.object_id))
    outputs.commit()

    def expired(signum, frame):
        raise RuntimeError('Original attempt deadline reached')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.01, attempt['deadline'] - time.time()))
    try:
        if attempt['stage'] == 'pilot':
            from experiments.expD34_readout_race.modal_pilot import run
            result = run(root, INPUT, attempt['optimizer'], outputs, attempt['deadline'])
        else:
            folder = run_training(root, attempt, attempt['deadline'])
            result = dict(bundle=str(folder.relative_to(root)),
                          scientific_status=json.loads((folder / 'status.json').read_text()))
        result.update(finished=time.time(), container=owner['container'], passed=True)
        write_json(logdir / 'result.json', result)
        outputs.commit()
        return result
    except BaseException as exc:
        write_json(logdir / 'failure.json', dict(error=repr(exc), time=time.time()))
        outputs.commit()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@app.function(image=image, cpu=(.125, .125), memory=(256, 256), retries=0,
              timeout=21600, max_containers=1, single_use_containers=True)
def watchdog(campaign, phase, deadline):
    """Independent CPU guard, including per-attempt deadlines and cancellations."""
    while not ledger.get(f'{campaign}/{phase}/finished', False):
        state = ledger.get(f'{campaign}/{phase}/state', {})
        stop = time.time() >= deadline or ledger.get(campaign + '/stop', False)
        for attempt in state.get('attempts', []):
            if attempt.get('call') and (stop or time.time() >= attempt['deadline']):
                modal.FunctionCall.from_id(attempt['call']).cancel(terminate_containers=True)
        if stop:
            ledger[campaign + '/stop'] = True
            return
        time.sleep(2)


def persist(campaign, phase, state):
    ledger[f'{campaign}/{phase}/state'] = state
    write_json(OUTPUT / campaign / f'{phase}_accounting.json', state)
    outputs.commit()


def execute(campaign, phase, state, jobs):
    from experiments.expD34_readout_race import plateau_gate
    pending = list(jobs)
    active = {}
    gates = {}
    root = OUTPUT / campaign
    while pending or active:
        if time.time() >= state['deadline'] or ledger.get(campaign + '/stop', False):
            raise TimeoutError('Independent campaign stop reached')
        for stage, required in [('discovery', 'confirm'), ('confirm', 'confirm-probes')]:
            if any(j['stage'] == required for j in pending) and stage not in gates:
                predecessors = [a for a in state['attempts'] if a['stage'] == stage]
                expected = 6 if stage == 'discovery' else 5
                if len(predecessors) == expected and all(a['status'] == 'returned' for a in predecessors):
                    outputs.reload()
                    try:
                        plateau_gate.check(root, stage)
                        gates[stage] = 'passed'
                    except (ValueError, FileNotFoundError) as exc:
                        gates[stage] = repr(exc)
                        blocked = ('confirm', 'confirm-probes') if stage == 'discovery' else ('confirm-probes',)
                        state.setdefault('blocked', []).extend(j for j in pending if j['stage'] in blocked)
                        pending = [j for j in pending if j['stage'] not in blocked]
                    state['gates'] = gates
                    persist(campaign, phase, state)
        ready = [j for j in pending if j['stage'] not in ('confirm', 'confirm-probes')
                 or gates.get('discovery' if j['stage'] == 'confirm' else 'confirm') == 'passed']
        while ready and len(active) < 2:
            job = ready.pop(0)
            pending.remove(job)
            attempt = reserve(state, job, time.time())
            state['attempts'].append(attempt)
            persist(campaign, phase, state)
            call = gpu_worker.spawn(campaign, attempt)
            attempt['call'] = call.object_id
            active[job['id']] = (call, attempt)
            persist(campaign, phase, state)
        for name, (call, attempt) in list(active.items()):
            if time.time() >= attempt['deadline']:
                raise RuntimeError(f"Attempt deadline reached: {name}")
            try:
                result = call.get(timeout=0)
            except TimeoutError:
                continue
            attempt.update(status='returned', result=result, returned=time.time())
            del active[name]
            persist(campaign, phase, state)
        if active:
            time.sleep(2)
    return state


@app.function(image=image, cpu=(4, 4), memory=(24576, 24576), retries=0,
              timeout=21600, max_containers=1, single_use_containers=True, volumes=mounts)
def coordinator(campaign, phase):
    os.chdir('/code')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    inputs.reload()
    outputs.reload()
    verify_inputs()
    lock = verify_source()
    root = OUTPUT / campaign
    root.mkdir(parents=True, exist_ok=True)
    owner_key = f'{campaign}/{phase}/owner'
    if phase == 'verify':
        owner_key += '/' + image.object_id
    if not ledger.put(owner_key, modal.current_function_call_id(), skip_if_exists=True):
        raise RuntimeError('This phase already has an owner; inspect its original ledger')
    if phase == 'verify':
        verification = root / 'verification' / image.object_id
        verification.mkdir(parents=True, exist_ok=True)
        try:
            with (verification / 'packages.txt').open('w') as packages:
                subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=packages, check=True)
            with (verification / 'cpu_tests.log').open('w') as log:
                subprocess.run([sys.executable, '-m', 'pytest', '-q', *lock['tests']],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        finally:
            outputs.commit()
        write_json(root / 'verification.json', dict(passed=True, image=image.object_id,
                   source=lock, time=time.time()))
        outputs.commit()
        ledger[campaign + '/verified'] = image.object_id
        return dict(passed=True)
    if ledger.get(campaign + '/verified') != image.object_id:
        raise RuntimeError('Remote CPU checks must pass on this exact image first')
    cutover = json.loads((INPUT / 'cutover.json').read_text())
    if cutover['jobs'] != [1025, *range(1036, 1044)] or cutover['gpu_seconds'] != 0:
        raise RuntimeError('Runpod accounting needs explicit reconciliation')
    previous = ledger.get(campaign + '/pilot/state', {})
    if phase == 'pilot':
        if cutover['state'] != 'held':
            raise RuntimeError('Hold the pending Runpod campaign before the pilot')
        jobs = [dict(id=f'pilot-{opt}', stage='pilot', optimizer=opt, seconds=450)
                for opt in ('gd', 'adam')]
        deadline = time.time() + 450
        # Reservation needs room for the two sequential submission RPCs.
        deadline += 10
    elif phase == 'campaign':
        if cutover['state'] != 'cancelled' or not ledger.get(campaign + '/pilot/passed', False):
            raise RuntimeError('Pilot must pass and Runpod campaign must be cancelled')
        jobs = protocol_jobs()
        deadline = time.time() + (GPU_CEILING_SECONDS - previous['reserved_gpu_seconds']) / 2 - 60
    else:
        raise ValueError(phase)
    state = dict(phase=phase, deadline=deadline, source=lock, started=time.time(),
                 attempts=[], reserved_gpu_seconds=previous.get('reserved_gpu_seconds', 0),
                 gpu_ceiling_seconds=GPU_CEILING_SECONDS, spending_stop_usd=SPENDING_STOP_USD,
                 gpu_second_usd=GPU_SECOND_USD, cpu_reserve_usd=CPU_RESERVE_USD,
                 accounting='full submission-to-deadline reservations; no reclaimed time', runpod=cutover)
    persist(campaign, phase, state)
    guard = watchdog.spawn(campaign, phase, deadline)
    state['watchdog_call'] = guard.object_id
    persist(campaign, phase, state)
    try:
        execute(campaign, phase, state, jobs)
        if phase == 'pilot':
            if len({a['result']['container'] for a in state['attempts']}) != 2:
                raise RuntimeError('Pilot did not execute on two independent containers')
            ledger[campaign + '/pilot/passed'] = True
        else:
            outputs.reload()
            from experiments.expD34_readout_race import plateau_results
            plateau_results.run(root, root / 'endpoint-audit')
        state['status'] = 'finished'
        return state
    except BaseException as exc:
        state['status'] = 'stopped'
        state['error'] = repr(exc)
        ledger[campaign + '/stop'] = True
        for attempt in state['attempts']:
            if attempt.get('call'):
                modal.FunctionCall.from_id(attempt['call']).cancel(terminate_containers=True)
        raise
    finally:
        persist(campaign, phase, state)
        ledger[f'{campaign}/{phase}/finished'] = True
        guard.cancel(terminate_containers=True)


@app.local_entrypoint()
def main(phase: str = 'verify', campaign: str = 'modal-20260922', input_dir: str = ''):
    """Local work is transfer/submission only; use modal run --detach for campaigns."""
    if input_dir:
        with inputs.batch_upload(force=True) as batch:
            batch.put_directory(input_dir, '/')
    print(coordinator.remote(campaign, phase))
