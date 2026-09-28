"""Execution checks for the plateau campaign; numerical kernels are unchanged."""
import os
from pathlib import Path

import jax
import numpy as np

from .run import source_hashes, verify_gpu as verify_slurm_gpu, write_json

# Set only by the remote Modal worker after mounting its output Volume.
checkpoint_volume = None


def verify_gpu(output, runtime="slurm"):
    if not jax.config.x64_enabled:
        raise RuntimeError("FP64 is required")
    if runtime == "slurm":
        return verify_slurm_gpu(output)
    if runtime != "modal":
        raise ValueError(f"Unknown runtime: {runtime}")
    import modal

    call_id = modal.current_function_call_id()
    input_id = modal.current_input_id()
    container = os.environ.get("MODAL_TASK_ID")
    if modal.is_local() or not call_id or not input_id or not container:
        raise RuntimeError("Modal GPU work requires a genuine remote function input")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError(f"Expected exactly one Modal GPU; saw {devices}")
    if not os.environ.get("RACE_MODAL_IMAGE_ID") or not os.environ.get("RACE_SOURCE_COMMIT"):
        raise RuntimeError("Missing Modal image/source provenance")
    write_json(Path(output) / f"environment_{container}_{input_id}.json", dict(
        runtime=runtime, container=container, call_id=call_id, input_id=input_id,
        image=os.environ["RACE_MODAL_IMAGE_ID"], source_commit=os.environ["RACE_SOURCE_COMMIT"],
        numerical_commit="0f3385d", devices=[str(d) for d in devices],
        device_kind=devices[0].device_kind, fp64=True, jax=jax.__version__,
        numpy=np.__version__, sources=source_hashes()))


def commit_checkpoint(runtime):
    """Publish a complete bundle before a dependent container can consume it."""
    if runtime == "modal":
        if checkpoint_volume is None:
            raise RuntimeError("Modal checkpoint Volume was not supplied by the worker")
        checkpoint_volume.commit()
