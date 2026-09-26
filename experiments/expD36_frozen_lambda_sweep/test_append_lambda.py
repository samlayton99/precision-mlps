from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import append_lambda
import run


def test_single_explicit_lambda_and_appended_grid():
    cfg = run.configuration()
    cfg.update(n=64, halo_per_side=2, n_train=131, n_eval=263, lambda_values=[1.0])
    p = run.make_problem(cfg)
    np.testing.assert_array_equal(p.lambdas, [1.0])
    np.testing.assert_array_equal(p.gammas, [32.0])


def test_append_preserves_every_old_case_and_native_state():
    def fixture(k, offset):
        e = np.arange(3*2*k*4, dtype=float).reshape(3, 2, k, 4)+offset
        state = np.arange(k*5*8, dtype=float).reshape(k, 5, 8)+offset
        return {"step": np.asarray(2), "steps": np.arange(3), "coefficient_steps": np.arange(3),
                "c": state, "adam_m": state[:, :, :4], "adam_v": state[:, :, :4]**2,
                "train_rel_l2": e, "eval_rel_l2": e,
                "coefficient_snapshots": np.stack([state]*3),
                "best_recorded_index": e.argmin(axis=0), "config_sha256": np.asarray("old")}
    old, new = fixture(2, 100), fixture(1, 500)
    # Insert a middle value to also verify sorting and case-axis bookkeeping.
    cfg = dict(run.configuration(), additional_lambdas=[.3])
    result, order, positions = append_lambda.merge_trajectory(old, new, [.1, .5], [.3], cfg)
    np.testing.assert_array_equal(order, [0, 2, 1])
    for key, axis in append_lambda.CASE_AXES.items():
        if key in old:
            np.testing.assert_array_equal(np.take(result[key], positions, axis=axis), old[key])
    assert str(result["config_sha256"]) == run.fingerprint(cfg)
    np.testing.assert_array_equal(result["physical_c"], result["c"])
