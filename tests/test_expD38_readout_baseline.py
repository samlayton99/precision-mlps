"""Scientific controls: paired readouts and a genuinely observational LS solve."""
import numpy as np
import torch

from experiments.expD38_init_readout_baseline.run import (
    affine_fit, config, make_model, predictions, same_identity, valid_data_identity,
)


def test_original_sarcos_protocol_cannot_be_reused():
    cfg = config()
    old = {"task": "sarcos", "seed": 0, "steps": 20000, "config": cfg}
    new = {**old, "data_protocol": cfg["task_data_protocols"]["sarcos"]}
    assert not same_identity(old, new)
    assert not valid_data_identity(old, cfg)
    assert valid_data_identity(new, cfg)
    assert valid_data_identity({"task": "airfoil"}, cfg)


def test_sarcos_ignores_duplicate_supplied_test_and_has_disjoint_splits(tmp_path, monkeypatch):
    from scipy.io import savemat
    from experiments.expD38_init_readout_baseline import run
    folder = tmp_path / "data/sarcos"
    folder.mkdir(parents=True)
    raw = np.random.default_rng(82).normal(size=(100, 28))
    savemat(folder / "sarcos_inv.mat", {"sarcos_inv": raw})
    savemat(folder / "sarcos_inv_test.mat", {"sarcos_inv_test": raw[:10]})
    monkeypatch.setattr(run, "ROOT", tmp_path)
    arrays, metadata = run.load_data("sarcos", config())
    rows = {split: set(map(tuple, x)) for split, (x, _) in arrays.items()}
    assert [len(rows[s]) for s in ("train", "val", "test")] == [64, 16, 20]
    assert rows["train"].isdisjoint(rows["val"] | rows["test"])
    assert rows["val"].isdisjoint(rows["test"])
    assert set(metadata["source_sha256"]) == {"data/sarcos/sarcos_inv.mat"}


def test_adding_seeds_reuses_runs_but_recipe_or_budget_changes_do_not():
    import copy
    a = {"task": "airfoil", "seed": 0, "steps": 20000, "config": config()}
    b = copy.deepcopy(a)
    b["config"]["comparison_seeds"] = [0, 1, 2, 3]
    assert same_identity(a, b)
    b["steps"] = 100000
    assert not same_identity(a, b)
    b["steps"] = 20000
    b["config"]["batch_size"] = 128
    assert not same_identity(a, b)


def test_affine_solver_handles_rank_deficiency_and_intercept():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(80, 4))
    x[:, 3] = x[:, 0]
    y = 3 + x @ np.array([[1.], [-2.], [.5], [1.]])
    w, b, info = affine_fit(x, y)
    np.testing.assert_allclose(x @ w + b, y, atol=1e-12)
    assert info["rank"] == 3


def test_paired_initializations_have_identical_readout():
    cfg = config()
    x = torch.randn(90, 5, dtype=torch.float64)
    a, _ = make_model(5, 32, 7, "standard", x, cfg)
    b, _ = make_model(5, 32, 7, "qi", x, cfg)
    assert torch.equal(a.fc3.weight, b.fc3.weight)
    assert torch.equal(a.fc3.bias, b.fc3.bias)
    assert not torch.equal(a.fc1.weight, b.fc1.weight)
    assert not torch.equal(a.fc2.weight, b.fc2.weight)
    assert all(p.requires_grad for p in b.parameters())


def test_readout_diagnostic_cannot_change_next_gradient_update():
    cfg = config()
    torch.manual_seed(12)
    x = torch.randn(90, 5, dtype=torch.float64)
    y = torch.sin(x[:, :1]) + x[:, 1:2] ** 2
    a, _ = make_model(5, 32, 8, "standard", x, cfg)
    b, _ = make_model(5, 32, 8, "standard", x, cfg)
    oa, ob = torch.optim.Adam(a.parameters()), torch.optim.Adam(b.parameters())
    for model, optimizer in ((a, oa), (b, ob)):
        ((model(x) - y) ** 2).mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    rng_before = torch.get_rng_state().clone()
    pred, hidden = predictions(a, {"train": x}, with_features=True)
    w, bias, _ = affine_fit(hidden["train"], y.numpy())
    solved_error = np.mean((hidden["train"] @ w + bias - y.numpy()) ** 2)
    assert solved_error <= np.mean((pred["train"] - y.numpy()) ** 2) + 1e-12
    assert torch.equal(rng_before, torch.get_rng_state())
    for model, optimizer in ((a, oa), (b, ob)):
        ((model(x) - y) ** 2).mean().backward()
        optimizer.step()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
