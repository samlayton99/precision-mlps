"""Lifecycle tests for recording barriers, replay branching and local requests."""
import copy
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from experiments.expG02_geometry_reader_codex.engine import GeometryEngine
from experiments.expG02_geometry_reader_codex.server import (
    Controller, RecordingStore, DEFAULT_SETTINGS, make_server, snapshot_steps, validate_settings,
)


@pytest.fixture
def controller(tmp_path):
    engine = GeometryEngine({"n_centers": 12, "n_train": 24, "n_test": 31,
                             "n_intervals": 8, "halo": 1, "reference_h": .25,
                             "center_min": -1.25, "center_max": 1.25})
    instance = Controller(run_root=tmp_path / "runs", engine=engine)
    yield instance
    instance.close()


def wait_for(condition, timeout=5):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for training state.")
        time.sleep(.005)


def test_update_restores_pending_preview_and_exact_discard_baseline(controller):
    baseline = controller.state()
    preview = controller.edit("center", 2, .321)
    controller.discard()
    controller.restore_session({"preview": preview, "committed": baseline})
    restored = controller.state()
    assert restored["status"]["pending"]["geometry"]
    assert restored["frame"]["state"] == preview["frame"]["state"]
    discarded = controller.discard()
    assert discarded["frame"]["state"] == baseline["frame"]["state"]


def finish(controller, steps=7, every=2):
    controller.train({"steps": steps, "snapshot_every": every, "delay_ms": 0})
    wait_for(lambda: controller.state()["status"]["mode"] != "training")
    state = controller.state()
    assert state["status"]["error"] is None
    return state


def test_fixed_and_geometric_schedules_include_exact_large_endpoint():
    assert list(snapshot_steps(7, 2)) == [0, 2, 4, 6, 7]
    assert list(snapshot_steps(10, 1, "geometric", 2)) == [0, 1, 3, 7, 10]
    schedule = list(snapshot_steps(50_000, 1, "geometric", 1.3))
    assert schedule[0] == 0 and schedule[-1] == 50_000
    assert len(schedule) < 50
    assert all(a < b for a, b in zip(schedule, schedule[1:]))
    assert len(list(snapshot_steps(50_000, 1))) == 50_001


@pytest.mark.parametrize("settings", [
    {"steps": 0}, {"steps": 1.5}, {"snapshot_every": 0}, {"delay_ms": -1},
    {"growth": float("nan")}, {"growth": .9}, {"schedule": "unknown"}, {"bogus": 2},
])
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        validate_settings(settings)


def test_snapshot_barrier_persists_both_readouts_and_post_feedback_state(controller):
    controller.configure({"ls_feedback": True})
    state = finish(controller)
    assert [entry["step"] for entry in state["timeline"]] == [0, 2, 4, 6, 7]
    assert state["status"]["finished"]
    initial = controller.store.frame(controller.run, 0)
    assert initial["adam_readout"] == [0.0] * 12
    assert not initial["feedback_applied"]
    saved = controller.store.frame(controller.run, 1)
    assert saved["feedback_applied"]
    np.testing.assert_array_equal(saved["state"]["params"]["v"], saved["solved_readout"])
    assert saved["adam_readout"] != saved["solved_readout"]
    assert saved["state"]["step_count"] == saved["step"] == 2
    assert saved["state"]["optimizer_step"] == 2
    assert "m" in saved["state"] and "rng" in saved["state"]


def test_seek_uses_exact_persisted_curves_without_solving(controller, monkeypatch):
    finish(controller)
    expected = controller.store.frame(controller.run, 1)
    run_id = controller.run["id"]
    monkeypatch.setattr(controller.engine, "evaluate", lambda **kwargs: pytest.fail("Replay re-solved a frame."))
    selected = controller.select(1, run_id)
    assert selected["frame"] == expected
    assert selected["status"]["mode"] == "replay"
    assert controller.engine.state_dict() == expected["state"]
    # New server instances see the durable run library and can load exact frames.
    reopened = RecordingStore(controller.store.root)
    assert reopened.frame(reopened.manifest(run_id), 1) == expected


def test_manual_edit_during_training_is_recorded_and_training_continues(controller):
    controller.train({"steps": 100, "snapshot_every": 1, "delay_ms": 200})
    wait_for(lambda: controller.state()["status"]["step"] >= 1)
    result = controller.edit("center", 2, -.37)
    assert result["status"]["mode"] == "training"
    assert result["frame"]["event"] == "manual"
    assert result["frame"]["centers"][2] == pytest.approx(-.37)
    assert result["frame"]["changes"][0]["kind"] == "center"
    step = result["frame"]["step"]
    wait_for(lambda: controller.state()["status"]["step"] > step)
    controller.pause()
    assert any(item["event"] == "manual" for item in controller.timeline)
    assert controller.mode == "paused"


def test_manual_readout_copies_displayed_vector_and_survives_next_step(controller):
    baseline = controller.state()["frame"]["solved_readout"]
    result = controller.edit("readout", 3, .123, "solved")
    expected = np.asarray(baseline).copy()
    expected[3] = .123
    np.testing.assert_array_equal(result["frame"]["adam_readout"], expected)
    assert result["frame"]["pending_readout_edit"]
    assert result["status"]["pending"] == {"geometry": False, "readout": True}
    assert not result["frame"]["feedback_applied"]
    np.testing.assert_array_equal(controller.engine.params["v"], expected)
    blocked = controller.train({"steps": 1, "snapshot_every": 1, "delay_ms": 0})
    assert blocked["status"]["requires_resolution"]
    assert blocked["status"]["mode"] == "idle"
    controller.train({"steps": 1, "snapshot_every": 1, "delay_ms": 0}, resolution="inject")
    wait_for(lambda: controller.mode == "paused")
    initial = controller.store.frame(controller.run, 0)
    np.testing.assert_array_equal(initial["adam_readout"], expected)
    assert initial["pending_readout_edit"]
    assert not controller.frame["pending_readout_edit"]


def test_replay_edit_forks_and_never_changes_original_recording(controller):
    finish(controller)
    old_id = controller.run["id"]
    old_manifest = controller.store.manifest(old_id)
    old_files = {path.name: path.read_bytes() for path in controller.store.directory(old_id).iterdir()}
    controller.select(1)
    before = controller.frame["adam_readout"]
    result = controller.edit("readout", 4, .321, "adam")
    assert result["status"]["mode"] == "idle"
    assert result["status"]["run_id"] is None
    assert result["status"]["pending"]["readout"]
    assert controller.store.manifest(old_id) == old_manifest
    result = controller.inject("readout", "adam")
    assert result["status"]["mode"] == "paused"
    assert result["status"]["run_id"] != old_id
    assert controller.run["parent"] == {"id": old_id, "frame": 1}
    assert [entry["event"] for entry in result["timeline"]] == ["branch_start", "manual"]
    expected = np.asarray(before).copy()
    expected[4] = .321
    np.testing.assert_array_equal(result["frame"]["adam_readout"], expected)
    assert controller.store.manifest(old_id) == old_manifest
    assert {path.name: path.read_bytes() for path in controller.store.directory(old_id).iterdir()} == old_files


def test_pause_between_sparse_snapshots_records_current_state_and_resumes(controller, monkeypatch):
    original_step = controller.engine.step

    def slow_step(count=1):
        result = original_step(count)
        time.sleep(.002)
        return result

    monkeypatch.setattr(controller.engine, "step", slow_step)
    controller.train({"steps": 100, "snapshot_every": 100, "delay_ms": 0})
    wait_for(lambda: controller.engine.step_count >= 3)
    paused = controller.pause()
    assert 3 <= paused["status"]["step"] < 100
    assert paused["frame"]["step"] == controller.engine.step_count
    assert paused["frame"]["event"] == "pause"
    moments = copy.deepcopy(controller.engine.state_dict()["m"])
    optimizer_step = controller.engine.optimizer_step
    controller.train()
    controller.pause()
    assert controller.engine.optimizer_step >= optimizer_step
    if controller.engine.optimizer_step == optimizer_step:
        assert controller.engine.state_dict()["m"] == moments
    assert controller._target_step == 100


def test_panel_config_stops_training_and_invalid_config_rolls_back(controller):
    controller.train({"steps": 100, "snapshot_every": 1, "delay_ms": 200})
    wait_for(lambda: controller.engine.step_count >= 1)
    state = controller.configure({"target": "cos(2*pi*x)"})
    assert state["status"]["mode"] == "idle"
    assert state["status"]["run_id"] is None
    assert len(state["runs"]) == 1
    count = controller.engine.step_count
    time.sleep(.025)
    assert controller.engine.step_count == count
    before = controller.engine.state_dict()
    with pytest.raises(ValueError):
        controller.configure({"target": "sqrt(-1)"})
    assert controller.engine.state_dict() == before


def test_readout_initialization_change_preserves_manual_geometry(controller):
    controller.edit("center", 2, -.17)
    controller.inject("geometry")
    centers = controller.engine.centers.copy()
    controller.configure({"readout_init": "xavier"})
    np.testing.assert_array_equal(controller.engine.centers, centers)
    assert np.linalg.norm(controller.engine.params["v"]) > 0


def test_staged_geometry_discard_restores_exact_state_and_has_no_training_effect(controller):
    before = controller.engine.state_dict()
    frame = copy.deepcopy(controller.frame)
    controller.edit("gamma", 2, 12.0)
    assert controller.state()["status"]["pending"]["geometry"]
    assert controller._committed_state == before
    controller.discard()
    assert controller.engine.state_dict() == before
    assert controller.frame == frame
    assert not any(controller.pending.values())
    controller.edit("center", 2, .7)
    controller.train({"steps": 1, "delay_ms": 0}, resolution="discard")
    wait_for(lambda: controller.mode == "paused")
    initial = controller.store.frame(controller.run, 0)
    assert initial["state"] == before


def test_component_injection_keeps_other_component_staged(controller):
    original_centers = controller.engine.centers.copy()
    original_readout = controller.engine.params["v"].copy()
    controller.edit("center", 2, .72)
    controller.edit("readout", 3, .55, "adam")
    result = controller.inject("geometry")
    assert result["status"]["pending"] == {"geometry": False, "readout": True}
    np.testing.assert_array_equal(controller._committed_state["params"]["v"], original_readout)
    assert result["frame"]["centers"][2] == pytest.approx(.72)
    assert result["frame"]["adam_readout"][3] == .55
    controller.discard()
    assert controller.frame["centers"][2] == pytest.approx(.72)
    assert controller.frame["adam_readout"][3] == 0
    assert controller.frame["centers"] != original_centers.tolist()


def test_panel_config_preserves_pending_edits_and_updates_discard_baseline(controller):
    before_centers = controller.engine.centers.copy()
    controller.edit("center", 2, .73)
    controller.edit("readout", 4, .25)
    result = controller.configure({"lr": .002, "target": "cos(2*pi*x)"})
    assert result["status"]["pending"] == {"geometry": True, "readout": True}
    assert result["frame"]["centers"][2] == pytest.approx(.73)
    assert result["frame"]["adam_readout"][4] == .25
    assert result["frame"]["config"]["lr"] == .002
    controller.discard()
    np.testing.assert_array_equal(controller.engine.centers, before_centers)
    assert controller.engine.params["v"][4] == 0
    assert controller.engine.config["lr"] == .002
    assert controller.engine.config["target"] == "cos(2*pi*x)"


def test_failed_config_preserves_preview_and_committed_state(controller):
    controller.edit("center", 2, .73)
    preview = controller.engine.state_dict()
    committed = copy.deepcopy(controller._committed_state)
    with pytest.raises(ValueError):
        controller.configure({"target": "sqrt(-1)"})
    assert controller.engine.state_dict() == preview
    assert controller._committed_state == committed
    assert controller.pending["geometry"]


def test_readout_only_injection_uses_solved_display_without_geometry_commit(controller):
    original_centers = controller.engine.centers.copy()
    controller.edit("center", 2, .71)
    solved = controller.frame["solved_readout"].copy()
    result = controller.inject("readout", "solved")
    assert result["status"]["pending"] == {"geometry": True, "readout": False}
    np.testing.assert_array_equal(controller._committed_state["params"]["v"], solved)
    controller.discard()
    np.testing.assert_allclose(controller.engine.centers, original_centers, atol=0, rtol=0)
    np.testing.assert_array_equal(controller.engine.params["v"], solved)


def test_resize_preview_injection_is_valid_and_preserves_committed_readout(controller):
    before = controller.engine.state_dict()
    controller.resize(18)
    expected_readout = np.interp(np.linspace(0, 1, 20), np.linspace(0, 1, 12), before["params"]["v"]) * (2/17) / before["config"]["reference_h"]
    assert controller.pending["geometry"]
    with pytest.raises(ValueError, match="Inject geometry first"):
        controller.inject("readout", "solved")
    controller.inject("geometry")
    assert len(controller.engine.params["v"]) == 20
    np.testing.assert_allclose(controller.engine.params["v"], expected_readout)
    assert not any(controller.pending.values())
    assert controller.engine.state_dict() == controller._committed_state


def test_new_recording_and_names_do_not_delete_old_data(controller):
    finish(controller)
    old_id = controller.run["id"]
    controller.save("My sine experiment")
    assert controller.store.manifest(old_id)["name"] == "My sine experiment"
    expected = controller.engine.state_dict()
    controller.new()
    assert controller.engine.state_dict() == expected
    assert controller.run is None
    assert controller.store.manifest(old_id)["frames"] == 5
    assert controller.state(since=controller.state()["version"])["unchanged"]


def test_path_traversal_and_bad_indices_are_rejected(controller):
    with pytest.raises(ValueError):
        controller.store.directory("../../anything")
    finish(controller)
    for index in (-1, .5, 999, True):
        with pytest.raises(ValueError):
            controller.select(index)


def test_http_rejects_cross_origin_and_malformed_mutations(controller, tmp_path):
    (tmp_path / "index.html").write_text("<title>Geometry reader</title>")
    server = make_server(controller, port=0, static_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/state", timeout=3) as response:
            assert json.load(response)["status"]["mode"] == "idle"
        for headers, body, status in (
            ({"Content-Type": "application/json", "Origin": "https://elsewhere.example"}, b"{}", 403),
            ({"Content-Type": "text/plain"}, b"{}", 403),
            ({"Content-Type": "application/json"}, b"not json", 400),
            ({"Content-Type": "application/json"}, b"[]", 400),
            ({"Content-Type": "application/json", "Host": "elsewhere.example"}, b"{}", 403),
        ):
            request = Request(base + "/api/pause", data=body, headers=headers, method="POST")
            with pytest.raises(HTTPError) as caught:
                urlopen(request, timeout=3)
            assert caught.value.code == status
            assert "error" in json.load(caught.value)
        request = Request(base + "/api/save", data=b'{"name":"HTTP run"}',
                          headers={"Content-Type": "application/json", "Origin": base})
        with urlopen(request, timeout=3) as response:
            assert json.load(response)["runs"][0]["name"] == "HTTP run"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_full_reset_restores_defaults_and_keeps_recorded_files(controller):
    finish(controller)
    run_id = controller.run["id"]
    files = {str(p.relative_to(controller.store.root)): p.read_bytes()
             for p in controller.store.directory(run_id).rglob("*") if p.is_file()}
    controller.configure({"target": "exp(x)", "lr": .02, "noise": .01})
    controller.update_settings({"steps": 30, "schedule": "geometric", "growth": 1.8})
    controller.edit("center", 2, .123)
    controller.edit("readout", 3, 7.0)
    controller.train()  # Pending previews require resolution.
    assert controller.state()["status"]["requires_resolution"]
    state = controller.dispatch("/api/reset", {})
    assert controller.engine.state_dict() == GeometryEngine().state_dict()
    assert controller.settings == DEFAULT_SETTINGS
    assert state["status"]["mode"] == "idle"
    assert state["status"]["step"] == 0
    assert state["status"]["run_id"] is None
    assert state["status"]["frame_count"] == 0
    assert not any(state["status"]["pending"].values())
    assert not state["status"]["requires_resolution"]
    assert controller._branch_parent is None
    assert controller.engine.optimizer_step == 0
    assert any(r["id"] == run_id for r in state["runs"])
    assert files == {str(p.relative_to(controller.store.root)): p.read_bytes()
                     for p in controller.store.directory(run_id).rglob("*") if p.is_file()}


def test_full_reset_stops_live_training_and_saves_latest_frame(controller):
    controller.train({"steps": 1000, "snapshot_every": 1, "delay_ms": 200})
    wait_for(lambda: controller.engine.step_count >= 1)
    run_id, worker = controller.run["id"], controller._worker
    state = controller.reset()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert state["status"]["mode"] == "idle"
    assert controller.engine.step_count == 0
    manifest = controller.store.manifest(run_id)
    assert manifest["last_step"] >= 1
    assert controller.store.frame(manifest, manifest["frames"]-1)["step"] == manifest["last_step"]


def test_full_reset_ends_replay_and_clears_branch_source(controller):
    finish(controller)
    run_id = controller.run["id"]
    controller.select(1)
    controller.edit("center", 2, .23)
    assert controller._branch_parent is not None
    controller.reset()
    assert controller._branch_parent is None and controller._branch_initial is None
    controller.train({"steps": 1, "snapshot_every": 1, "delay_ms": 0})
    wait_for(lambda: controller.mode == "paused")
    assert controller.run["id"] != run_id and controller.run["parent"] is None


def test_resize_api_counts_interior_points_and_preserves_discard_baseline(controller):
    controller.reset()
    before = controller.engine.state_dict()
    state = controller.dispatch("/api/resize", {"n_interior": 100})
    assert state["frame"]["config"]["n_interior"] == 100
    assert len(state["frame"]["centers"]) == 120
    assert state["frame"]["ideal_gamma"] == pytest.approx(12.375)
    assert state["status"]["pending"]["geometry"]
    controller.discard()
    assert controller.engine.state_dict() == before
