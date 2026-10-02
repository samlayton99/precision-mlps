"""Local HTTP controller and durable recordings for the Codex geometry reader.

Every evaluation is a training barrier: solve, evaluate, atomically write the
frame (including optimizer state), and only then take the next Adam step.
Recorded curves are replayed exactly, without another numerical solve.
"""
from __future__ import annotations

import copy
import gzip
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
import uuid

import numpy as np

try:
    from .engine import GeometryEngine
except ImportError:  # The standalone launcher also supports direct execution.
    from engine import GeometryEngine


HERE = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = HERE.parents[1] / "results/checkpoint_G_interactive/geometry_reader_codex/runs"
DEFAULT_SETTINGS = {"steps": 2000, "snapshot_every": 10, "schedule": "fixed",
                    "growth": 1.3, "delay_ms": 100}
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _json_bytes(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _atomic_write(path, payload):
    """A frame is either complete or absent, including after interruption."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_settings(updates, previous=None):
    if not isinstance(updates, dict):
        raise ValueError("Recording settings must be an object.")
    unknown = set(updates) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown recording setting: {sorted(unknown)[0]}")
    result = {**(previous or DEFAULT_SETTINGS), **updates}
    for key, minimum, maximum in (("steps", 1, 10_000_000),
                                  ("snapshot_every", 1, 10_000_000),
                                  ("delay_ms", 0, 60_000)):
        value = float(result[key])
        if not math.isfinite(value) or value != int(value) or not minimum <= value <= maximum:
            raise ValueError(f"{key} must be an integer from {minimum} to {maximum}.")
        result[key] = int(value)
    result["growth"] = float(result["growth"])
    if not math.isfinite(result["growth"]) or not 1 <= result["growth"] <= 10:
        raise ValueError("Geometric growth must be between 1 and 10.")
    if result["schedule"] not in ("fixed", "geometric"):
        raise ValueError("Snapshot schedule must be fixed or geometric.")
    return result


def snapshot_steps(steps, every=1, schedule="fixed", growth=1.3):
    """Yield a complete schedule, including zero and the requested endpoint."""
    settings = validate_settings({"steps": steps, "snapshot_every": every,
                                  "schedule": schedule, "growth": growth})
    position, gap = 0, float(settings["snapshot_every"])
    yield 0
    while position < settings["steps"]:
        position = min(settings["steps"], position + int(math.ceil(gap)))
        yield position
        if schedule == "geometric":
            gap = min(float(settings["steps"]), gap * settings["growth"])


class RecordingStore:
    """Compressed frame files plus a small manifest and append-only timeline.

    Only the selected frame is loaded.  The timeline contains no arrays, so a
    long run's memory requirement does not grow with its plotted sample count.
    """

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, run_id):
        if not isinstance(run_id, str) or not _UUID.fullmatch(run_id):
            raise ValueError("Invalid recording identifier.")
        return self.root / run_id

    def create(self, name, settings, parent=None):
        run_id = str(uuid.uuid4())
        directory = self.directory(run_id)
        directory.mkdir()
        manifest = {"version": 1, "id": run_id, "name": name,
                    "created": time.time(), "updated": time.time(),
                    "frames": 0, "last_step": 0, "settings": copy.deepcopy(settings),
                    "parent": parent, "status": "paused"}
        self.write_manifest(manifest)
        return manifest

    def write_manifest(self, manifest):
        _atomic_write(self.directory(manifest["id"]) / "manifest.json", _json_bytes(manifest))

    def manifest(self, run_id):
        path = self.directory(run_id) / "manifest.json"
        if not path.is_file():
            raise ValueError("Recording was not found.")
        result = json.loads(path.read_text())
        if result.get("version") != 1 or result.get("id") != run_id:
            raise ValueError("Unsupported or invalid recording manifest.")
        return result

    def list_runs(self):
        runs = []
        for path in self.root.glob("*/manifest.json"):
            if not _UUID.fullmatch(path.parent.name):
                continue
            try:
                runs.append(self.manifest(path.parent.name))
            except (ValueError, OSError):
                continue
        return sorted(runs, key=lambda item: item["updated"], reverse=True)

    def append(self, manifest, frame):
        index = manifest["frames"]
        path = self.directory(manifest["id"])
        frame = copy.deepcopy(frame)
        frame["index"] = index
        entry = {"index": index, "step": frame["step"], "event": frame.get("event", "snapshot")}
        if frame.get("changes"):
            entry["changes"] = copy.deepcopy(frame["changes"])
        _atomic_write(path / f"{index:08d}.json.gz", gzip.compress(_json_bytes(frame), compresslevel=3))
        # Persist the small entry only after its frame exists. A fixed-size
        # manifest avoids rewriting a 50,000-entry timeline at every snapshot.
        with (path / "timeline.jsonl").open("ab") as handle:
            handle.write(_json_bytes(entry) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        manifest.update(frames=index + 1, last_step=frame["step"], updated=time.time())
        self.write_manifest(manifest)
        return frame, entry

    def timeline(self, manifest):
        path = self.directory(manifest["id"]) / "timeline.jsonl"
        if not path.exists():
            return []
        result = []
        with path.open() as handle:
            for line in handle:
                if len(result) >= manifest["frames"]:
                    break
                entry = json.loads(line)
                if entry.get("index") != len(result):
                    raise ValueError("Recording timeline is inconsistent.")
                result.append(entry)
        if len(result) != manifest["frames"]:
            raise ValueError("Recording timeline is incomplete.")
        return result

    def frame(self, manifest, index):
        if isinstance(index, bool) or not isinstance(index, (int, float)) or int(index) != index:
            raise ValueError("Frame index must be an integer.")
        index = int(index)
        if not 0 <= index < manifest["frames"]:
            raise ValueError("Frame index is out of range.")
        path = self.directory(manifest["id"]) / f"{index:08d}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("index") != index or "state" not in result:
            raise ValueError("Recording frame is invalid.")
        return result


class Controller:
    def __init__(self, run_root=None, engine=None, catalog_path=None):
        self.engine = engine or GeometryEngine()
        self.store = RecordingStore(run_root or DEFAULT_RUN_ROOT)
        self.lock = threading.RLock()
        self.settings = dict(DEFAULT_SETTINGS)
        self.mode, self.error = "idle", None
        self.frame = self.engine.evaluate(feedback=False)
        self.frame.update(event="initial", changes=[])
        self._committed_state = self.engine.state_dict()
        self._committed_frame = copy.deepcopy(self.frame)
        self.pending = {"geometry": False, "readout": False}
        self._staged_changes = []
        self._branch_parent = None
        self._branch_initial = None
        self._requires_resolution = False
        self.run = None
        self.timeline = []
        self.selected_frame = None
        self._runs = {run["id"]: run for run in self.store.list_runs()}
        self._stop = threading.Event()
        self._worker = None
        self._generation = 0
        self._version = 1
        self._target_step = None
        self._next_due = None
        self._gap = float(self.settings["snapshot_every"])
        self._closed = False
        catalog = Path(catalog_path or HERE / "presets/catalog.json")
        self.presets = []
        if catalog.is_file():
            self.presets = json.loads(catalog.read_text()).get("presets", [])

    def _touch(self):
        self._version += 1

    def restore_session(self, session):
        """Restore a captured preview and committed baseline after an app update."""
        preview, committed = session["preview"], session["committed"]
        baseline = GeometryEngine().load_state(committed["frame"]["state"])
        current = GeometryEngine().load_state(preview["frame"]["state"])
        with self.lock:
            if self.mode == "training":
                raise ValueError("Pause training before restoring a session.")
            self.engine = current
            self.frame = copy.deepcopy(preview["frame"])
            self.frame.update(config=copy.deepcopy(current.config), state=current.state_dict())
            self._committed_state = baseline.state_dict()
            self._committed_frame = copy.deepcopy(committed["frame"])
            self._committed_frame.update(config=copy.deepcopy(baseline.config), state=baseline.state_dict())
            self.pending = dict(preview["status"].get("pending", {"geometry": False, "readout": False}))
            self._staged_changes = copy.deepcopy(preview["frame"].get("changes", []))
            self.settings = validate_settings(preview.get("settings", {}))
            run_id = preview["status"].get("run_id")
            self.run = self.store.manifest(run_id) if run_id else None
            self.timeline = self.store.timeline(self.run) if self.run else []
            self.selected_frame = preview["status"].get("selected_frame") if self.run else None
            self.mode = preview["status"]["mode"]
            if self.mode == "training":
                self.mode = "paused"
            self._target_step = preview["status"].get("target_step")
            self._next_due = None
            self._branch_parent = copy.deepcopy(session.get("branch_parent"))
            self._branch_initial = copy.deepcopy(self._committed_frame) if self._branch_parent else None
            self._requires_resolution = False
            self.error = None
            self._touch()
            return self.state()

    def _public_run(self, run):
        return {key: copy.deepcopy(value) for key, value in run.items()
                if key in ("id", "name", "created", "updated", "frames", "last_step", "status", "parent")}

    def state(self, since=None):
        with self.lock:
            if since is not None and str(self._version) == str(since):
                return {"unchanged": True, "version": self._version}
            presets = [{key: copy.deepcopy(value) for key, value in item.items()
                        if key not in ("geometry", "readout")} for item in self.presets]
            return {"version": self._version, "frame": copy.deepcopy(self.frame),
                    "status": {"mode": self.mode, "error": self.error,
                               "step": self.engine.step_count,
                               "run_id": self.run["id"] if self.run else None,
                               "frame_count": len(self.timeline), "selected_frame": self.selected_frame,
                               "target_step": self._target_step,
                               "pending": dict(self.pending),
                               "requires_resolution": self._requires_resolution,
                               "finished": self._target_step is not None and self.engine.step_count >= self._target_step},
                    "timeline": copy.deepcopy(self.timeline), "settings": dict(self.settings),
                    "presets": presets,
                    "runs": [self._public_run(run) for run in sorted(self._runs.values(),
                              key=lambda item: item["updated"], reverse=True)],
                    "recording_directory": str(self.store.root)}

    def _mark_run(self, status=None):
        if self.run is not None:
            self.run.update(status=status or self.mode, settings=dict(self.settings), updated=time.time())
            self.store.write_manifest(self.run)
            self._runs[self.run["id"]] = self.run.copy()

    def _append(self, frame):
        frame, entry = self.store.append(self.run, frame)
        self.timeline.append(entry)
        self.selected_frame = entry["index"]
        self.frame = frame
        self._runs[self.run["id"]] = self.run.copy()
        self._touch()
        return frame

    def _evaluate(self, event="snapshot", changes=None, feedback=False, record=True):
        frame = self.engine.evaluate(feedback=feedback)
        frame.update(event=event, changes=copy.deepcopy(changes or []))
        if record and self.run is not None:
            self._append(frame)
        else:
            self.frame = frame
            self._touch()
        if not any(self.pending.values()):
            self._committed_state = self.engine.state_dict()
            self._committed_frame = copy.deepcopy(self.frame)
        return frame

    def _commit_current(self):
        self._committed_state = self.engine.state_dict()
        self._committed_frame = copy.deepcopy(self.frame)
        self.pending = {"geometry": False, "readout": False}
        self._staged_changes = []
        self._requires_resolution = False

    def _begin_preview(self):
        if self.mode == "replay":
            self._branch_parent = {"id": self.run["id"], "frame": self.selected_frame}
            self._branch_initial = copy.deepcopy(self._committed_frame)
            self._detach_recording()

    def _fork_source(self):
        if self._branch_parent is not None:
            self._start_recording(parent=self._branch_parent, initial=self._branch_initial)
            self._branch_parent = self._branch_initial = None
            self.mode = "paused"

    def _start_recording(self, parent=None, initial=None):
        name = f"{self.engine.config['target']} · {time.strftime('%Y-%m-%d %H:%M:%S')}"
        self.run = self.store.create(name, self.settings, parent=parent)
        self.timeline, self.selected_frame = [], None
        self._target_step, self._next_due = None, None
        if initial is not None:
            initial = copy.deepcopy(initial)
            initial.update(event="branch_start", changes=[])
            self._append(initial)
        self._runs[self.run["id"]] = self.run.copy()

    def _fork_replay(self):
        self._begin_preview()
        self._fork_source()

    def _halt(self, snapshot=True):
        """Call with lock held after setting the cancellation event."""
        was_training = self.mode == "training"
        self._generation += 1
        if was_training and snapshot and self.frame["step"] != self.engine.step_count:
            self._evaluate(event="pause", feedback=False)
        if was_training:
            self.mode = "paused"
            self._mark_run()
        self._touch()

    def pause(self):
        self._stop.set()
        with self.lock:
            self._halt()
            return self.state()

    def _detach_recording(self):
        # Parameter-panel changes conclude the old record. Its files remain in
        # the library; subsequent Play begins a fresh recording of the new setup.
        self.run, self.timeline, self.selected_frame = None, [], None
        self._target_step, self._next_due = None, None
        self.mode = "idle"

    def configure(self, config):
        self._stop.set()
        with self.lock:
            self._halt()
            if not isinstance(config, dict):
                raise ValueError("Configuration must be an object.")
            unknown = set(config) - set(self.engine.config)
            if unknown:
                raise ValueError(f"Unknown control: {sorted(unknown)[0]}")
            saved = self.engine.state_dict()
            had_pending = any(self.pending.values())
            committed = copy.deepcopy(self._committed_state)
            staged_changes = copy.deepcopy(self._staged_changes)

            def apply_controls(state):
                self.engine.load_state(state)
                reset_keys = {"n_interior", "n_centers", "n_intervals", "halo", "auto_halo", "center_min", "center_max"}
                reset = any(key in config and config[key] != self.engine.config[key] for key in reset_keys)
                reset_readout = "readout_init" in config and config["readout_init"] != self.engine.config["readout_init"]
                self.engine.configure(config, reset=reset)
                if reset_readout and not reset:
                    self.engine.reset_readout()
                result = self.engine.evaluate(feedback=False)
                result.update(event="configuration", changes=[])
                return self.engine.state_dict(), result

            try:
                # Apply ordinary controls to both states. Changing the learning
                # rate or target must neither inject nor erase a staged drag.
                if had_pending:
                    next_committed, committed_frame = apply_controls(committed)
                next_preview, frame = apply_controls(saved)
            except Exception:
                self.engine.load_state(saved)
                raise
            self._detach_recording()
            self.frame = frame
            self.frame.update(event="configuration", changes=[])
            self.error = None
            self._branch_parent = self._branch_initial = None
            if had_pending:
                self._committed_state, self._committed_frame = next_committed, committed_frame
                self.pending = {
                    "geometry": any(next_preview["params"][key] != next_committed["params"][key]
                                    for key in ("w", "b"))
                                or next_preview["config"]["reference_h"] != next_committed["config"]["reference_h"],
                    "readout": any(next_preview["params"][key] != next_committed["params"][key]
                                   for key in ("v", "c")),
                }
                self._staged_changes = staged_changes
                self._requires_resolution = False
                if any(self.pending.values()):
                    self.frame.update(event="preview", changes=staged_changes)
                else:
                    self._commit_current()
            else:
                self._commit_current()
            self._touch()
            return self.state()

    def apply_preset(self, preset_id, apply_settings=False):
        self._stop.set()
        with self.lock:
            self._halt()
            preset = next((p for p in self.presets if p.get("id") == preset_id), None)
            if preset is None and preset_id in ("uniform", "xavier"):
                preset = {"kind": preset_id}
            if preset is None:
                raise ValueError("Unknown geometry preset.")
            saved = self.engine.state_dict()
            try:
                self.engine.initialize(preset, apply_settings=bool(apply_settings))
                frame = self.engine.evaluate(feedback=False)
            except Exception:
                self.engine.load_state(saved)
                raise
            self._begin_preview()
            self.frame = frame
            self.frame.update(event="preset", preset_id=preset_id, changes=[])
            self.pending = {"geometry": True, "readout": True}
            self._staged_changes.append({"kind": "preset", "index": None,
                                         "before": None, "after": preset_id})
            self.error = None
            self._touch()
            return self.state()

    def _edit_value(self, kind, index):
        if kind == "mean_gamma":
            return float(np.abs(self.engine.params["w"]).mean())
        if kind == "mean_lambda":
            return float(self.engine.lambdas.mean())
        if kind == "output_bias":
            return float(self.engine.params["c"][0])
        if kind not in ("center", "lambda", "gamma", "readout"):
            raise ValueError("Unknown plotted control.")
        if isinstance(index, bool) or index is None or int(index) != index:
            raise ValueError("A neuron index must be an integer.")
        index = int(index)
        values = {"center": self.engine.centers, "lambda": self.engine.lambdas,
                  "gamma": np.abs(self.engine.params["w"]),
                  "readout": self.engine.params["v"]}[kind]
        if not 0 <= index < len(values):
            raise ValueError("Neuron index is out of range.")
        return float(values[index])

    def edit(self, kind, index=None, value=None, readout_source="adam"):
        with self.lock:
            if readout_source not in ("adam", "solved"):
                raise ValueError("Readout source must be adam or solved.")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("Control value must be finite.")
            self._edit_value(kind, index)  # Validate before a replay can branch.
            saved, previous_frame = self.engine.state_dict(), copy.deepcopy(self.frame)
            previous_mode = self.mode
            try:
                # Readout dots refer to the complete displayed solved vector;
                # copying only the edited element would produce a different net.
                if kind in ("readout", "output_bias"):
                    before_frame = self.frame
                    self.engine.params["v"] = np.asarray(before_frame[f"{readout_source}_readout"], dtype=float).copy()
                    self.engine.params["c"] = np.asarray([before_frame[f"{readout_source}_output_bias"]], dtype=float)
                before = self._edit_value(kind, index)
                self.engine.edit(kind, index, value)
                after = self._edit_value(kind, index)
                frame = self.engine.evaluate(feedback=False)
            except Exception:
                self.engine.load_state(saved)
                raise
            if previous_mode != "training":
                self._begin_preview()
                component = "readout" if kind in ("readout", "output_bias") else "geometry"
                self.pending[component] = True
                change = {"kind": kind, "index": index, "before": before, "after": after}
                self._staged_changes.append(change)
                self.frame = frame
                self.frame.update(event="preview", changes=copy.deepcopy(self._staged_changes), highlight_ms=1000)
                self.error = None
                self._requires_resolution = False
                self._touch()
                return self.state()
            if self.run is not None and previous_frame["step"] != saved["step_count"]:
                # Keep an exact pre-intervention state if steps occurred since
                # the last scheduled snapshot. No optimizer update runs here.
                after_state = self.engine.state_dict()
                self.engine.load_state(saved)
                self._evaluate(event="before_manual", feedback=False)
                self.engine.load_state(after_state)
            changes = [{"kind": kind, "index": index, "before": before, "after": after}]
            frame.update(event="manual", changes=changes, highlight_ms=1000)
            if self.run is not None:
                self._append(frame)
            else:
                self.frame = frame
                self._touch()
            self.error = None
            self._commit_current()
            return self.state()

    def transform(self, action, amount, preserve_mean=False):
        def apply():
            rule = self.engine.config["clean_lambda"]
            try:
                if action == "clean" and preserve_mean:
                    self.engine.config["clean_lambda"] = float(self.engine.lambdas.mean())
                self.engine.transform(action, amount)
            finally:
                self.engine.config["clean_lambda"] = rule
        return self._geometry_action(apply, action)

    def resize(self, count):
        return self._geometry_action(lambda: self.engine.resize(count), "resize")

    def _geometry_action(self, action, event):
        self._stop.set()
        with self.lock:
            self._halt()
            saved = self.engine.state_dict()
            try:
                action()
                frame = self.engine.evaluate(feedback=False)
            except Exception:
                self.engine.load_state(saved)
                raise
            self._begin_preview()
            self.frame = frame
            self.frame.update(event=event, changes=[])
            component = "readout" if event == "reset_readout" else "geometry"
            self.pending[component] = True
            self._staged_changes.append({"kind": event, "index": None, "before": None, "after": None})
            self.error = None
            self._touch()
            return self.state()

    def update_settings(self, settings):
        candidate = validate_settings(settings, self.settings)
        self._stop.set()
        with self.lock:
            self._halt()
            if self.mode == "replay":
                self._begin_preview()
            self.settings = candidate
            self._next_due, self._target_step = None, None
            self._mark_run()
            self._touch()
            return self.state()

    def inject(self, component="all", readout_source="adam"):
        if component not in ("geometry", "readout", "all"):
            raise ValueError("Injection component must be geometry, readout or all.")
        if readout_source not in ("adam", "solved"):
            raise ValueError("Readout source must be adam or solved.")
        self._stop.set()
        with self.lock:
            self._halt()
            preview = self.engine.state_dict()
            preview_frame = copy.deepcopy(self.frame)
            baseline = copy.deepcopy(self._committed_state)
            geometry = component in ("geometry", "all")
            readout = component in ("readout", "all")
            if not geometry and len(preview["params"]["w"]) != len(baseline["params"]["w"]):
                raise ValueError("Inject geometry first when the preview changes the number of centers.")
            commit = copy.deepcopy(preview if geometry else baseline)
            if geometry and not readout:
                old_n, new_n = len(baseline["params"]["v"]), len(preview["params"]["v"])
                if old_n == new_n:
                    commit["params"]["v"] = copy.deepcopy(baseline["params"]["v"])
                else:
                    commit["params"]["v"] = (np.interp(np.linspace(0, 1, new_n),
                        np.linspace(0, 1, old_n), baseline["params"]["v"])
                        * preview["config"]["reference_h"] / baseline["config"]["reference_h"]).tolist()
                commit["params"]["c"] = copy.deepcopy(baseline["params"]["c"])
            if readout:
                commit["params"]["v"] = copy.deepcopy(preview_frame[f"{readout_source}_readout"])
                commit["params"]["c"] = [preview_frame[f"{readout_source}_output_bias"]]
                commit["pending_readout_edit"] = True
            # An explicit intervention starts fresh Adam moments. The global
            # training-step counter is retained for the recorded timeline.
            commit["step_count"] = baseline["step_count"]
            commit["optimizer_step"] = 0
            for group in ("m", "q"):
                commit[group] = {key: [0.0] * len(values) for key, values in commit["params"].items()}
            try:
                self.engine.load_state(commit)
                frame = self.engine.evaluate(feedback=False)
            except Exception:
                self.engine.load_state(preview)
                raise
            self._begin_preview()
            self._fork_source()
            changes = [item for item in self._staged_changes
                       if (item["kind"] in ("readout", "output_bias", "reset_readout")) == (not geometry)
                       or component == "all"]
            if readout and not any(item["kind"] in ("readout", "output_bias", "reset_readout") for item in changes):
                changes.append({"kind": "readout_injection", "index": None,
                                "before": None, "after": readout_source})
            frame.update(event="manual", changes=changes, highlight_ms=1000)
            if self.run is not None:
                self._append(frame)
            else:
                self.frame = frame
            remaining = {"geometry": self.pending["geometry"] and not geometry,
                         "readout": self.pending["readout"] and not readout}
            remaining_changes = [item for item in self._staged_changes if item not in changes]
            self._commit_current()
            # Keep any un-injected component visible, over the newly committed
            # state, so the second injection button can apply it separately.
            if any(remaining.values()):
                remainder = copy.deepcopy(commit)
                if remaining["geometry"]:
                    remainder["config"] = copy.deepcopy(preview["config"])
                    remainder["params"]["w"] = copy.deepcopy(preview["params"]["w"])
                    remainder["params"]["b"] = copy.deepcopy(preview["params"]["b"])
                if remaining["readout"]:
                    remainder["params"]["v"] = copy.deepcopy(preview["params"]["v"])
                    remainder["params"]["c"] = copy.deepcopy(preview["params"]["c"])
                    remainder["pending_readout_edit"] = True
                self.engine.load_state(remainder)
                self.frame = self.engine.evaluate(feedback=False)
                self.frame.update(event="preview", changes=remaining_changes)
                self.pending = remaining
                self._staged_changes = remaining_changes
            self.error = None
            self._touch()
            return self.state()

    def discard(self):
        self._stop.set()
        with self.lock:
            self._halt()
            self.engine.load_state(self._committed_state)
            self.frame = copy.deepcopy(self._committed_frame)
            self.pending = {"geometry": False, "readout": False}
            self._staged_changes = []
            self._requires_resolution = False
            self.error = None
            self._touch()
            return self.state()

    def train(self, settings=None, resolution=None, readout_source="adam"):
        with self.lock:
            if self._closed:
                raise ValueError("The geometry reader is shutting down.")
            candidate = validate_settings(settings or {}, self.settings)
            if self.mode == "training":
                return self.state()
            if any(self.pending.values()):
                if resolution == "inject":
                    component = "all" if all(self.pending.values()) else ("geometry" if self.pending["geometry"] else "readout")
                    self.inject(component, readout_source)
                elif resolution == "discard":
                    self.discard()
                else:
                    self._requires_resolution = True
                    self._touch()
                    return self.state()
            self.settings = candidate
            self._fork_replay()
            if self.run is None:
                self._start_recording()
            if not self.timeline:
                # Preserve the requested zero/Xavier initialization for the
                # first gradient step. Later scheduled frames apply LS feedback.
                self._evaluate(event="initial", feedback=False)
            elif self.frame["step"] != self.engine.step_count:
                self._evaluate(event="resume", feedback=False)
            if self._target_step is None or self.engine.step_count >= self._target_step:
                self._target_step = self.engine.step_count + self.settings["steps"]
                self._next_due = None
            if self._next_due is None:
                self._gap = float(self.settings["snapshot_every"])
                self._next_due = min(self._target_step, self.engine.step_count + int(self._gap))
            self._generation += 1
            generation = self._generation
            self._stop = threading.Event()
            stop = self._stop
            self.mode, self.error = "training", None
            self._mark_run()
            self._touch()
            self._worker = threading.Thread(target=self._training_loop, args=(generation, stop),
                                            daemon=True, name="geometry-reader-adam")
            self._worker.start()
            return self.state()

    def _training_loop(self, generation, stop):
        try:
            while not stop.is_set():
                delay = 0
                with self.lock:
                    if generation != self._generation or self.mode != "training" or stop.is_set():
                        return
                    self.engine.step(1)
                    current = self.engine.step_count
                    if current >= self._next_due or current >= self._target_step:
                        self._evaluate(event="snapshot", feedback=True)
                        if self.settings["schedule"] == "geometric":
                            self._gap = min(float(self.settings["steps"]), self._gap * self.settings["growth"])
                        self._next_due = min(self._target_step, current + int(math.ceil(self._gap)))
                        delay = self.settings["delay_ms"] / 1000.0
                    if current >= self._target_step:
                        self.mode = "paused"
                        self._mark_run(status="complete")
                        self._touch()
                        return
                if delay:
                    stop.wait(delay)
                else:
                    # Yield even between sparse evaluations so control requests
                    # obtain the lock promptly. Cancellation is checked each step.
                    time.sleep(0)
        except Exception as exc:
            with self.lock:
                if generation == self._generation:
                    self.mode, self.error = "paused", str(exc)
                    stop.set()
                    # Restore the last complete, resumable frame after a failed
                    # step/solve, rather than leaving nonfinite parameters live.
                    self.engine.load_state(self.frame["state"])
                    try:
                        self._mark_run(status="error")
                    except OSError:
                        pass
                    self._touch()

    def select(self, index, run_id=None):
        self._stop.set()
        with self.lock:
            self._halt()
            if run_id is None and self.run is None:
                raise ValueError("Choose a saved recording first.")
            run_id = run_id or self.run["id"]
            manifest = self.store.manifest(run_id)
            timeline = self.store.timeline(manifest)
            frame = self.store.frame(manifest, index)
            self.engine.load_state(frame["state"])
            self.run, self.timeline, self.frame = manifest, timeline, frame
            self.settings = validate_settings(manifest.get("settings", {}))
            self.selected_frame = int(index)
            self.mode, self.error = "replay", None
            self._target_step, self._next_due = None, None
            self._branch_parent = self._branch_initial = None
            self._commit_current()
            self._touch()
            return self.state()

    def load(self, run_id):
        return self.select(0, run_id)

    def save(self, name):
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 160:
            raise ValueError("Recording name must contain 1–160 characters.")
        with self.lock:
            if any(self.pending.values()):
                raise ValueError("Inject or discard the preview before saving it as a recording.")
            if self.run is None:
                self._start_recording()
                self._evaluate(event="initial", feedback=False)
            self.run["name"] = name.strip()
            # Naming a loaded recording does not change its stored training status.
            self._mark_run(status=self.run.get("status") if self.mode == "replay" else self.mode)
            self._touch()
            return self.state()

    def new(self):
        self._stop.set()
        with self.lock:
            self._halt()
            self._detach_recording()
            self.error = None
            self._touch()
            return self.state()

    def reset(self):
        """Stop activity and restore the default experiment; retain saved runs."""
        self._stop.set()
        with self.lock:
            self._halt()
            fresh = GeometryEngine()
            frame = fresh.evaluate(feedback=False)
            self._detach_recording()
            self.engine, self.frame = fresh, frame
            self.frame.update(event="reset", changes=[])
            self.settings = dict(DEFAULT_SETTINGS)
            self._gap = float(self.settings["snapshot_every"])
            self._branch_parent = self._branch_initial = None
            self.error = None
            self._commit_current()
            self._touch()
            return self.state()

    def close(self):
        self._stop.set()
        with self.lock:
            self._halt()
            self._closed = True
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=5)

    def dispatch(self, path, payload):
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        if path == "/api/config":
            return self.configure(payload["config"])
        if path == "/api/preset":
            return self.apply_preset(payload["id"], payload.get("apply_settings", False))
        if path == "/api/edit":
            return self.edit(payload["kind"], payload.get("index"), payload.get("value"),
                             payload.get("readout_source", "adam"))
        if path == "/api/transform":
            return self.transform(payload["action"], payload["amount"], payload.get("preserve_mean", False))
        if path == "/api/resize":
            return self.resize(payload["n_interior"] if "n_interior" in payload else payload["count"])
        if path == "/api/train":
            return self.train(payload.get("settings"), payload.get("resolution"),
                              payload.get("readout_source", "adam"))
        if path == "/api/inject":
            return self.inject(payload.get("component", "all"), payload.get("readout_source", "adam"))
        if path == "/api/discard":
            return self.discard()
        if path == "/api/pause":
            return self.pause()
        if path == "/api/select":
            return self.select(payload["index"], payload.get("run_id"))
        if path == "/api/load":
            return self.load(payload["id"])
        if path == "/api/save":
            return self.save(payload["name"])
        if path == "/api/new":
            return self.new()
        if path == "/api/reset":
            return self.reset()
        if path == "/api/settings":
            return self.update_settings(payload["settings"])
        raise ValueError("Unknown API route.")


def make_server(controller, host="127.0.0.1", port=8068, static_root=None):
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("The geometry reader serves this laptop only; use 127.0.0.1.")
    static_root = Path(static_root or HERE / "static").resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "GeometryReaderCodex/1.0"

        def log_message(self, format, *args):
            pass

        def _json(self, payload, status=200):
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _local_request(self, mutation=False):
            try:
                host_header = self.headers.get("Host", "")
                host = urlsplit("http://" + host_header)
                if host.hostname not in ("localhost", "127.0.0.1") or host.port != self.server.server_port:
                    raise ValueError("Request host must match this local geometry reader.")
                if mutation:
                    origin = self.headers.get("Origin")
                    if origin:
                        parsed = urlsplit(origin)
                        if (parsed.scheme != "http" or parsed.hostname != host.hostname
                                or parsed.port != host.port):
                            raise ValueError("Cross-origin control requests are not allowed.")
                    if self.headers.get_content_type() != "application/json":
                        raise ValueError("Control requests require application/json.")
                return True
            except ValueError as exc:
                self._json({"error": str(exc)}, 403)
                return False

        def do_GET(self):
            if not self._local_request():
                return
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/api/state":
                    self._json(controller.state(parse_qs(parsed.query).get("since", [None])[0]))
                    return
                if parsed.path == "/api/runs":
                    self._json({"runs": controller.state()["runs"]})
                    return
                if parsed.path == "/api/presets":
                    self._json({"presets": controller.state()["presets"]})
                    return
                name = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
                target = (static_root / name).resolve()
                if not target.is_relative_to(static_root) or not target.is_file():
                    self._json({"error": "File was not found."}, 404)
                    return
                body = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, KeyError, OSError) as exc:
                self._json({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self):
            if not self._local_request(mutation=True):
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise ValueError("Request size must be between 1 byte and 1 MB.")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request body.")
                payload = json.loads(raw)
                result = controller.dispatch(urlsplit(self.path).path, payload)
                self._json(result)
            except (ValueError, TypeError, KeyError, IndexError, OSError, OverflowError) as exc:
                self._json({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:
                self._json({"error": f"Operation failed: {exc}"}, 500)

    server = ThreadingHTTPServer((host, int(port)), Handler)
    server.daemon_threads = True
    return server


def serve(host="127.0.0.1", port=8068, run_root=None, session_path=None):
    controller = Controller(run_root=run_root)
    if session_path is not None:
        controller.restore_session(json.loads(Path(session_path).read_text()))
    server = make_server(controller, host=host, port=port)
    print(f"Codex geometry reader: http://{host}:{server.server_port}", flush=True)
    print(f"Recordings: {controller.store.root}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        controller.close()
        server.server_close()


if __name__ == "__main__":
    serve()
