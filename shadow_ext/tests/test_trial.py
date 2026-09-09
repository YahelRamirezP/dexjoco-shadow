"""Measured-trial contracts: clocks, task metrics, events, both hands and summaries."""
import json
import socket
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

from shadow_ext.state_recording import StateTrace, TraceReader
from shadow_ext.trial import Evaluation, consolidate, session_summary
from shadow_ext import teleop_driver as driver
from shadow_ext.build import build_spec
from shadow_ext.mapping import build_finger_map, qpos_to_ctrl
from test_session_recording import XML


def test_evaluation_clock_excludes_preparation_and_late_success():
    evaluation = Evaluation("manual", 2.0)
    evaluation.observe(True, 1_000_000_000, 1)
    assert evaluation.first_success_ns is None
    assert evaluation.start(10_000_000_000, 3)
    assert not evaluation.start(11_000_000_000, 4)
    evaluation.observe(False, 10_500_000_000, 3.1)
    evaluation.observe(True, 11_000_000_000, 3.2)
    result = evaluation.summary(12_000_000_000, 3.5)
    assert result["time_to_success_wall_s"] == 1
    assert result["time_to_success_sim_s"] == pytest.approx(.2)
    late = Evaluation("immediate", 1)
    late.start(0, 0)
    late.observe(True, 1_000_000_000, .5)
    assert late.expired(1_000_000_000)
    assert late.first_success_ns is None


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf")])
def test_invalid_time_limits(limit):
    with pytest.raises(ValueError):
        Evaluation("manual", limit)


def test_metrics_events_and_legacy_reader(tmp_path):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    trace = StateTrace(tmp_path / "sim", model, {"evaluation_start": "manual",
                                               "time_limit_wall_s": 5}, chunk_size=2)
    trace.append(data, target_ctrl=[], slip_bias=[], success=True, timestamp_ns=0)
    assert trace.start_evaluation(data, 1_000_000_000)
    assert not trace.start_evaluation(data, 2_000_000_000)
    trace.event("manual_correction", data, timestamp_ns=1_100_000_000, position_delta_m=[.005, 0, 0])
    for k in range(1, 4):
        data.time = k * .1
        trace.append(data, target_ctrl=[], slip_bias=[], success=k == 3,
                     timestamp_ns=1_000_000_000 + k * 100_000_000,
                     task_metrics={"minimum_bottom_lift_m": k * .05, "inside": True})
    trace.close("success")
    reader = TraceReader(tmp_path / "sim")
    assert reader.task_metrics_at(3)["minimum_bottom_lift_m"] == pytest.approx(.15)
    assert reader.task_metrics_at(0) is None
    assert reader.metadata["time_to_success_wall_s"] == 0  # raw preparation success
    assert reader.metadata["evaluation"]["time_to_success_wall_s"] == .3
    events = [json.loads(line) for line in (tmp_path / "sim/events.jsonl").read_text().splitlines()]
    assert [e["type"] for e in events] == ["evaluation_start", "manual_correction"]
    assert events[1]["preceding_state_index"] == 0
    reader.close()
    # Version 1 chunks lack the new columns; state replay must remain supported.
    meta = trace.metadata.copy()
    meta["format_version"] = 1
    (tmp_path / "sim/metadata.json").write_text(json.dumps(meta))
    for path in (tmp_path / "sim").glob("*.npz"):
        with np.load(path) as chunk:
            arrays = {k: chunk[k] for k in chunk.files if k not in ("task_metrics", "evaluation_active")}
        np.savez(path, **arrays)
    reader = TraceReader(tmp_path / "sim")
    assert reader.success_at(3)
    assert reader.task_metrics_at(3) is None
    reader.close()


def test_manual_start_resets_task_once_and_logs_keys(tmp_path, monkeypatch):
    import mujoco.viewer
    class Task:
        arena = "arena_arm_hand_bucket_pick.xml"
        def reset(self):
            self.resets = getattr(self, "resets", 0) + 1
            self.calls = 0
        def update(self, model, data):
            self.calls += 1
            self.metrics = {"calls": self.calls}
            return self.calls >= 2
    task = Task()
    monkeypatch.setitem(driver.REGISTRY, "manual_test", task)
    class Viewer:
        def __init__(self, callback):
            self.callback, self.frames = callback, 0
        def is_running(self):
            return self.frames < 10
        def sync(self):
            self.frames += 1
            if self.frames == 2:  # preparation already had success; do not stop
                for key in (294, 326, 295, 294):  # F5, X+, phase, repeated F5
                    self.callback(key)
        def close(self):
            pass
    monkeypatch.setattr(mujoco.viewer, "launch_passive",
                        lambda *a, key_callback=None, **kw: Viewer(key_callback))
    assert driver.run(task_name="manual_test", view=True, key_control=True,
                      n_steps=10, session=str(tmp_path), evaluation_start="manual", time_limit=5)
    meta = json.loads((tmp_path / "sim/metadata.json").read_text())
    assert task.resets == 2
    assert meta["event_counts"]["evaluation_start"] == 1
    assert meta["event_counts"]["manual_correction"] == 1
    assert meta["event_counts"]["phase_marker"] == 1
    assert meta["evaluation"]["start_sim_time"] == pytest.approx(.004)
    assert meta["evaluation"]["time_to_success_sim_s"] == pytest.approx(.004)
    assert meta["frames"] == 5


def test_driver_excludes_success_after_deadline(tmp_path, monkeypatch):
    class SlowTask:
        arena = "arena_arm_hand_bucket_pick.xml"
        def reset(self):
            pass
        def update(self, model, data):
            time.sleep(.08)
            return True
    monkeypatch.setitem(driver.REGISTRY, "slow_test", SlowTask())
    assert not driver.run(task_name="slow_test", n_steps=3, session=str(tmp_path),
                          evaluation_start="immediate", time_limit=.05)
    meta = json.loads((tmp_path / "sim/metadata.json").read_text())
    assert meta["stop_reason"] == "time_limit"
    assert meta["success_ever"]  # capture preserves the raw result
    assert not meta["evaluation"]["success_ever"]
    assert meta["evaluation"]["time_to_success_wall_s"] is None


@pytest.mark.parametrize("task_name", ["pick_bucket", "water_plant", "pinch_tongs", "hammer_nail"])
def test_actual_task_metrics_are_serializable_and_match_result(task_name):
    task = driver.REGISTRY[task_name]
    model = build_spec(task.arena).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    task.reset()
    success = task.update(model, data)
    json.dumps(task.metrics, allow_nan=False)
    if task_name == "pick_bucket":
        assert success == (task.metrics["inside_reference_aabb"] and task.metrics["lifted"])
        assert task.metrics["minimum_bottom_lift_m"] == 0
    elif task_name == "water_plant":
        assert success == (task.metrics["consecutive_steps"] >= task.metrics["required_steps"])
    elif task_name == "pinch_tongs":
        assert success == (task.metrics["consecutive_steps"] >= task.metrics["required_steps"])
    else:
        assert success == (task.metrics["nail_depth_m"] >= task.metrics["success_depth_m"])


def test_allegro_mapping_driver_and_restore(tmp_path, monkeypatch):
    class Receiver:
        def __init__(self, n_joints):
            assert n_joints == 16
        def latest(self):
            return np.array([.1, .4, .5, .6] * 3 + [.7, .3, .4, .5])
        def close(self):
            pass
    monkeypatch.setattr(driver, "FingerReceiver", Receiver)
    driver.run(hand="allegro", recv_fingers=True, n_steps=500, session=str(tmp_path),
               evaluation_start="immediate", time_limit=5)
    reader = TraceReader(tmp_path / "sim")
    assert reader.metadata["hand"] == "allegro"
    data = reader.restore(500)
    assert data.time == pytest.approx(1.0)
    ids, plan = build_finger_map(reader.model, hand="allegro")
    assert len(ids) == 16
    np.testing.assert_allclose(data.ctrl[ids], Receiver(16).latest())
    q = np.zeros(reader.model.nq)
    q[[p[0] for p in plan]] = Receiver(16).latest()
    np.testing.assert_allclose(qpos_to_ctrl(q, ids, plan), Receiver(16).latest())
    assert np.all(np.isfinite(data.qpos))
    assert reader.task_metrics_at(500)["lift_threshold_m"] == .15
    reader.close()
    from shadow_ext.replay import render_trace
    def forbidden_step(*args, **kwargs):
        raise AssertionError("Allegro replay must not advance physics")
    monkeypatch.setattr(mujoco, "mj_step", forbidden_step)
    outputs = render_trace(tmp_path / "sim", tmp_path / "render/allegro.mp4",
                           "front,left", 160, 120, 2)
    assert len(outputs) == 2


def test_continuous_move_tracker_accumulates_while_held(monkeypatch):
    # No real X11 listener here -- this only exercises the press/release/step
    # bookkeeping, so it can't interfere with an operator's live session on
    # the same display.
    monkeypatch.setattr(driver, "_PYNPUT_OK", False)
    tracker = driver._ContinuousMoveTracker(rate=2.0, rot_rate=3.0)

    class FakeKey:
        def __init__(self, char):
            self.char = char

    tracker._on_press(FakeKey('6'))   # +X
    tracker._on_press(FakeKey('/'))   # +yaw
    dpos, dquat = tracker.step(0.5)
    np.testing.assert_allclose(dpos, [1.0, 0.0, 0.0])
    expected = np.zeros(4)
    mujoco.mju_axisAngle2Quat(expected, [0., 0., 1.], 1.5)
    np.testing.assert_allclose(dquat, expected, atol=1e-9)

    tracker._on_release(FakeKey('6'))
    tracker._on_release(FakeKey('/'))
    dpos, dquat = tracker.step(0.5)
    np.testing.assert_allclose(dpos, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(dquat, [1.0, 0.0, 0.0, 0.0])

    tracker._on_press(FakeKey('x'))  # not a bound key -- ignored
    dpos, dquat = tracker.step(1.0)
    np.testing.assert_allclose(dpos, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(dquat, [1.0, 0.0, 0.0, 0.0])


def test_continuous_move_requires_key_control_and_pynput(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="requires --key-control"):
        driver.run(n_steps=1, continuous_move=True, session=str(tmp_path))
    monkeypatch.setattr(driver, "_PYNPUT_OK", False)
    with pytest.raises(ValueError, match="needs pynput"):
        driver.run(n_steps=1, key_control=True, continuous_move=True,
                   session=str(tmp_path / "b"))


@pytest.mark.parametrize("joints", [16, 24])
def test_udp_rejects_wrong_hand_and_nonfinite_packets(joints):
    receiver = driver.FingerReceiver(port=0, n_joints=joints)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    address = receiver._sock.getsockname()
    try:
        for values in (np.zeros(24 if joints == 16 else 16), np.full(joints, np.nan)):
            sock.sendto(values.astype(np.float64).tobytes(), address)
        time.sleep(.03)
        assert receiver.latest() is None
        expected = np.arange(joints, dtype=np.float64) / 20
        sock.sendto(expected.tobytes(), address)
        until = time.monotonic() + 1
        while receiver.latest() is None and time.monotonic() < until:
            time.sleep(.005)
        np.testing.assert_array_equal(receiver.latest(), expected)
    finally:
        receiver.close()
        sock.close()


def make_session(path, *, reason="success", started=True, kind="evaluation", hand="shadow"):
    (path / "sim").mkdir(parents=True)
    (path / "operator").mkdir()
    sim = {"hand": hand, "task": "pick_bucket", "complete": True, "trial_kind": kind,
           "condition_id": "baseline-v1", "stop_reason": reason, "boot_id": "test", "clock": "monotonic",
           "end_ns": 4_000_000_000, "evaluation": {
               "started": started, "start_monotonic_ns": 1_000_000_000 if started else None,
               "success_ever": reason == "success",
               "time_to_success_wall_s": 3 if reason == "success" else None}}
    operator = {"complete": True, "robot": hand, "boot_id": "test", "clock": "monotonic",
                "start_ns": 0, "end_ns": 5_000_000_000}
    (path / "sim/metadata.json").write_text(json.dumps(sim))
    (path / "operator/metadata.json").write_text(json.dumps(operator))
    return path


def test_consolidation_keeps_interruptions_practice_and_missing_data(tmp_path):
    sessions = [make_session(tmp_path / "ok"),
                make_session(tmp_path / "timeout", reason="time_limit"),
                make_session(tmp_path / "interrupt", reason="manual_interruption"),
                make_session(tmp_path / "practice", kind="practice"),
                make_session(tmp_path / "unstarted", started=False)]
    report = consolidate(sessions)
    assert len(report["trials"]) == 5
    group = report["groups"][0]
    assert group["successful_trials"] == 1
    # manual interruption is a real attempted outcome (counts as a failure,
    # same as a time_limit), not an excluded data point -- only "practice"
    # and "unstarted" are excluded here.
    assert group["evaluable_trials"] == 3
    assert group["excluded_trials"] == 2
    assert group["success_rate"] == pytest.approx(1 / 3)
    assert group["median_time_to_success_wall_s"] == 3
    assert not session_summary(tmp_path / "missing")["eligible"]
    with pytest.raises(ValueError, match="once"):
        consolidate([sessions[0], sessions[0]])


def test_consolidation_flags_wrong_hand_or_missing_camera_coverage(tmp_path):
    session = make_session(tmp_path / "wrong")
    op_path = session / "operator/metadata.json"
    operator = json.loads(op_path.read_text())
    operator.update(robot="allegro", start_ns=2_000_000_000)
    op_path.write_text(json.dumps(operator))
    row = session_summary(session)
    assert row["eligible"]
    assert not row["evidence_eligible"]
    assert "hand_mismatch" in row["evidence_warnings"]
    assert "operator_does_not_cover_evaluation" in row["evidence_warnings"]
    operator.update(robot="shadow", start_ns=0)
    op_path.write_text(json.dumps(operator))
    sim_path = session / "sim/metadata.json"
    sim = json.loads(sim_path.read_text())
    sim["end_ns"] = 20_000_000_000  # past operator end_ns (5e9) by more than the tail tolerance
    sim_path.write_text(json.dumps(sim))
    assert session_summary(session)["evidence_warnings"] == ["operator_does_not_cover_evaluation"]


def test_manual_interruption_within_shutdown_tolerance_is_eligible(tmp_path):
    # Real pattern from a live manual interruption: the operator closes the
    # webcam window a couple of real seconds before the sim's own last
    # recorded physics tick, since the two are stopped by hand in separate
    # terminals. That gap alone must not disqualify an otherwise-valid trial.
    session = make_session(tmp_path / "interrupted", reason="manual_interruption")
    op_path = session / "operator/metadata.json"
    operator = json.loads(op_path.read_text())
    operator["end_ns"] = 4_000_000_000 - 2_400_000_000  # camera stopped 2.4s before sim's end_ns
    op_path.write_text(json.dumps(operator))
    row = session_summary(session)
    assert row["eligible"]
    assert row["exclusion_reasons"] == []

    # A longer gap is an evidence warning; the failed attempt still counts.
    operator["end_ns"] = 4_000_000_000 - 6_000_000_000
    op_path.write_text(json.dumps(operator))
    assert session_summary(session)["eligible"]
    assert "operator_does_not_cover_evaluation" in session_summary(session)["evidence_warnings"]



def test_cleanup_error_preserves_trace_and_closes_receiver(tmp_path, monkeypatch):
    import mujoco.viewer
    closed = []
    class Receiver:
        def __init__(self, n_joints):
            pass
        def latest(self):
            return np.zeros(24)
        def close(self):
            closed.append(True)
    class Viewer:
        def is_running(self):
            return True
        def sync(self):
            pass
        def close(self):
            raise RuntimeError("Synthetic viewer shutdown failure")
    monkeypatch.setattr(driver, "FingerReceiver", Receiver)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *a, **k: Viewer())
    with pytest.raises(RuntimeError, match="Trial cleanup failed"):
        driver.run(view=True, recv_fingers=True, n_steps=2, session=str(tmp_path))
    assert closed == [True]
    reader = TraceReader(tmp_path / "sim")
    assert reader.metadata["complete"]
    assert reader.metadata["stop_reason"] == "technical_error"
    assert reader.metadata["loop_stop_reason"] == "step_limit"
    assert "Synthetic viewer shutdown failure" in reader.metadata["error"]
    reader.close()


def test_simulation_reset_is_a_technical_error(tmp_path, monkeypatch):
    monkeypatch.setattr(mujoco, "mj_step", lambda model, data: setattr(data, "time", 0))
    with pytest.raises(RuntimeError, match="unstable state"):
        driver.run(n_steps=2, session=str(tmp_path))
    reader = TraceReader(tmp_path / "sim")
    assert reader.metadata["stop_reason"] == "technical_error"
    assert len(reader.timestamps) == 1  # only the valid initial state
    reader.close()


def test_camera_provenance_freezes_loaded_local_source(tmp_path, monkeypatch):
    import hashlib
    import sys
    from types import ModuleType
    from human.perception.timed_recording import TimedCameraRecorder
    source = tmp_path / "source_repo/local_module.py"
    source.parent.mkdir()
    source.write_text("VALUE = 'working copy'\n")
    module = ModuleType("local_provenance_test")
    module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    camera = TimedCameraRecorder(tmp_path / "operator", {"source_root": str(source.parent)})
    camera.submit(np.zeros((24, 32, 3), dtype=np.uint8), 10)
    camera.close()
    frozen = tmp_path / "operator/source/local_module.py"
    assert frozen.read_bytes() == source.read_bytes()
    assert camera.metadata["provenance"]["source_sha256"]["local_module.py"] == hashlib.sha256(source.read_bytes()).hexdigest()


@pytest.mark.parametrize("camera_issue", ["early_stop", "incomplete", "missing"])
def test_failed_interruption_stays_in_denominator_despite_camera_issue(tmp_path, camera_issue):
    success = make_session(tmp_path / "success")
    failure = make_session(tmp_path / "failure", reason="manual_interruption")
    op_path = failure / "operator/metadata.json"
    operator = json.loads(op_path.read_text())
    if camera_issue == "missing":
        op_path.unlink()
    else:
        if camera_issue == "early_stop":
            sim_path = failure / "sim/metadata.json"
            sim = json.loads(sim_path.read_text())
            sim["end_ns"] = 60_000_000_000  # camera ends 55 seconds early
            sim_path.write_text(json.dumps(sim))
        else:
            operator["complete"] = False
        op_path.write_text(json.dumps(operator))
    report = consolidate([success, failure])
    row = report["trials"][1]
    assert row["eligible"]
    assert not row["evidence_eligible"]
    assert row["evidence_warnings"]
    group = report["groups"][0]
    assert group["evaluable_trials"] == 2
    assert group["successful_trials"] == 1
    assert group["failed_trials"] == 1
    assert group["manual_interrupted_trials"] == 1
    assert group["excluded_trials"] == 0
    assert group["evidence_complete_trials"] == 1
    assert group["success_rate"] == .5


def test_manual_stop_is_failure_even_with_raw_success(tmp_path):
    session = make_session(tmp_path / "interrupted", reason="manual_interruption")
    sim_path = session / "sim/metadata.json"
    sim = json.loads(sim_path.read_text())
    sim["evaluation"].update(success_ever=True, time_to_success_wall_s=1.0)
    sim_path.write_text(json.dumps(sim))
    group = consolidate([session])["groups"][0]
    assert group["successful_trials"] == 0
    assert group["failed_trials"] == 1
    assert group["success_rate"] == 0
    assert group["median_time_to_success_wall_s"] is None


@pytest.mark.parametrize("reason,started,complete", [
    ("technical_error", True, True),
    ("success", False, True),
    ("manual_interruption", True, False),
])
def test_camera_rule_does_not_admit_unstarted_or_technical_trials(tmp_path, reason, started, complete):
    session = make_session(tmp_path / "excluded", reason=reason, started=started)
    sim_path = session / "sim/metadata.json"
    sim = json.loads(sim_path.read_text())
    sim["complete"] = complete
    sim_path.write_text(json.dumps(sim))
    row = session_summary(session)
    assert not row["eligible"]
    assert row["exclusion_reasons"]


@pytest.mark.parametrize("started", [True, False])
def test_operator_declared_formal_abort_counts_as_failure_with_or_without_f5(tmp_path, started):
    success = make_session(tmp_path / "success")
    failure = make_session(tmp_path / "aborted", reason="manual_interruption", started=started)
    report = consolidate([success, failure])
    row = report["trials"][1]
    assert row["eligible"]
    assert row["evaluation"]["time_to_success_wall_s"] is None
    if not started:
        assert "evaluation_not_started" in row["evidence_warnings"]
        assert row["evaluation"]["start_monotonic_ns"] is None
    group = report["groups"][0]
    assert group["success_rate"] == .5
    assert group["evaluable_trials"] == 2
    assert group["failed_trials"] == 1
    assert group["manual_interrupted_trials"] == 1
    assert group["excluded_trials"] == 0
