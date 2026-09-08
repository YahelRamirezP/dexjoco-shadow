"""Recording contracts: state fidelity, task outcomes, and synchronized exports."""
import csv
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "AIST-hand"))

from shadow_ext.state_recording import StateTrace, TraceReader
from human.perception.timed_recording import TimedCameraRecorder

XML = """
<mujoco>
 <size nuserdata="2"/>
 <visual><global offwidth="320" offheight="240"/></visual>
 <worldbody>
  <light pos="0 0 3"/>
  <camera name="front" pos="0 -3 1" xyaxes="1 0 0 0 0.3 1"/>
  <camera name="left" pos="-3 0 1" xyaxes="0 -1 0 0.3 0 1"/>
  <geom type="plane" size="2 2 .1"/>
  <body name="arm" pos="0 0 1">
   <joint name="arm_joint" type="hinge"/>
   <geom type="capsule" size=".05" fromto="0 0 0 .3 0 0"/>
   <body name="finger" pos=".3 0 0">
    <joint name="finger_joint" type="hinge"/>
    <geom type="capsule" size=".03" fromto="0 0 0 .1 0 0"/>
   </body>
  </body>
  <body name="object" pos="0 .2 .2">
   <freejoint/><geom type="box" size=".05 .05 .05"/>
  </body>
  <body name="target" mocap="true" pos=".2 0 1">
   <geom type="sphere" size=".01" contype="0" conaffinity="0"/>
  </body>
 </worldbody>
 <actuator>
  <position joint="arm_joint" kp="2"/>
  <general joint="finger_joint" dyntype="filter" dynprm=".02"/>
 </actuator>
</mujoco>
"""


def test_full_state_roundtrip_and_success(tmp_path):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    trace = StateTrace(tmp_path / "sim", model, {"hand": "shadow"}, chunk_size=2)
    expected = []
    for k in range(5):
        data.ctrl[:] = [.1 * k, .2 * k]
        data.mocap_pos[0] = [.2 + .01 * k, 0, 1]
        data.mocap_quat[0] = [np.cos(k*.01), 0, 0, np.sin(k*.01)]
        data.userdata[:] = [k, -k]
        data.qfrc_applied[0] = .3 * k
        data.xfrc_applied[model.body("object").id, 0] = .1 * k
        model.body_gravcomp[model.body("object").id] = k / 4
        mujoco.mj_step(model, data)
        state = np.empty(mujoco.mj_stateSize(model, trace.spec))
        mujoco.mj_getState(model, data, state, trace.spec)
        expected.append((state.copy(), model.body_gravcomp.copy()))
        trace.append(data, target_ctrl=data.ctrl, slip_bias=np.zeros(2),
                     success=k >= 3, timestamp_ns=1_000_000_000 + k * 100_000_000)
    trace.close("success")
    reader = TraceReader(tmp_path / "sim")
    for k in [4, 0, 3, 1, 2]:  # cross chunks and seek backwards
        restored = reader.restore(k)
        actual = np.empty_like(expected[k][0])
        mujoco.mj_getState(reader.model, restored, actual, reader.spec)
        np.testing.assert_array_equal(actual, expected[k][0])
        np.testing.assert_array_equal(reader.model.body_gravcomp, expected[k][1])
        assert reader.success_at(k) == (k >= 3)
    assert reader.metadata["time_to_success_wall_s"] == .3
    assert reader.metadata["stop_reason"] == "success"
    assert reader.index_at(1_150_000_000) == 1
    reader.close()
    with pytest.raises(FileExistsError):
        StateTrace(tmp_path / "sim", model)


def test_incomplete_trace_recovers_only_flushed_chunks(tmp_path):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    trace = StateTrace(tmp_path / "sim", model, chunk_size=2)
    for k in range(3):
        trace.append(data, target_ctrl=[], slip_bias=[], timestamp_ns=k)
    reader = TraceReader(tmp_path / "sim")
    assert len(reader.timestamps) == 2
    assert reader.metadata["complete"] is False
    reader.close()
    trace.close("manual_interruption")
    reader = TraceReader(tmp_path / "sim")
    assert len(reader.timestamps) == 3
    assert reader.metadata["stop_reason"] == "manual_interruption"
    reader.close()


def test_camera_and_two_views_share_timeline_without_physics(tmp_path, monkeypatch):
    from shadow_ext.replay import OperatorVideo, render_trace
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    trace = StateTrace(tmp_path / "sim", model, {"hand": "shadow"})
    camera = TimedCameraRecorder(tmp_path / "operator", {"robot": "shadow"})
    for k in range(5):
        stamp = 1_000_000_000 + k * 100_000_000
        frame = np.full((48, 64, 3), k * 40, dtype=np.uint8)
        camera.submit(frame, stamp)
        data.qpos[0] = .1 * k
        data.qpos[1] = -.1 * k
        data.qpos[2] = k * .02  # free object translation
        data.time = k * .01
        trace.append(data, target_ctrl=data.ctrl, slip_bias=np.zeros(2),
                     success=k == 4, timestamp_ns=stamp)
    camera.close()
    trace.close("success")
    op = OperatorVideo(tmp_path / "operator", trace.metadata)
    assert op.at(999_999_999) is None
    assert abs(float(op.at(1_150_000_000).mean()) - 40) < 3
    assert op.index == 1
    assert op.at(1_400_000_001) is None
    op.close()
    def no_dynamics(*args, **kwargs):
        raise AssertionError("Replay must never advance the physics")
    monkeypatch.setattr(mujoco, "mj_step", no_dynamics)
    paths = render_trace(tmp_path / "sim", tmp_path / "render" / "take.mp4",
                         "front,left", 160, 120, 10, with_operator=True)
    assert len(paths) == 5
    for path in paths:
        cap = cv2.VideoCapture(path)
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(10)
        cap.release()
    with (tmp_path / "render" / "take_timeline.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["simulation_state_index"]) for r in rows] == list(range(5))
    assert [int(r["operator_frame_index"]) for r in rows] == list(range(5))
    assert rows[-1]["task_success"] == "True"
    with pytest.raises(FileExistsError):
        render_trace(tmp_path / "sim", tmp_path / "render" / "take.mp4",
                     "front,left", 160, 120, 10)


@pytest.mark.parametrize("ending", ["success", "manual_interruption", "technical_error"])
def test_driver_finalizes_outcome_after_task_update(tmp_path, monkeypatch, ending):
    from shadow_ext import teleop_driver as driver
    class RecordingTask:
        arena = "arena_arm_hand_bucket_pick.xml"
        def reset(self):
            self.calls = 0
        def update(self, model, data):
            self.calls += 1
            if self.calls == 3:
                if ending == "manual_interruption":
                    raise KeyboardInterrupt
                if ending == "technical_error":
                    raise RuntimeError("Synthetic task failure")
            # Mimics a task-side mocap update; capture must preserve this change.
            data.mocap_pos[0, 0] = .01 * self.calls
            return self.calls == 3
    monkeypatch.setitem(driver.REGISTRY, "recording_test", RecordingTask())
    if ending == "technical_error":
        with pytest.raises(RuntimeError, match="Synthetic task failure"):
            driver.run(task_name="recording_test", n_steps=10, session=str(tmp_path))
    else:
        driver.run(task_name="recording_test", n_steps=10, session=str(tmp_path))
    reader = TraceReader(tmp_path / "sim")
    assert reader.metadata["complete"]
    assert reader.metadata["stop_reason"] == ending
    assert reader.metadata["success_ever"] == (ending == "success")
    data = reader.restore(len(reader.timestamps) - 1)
    assert data.mocap_pos[0, 0] == pytest.approx(.03 if ending == "success" else .02)
    reader.close()
