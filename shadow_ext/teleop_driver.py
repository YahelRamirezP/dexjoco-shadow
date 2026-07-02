"""Stage-1 functional teleop driver for Panda+Shadow (no RL machinery).

What it is: the lean path to a task-success METRIC for evaluating the retargeter
in sim. It loads the attached Panda+Shadow scene, holds the wrist fixed via the
Panda OSC controller (stage 1: no wrist motion), and streams finger angles to the
Shadow actuators through the 24->20 coupling map. Each step it reports:

  - grasp_lift: object lifted >= LIFT_THRESH off the table (grasp success, the
    metric available at stage 1 with a fixed wrist).
  - bucket_place: object inside the bucket footprint AND lifted (the full
    DexJoCo task; only reachable once wrist translation lands -> stage 3).

What it is NOT: a Gymnasium env. No cameras, no obs space, no reward shaping, no
domain randomization. Those belong to RL data collection, which is a separate
goal (copy the gym env then).

The finger input here is a hand-tuned close target (a placeholder for the live
retargeter qpos). Swapping in the retargeter = replace `finger_target_qpos()`
with a per-frame qpos stream; the mapping + ctrl path is unchanged.

Usage:
    python -m shadow_ext.teleop_driver            # headless, prints metrics
    python -m shadow_ext.teleop_driver --view     # + viewer
"""
from __future__ import annotations
import json
import socket
import sys
import threading
import time
import numpy as np
import mujoco

try:
    from pynput import keyboard as _kb
    _PYNPUT_OK = True
except ImportError:
    _PYNPUT_OK = False

from dexjoco.sim.controllers import opspace

from .build import build_spec
from .mapping import build_finger_map, qpos_to_ctrl
from .tasks import REGISTRY
from .recorder import Recorder

# Shadow 24-DOF order emitted by the retargeter (Menagerie order, matches
# _SHADOW_LOWER/_UPPER in retarget.py). First 2 (wrist) are padded to 0 there.
# The streaming receiver scatters an incoming [24] vector to model qpos by NAME.
_SHADOW_24_ORDER = (
    "WRJ2", "WRJ1",
    "FFJ4", "FFJ3", "FFJ2", "FFJ1",
    "MFJ4", "MFJ3", "MFJ2", "MFJ1",
    "RFJ4", "RFJ3", "RFJ2", "RFJ1",
    "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",
    "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",
)
_FINGER_UDP_PORT = 5014   # matches sim_teleop hand port (right hand)
_WRIST_UDP_PORT = 5012    # matches sim_teleop VIVE/wrist port
_WRIST_POSE_SCALE = 1.0   # delta-translation gain (sim_teleop uses 1.5 for VIVE)

# Fixed alignment Dong/WiLoR wrist frame -> Panda flange frame. Identity until
# calibrated in the viewer (etapa 2): the one empirical value, not a design
# choice. Rotates the relative wrist delta into the flange's axes.
_R_ALIGN = np.eye(3)

_KEY_STEP = 0.005   # metres per keypress


class _KeyController:
    """Non-blocking keyboard arm-position control (requires pynput).

    Keys (work even when MuJoCo viewer has focus):
        W/S  -> +Y / -Y   (forward / back)
        A/D  -> -X / +X   (left / right)
        Q/E  -> +Z / -Z   (up / down)
        R    -> reset delta to zero
    """

    _MAP = {
        'w': np.array([ 0,  1,  0], dtype=float),
        's': np.array([ 0, -1,  0], dtype=float),
        'a': np.array([-1,  0,  0], dtype=float),
        'd': np.array([ 1,  0,  0], dtype=float),
        'q': np.array([ 0,  0,  1], dtype=float),
        'e': np.array([ 0,  0, -1], dtype=float),
    }

    def __init__(self, step: float = _KEY_STEP):
        self._step = step
        self._lock = threading.Lock()
        self._delta = np.zeros(3)
        self._listener = _kb.Listener(on_press=self._on_press)
        self._listener.start()

    def _on_press(self, key):
        try:
            ch = key.char.lower() if hasattr(key, 'char') and key.char else None
        except Exception:
            ch = None
        if ch == 'r':
            with self._lock:
                self._delta[:] = 0.0
        elif ch in self._MAP:
            with self._lock:
                self._delta += self._MAP[ch] * self._step

    def delta(self) -> np.ndarray:
        with self._lock:
            return self._delta.copy()

    def close(self):
        self._listener.stop()


_WAYPOINT_DWELL = 0.5   # seconds to hold each waypoint before moving to next


class _PoseRecorder:
    """Record arm waypoints interactively (Space = save current pos, Ctrl+C = done)."""

    def __init__(self, path: str):
        self._path = path
        self._waypoints: list[dict] = []
        self._pending_save = False
        self._lock = threading.Lock()
        if _PYNPUT_OK:
            self._listener = _kb.Listener(on_press=self._on_press)
            self._listener.start()

    def _on_press(self, key):
        if key == _kb.Key.space:
            with self._lock:
                self._pending_save = True

    def poll_save(self, pos: np.ndarray, quat: np.ndarray) -> bool:
        with self._lock:
            if self._pending_save:
                self._pending_save = False
                wp = {"pos": pos.tolist(), "quat": quat.tolist()}
                self._waypoints.append(wp)
                print(f"  Waypoint {len(self._waypoints)} saved: pos={np.round(pos,3)}")
                return True
        return False

    def save(self):
        with open(self._path, "w") as f:
            json.dump(self._waypoints, f, indent=2)
        print(f"Saved {len(self._waypoints)} waypoints -> {self._path}")

    def close(self):
        if _PYNPUT_OK:
            self._listener.stop()


class _PosePlayback:
    """Play back recorded waypoints, dwelling at each for DWELL seconds."""

    def __init__(self, path: str, dwell: float = _WAYPOINT_DWELL):
        with open(path) as f:
            data = json.load(f)
        self._waypoints = [(np.array(w["pos"]), np.array(w["quat"])) for w in data]
        self._dwell = dwell
        self._idx = 0
        self._t_arrived: float | None = None
        print(f"Loaded {len(self._waypoints)} waypoints from {path}")

    def current(self) -> tuple[np.ndarray, np.ndarray]:
        return self._waypoints[min(self._idx, len(self._waypoints) - 1)]

    def step(self, pos: np.ndarray):
        """Advance to next waypoint when close enough and dwell elapsed."""
        if self._idx >= len(self._waypoints):
            return
        target_pos, _ = self._waypoints[self._idx]
        dist = np.linalg.norm(pos - target_pos)
        if dist < 0.02:
            if self._t_arrived is None:
                self._t_arrived = time.time()
            elif time.time() - self._t_arrived >= self._dwell:
                self._idx = min(self._idx + 1, len(self._waypoints) - 1)
                self._t_arrived = None
        else:
            self._t_arrived = None

    @property
    def done(self) -> bool:
        return self._idx >= len(self._waypoints) - 1


def _pose4x4(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from position + wxyz quaternion."""
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(quat_wxyz, dtype=np.float64))
    t = np.eye(4)
    t[:3, :3] = m.reshape(3, 3)
    t[:3, 3] = pos
    return t


def _mat2quat(r: np.ndarray) -> np.ndarray:
    """wxyz quaternion from a 3x3 rotation matrix."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(r, dtype=np.float64).reshape(9))
    return q

# Panda home (identical to the Allegro env so the arm starts in a known pose).
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))

# Hand-tuned close target, per Shadow finger joint (placeholder for retargeter).
# Coupled distal pair J2,J1 each set here; the map sums them into A_*J0.
_CLOSE_JOINTS = {
    "FFJ3": 1.0, "FFJ2": 0.9, "FFJ1": 0.9,
    "MFJ3": 1.0, "MFJ2": 0.9, "MFJ1": 0.9,
    "RFJ3": 1.0, "RFJ2": 0.9, "RFJ1": 0.9,
    "LFJ3": 1.0, "LFJ2": 0.9, "LFJ1": 0.9,
    "THJ5": 0.3, "THJ4": 1.1, "THJ2": 0.5, "THJ1": 1.0,
}
_JNT_PREFIX = "rh-rh_"


def finger_target_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Build a model-sized qpos with the hand-close target written at the Shadow
    joint addresses (wrist + everything else left at 0)."""
    q = np.zeros(model.nq)
    for jn, val in _CLOSE_JOINTS.items():
        jid = model.joint(f"{_JNT_PREFIX}{jn}").id
        q[model.jnt_qposadr[jid]] = val
    return q


def _panda_ids(model):
    dof = np.asarray([model.joint(f"joint{i}").id for i in range(1, 8)])
    ctrl = np.asarray([model.actuator(f"actuator{i}").id for i in range(1, 8)])
    return dof, ctrl


def wrist_sweep_pose(k: int, n_steps: int, home_pos: np.ndarray,
                     home_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stage-0 foundation check (decoupled from the live path): a known
    time-varying wrist target around the welded home pose. Translation = small
    lissajous; orientation = oscillating tilt about a rotating axis. Drives the
    mocap so we can confirm in the viewer that the Panda flange TRACKS both
    position AND orientation through OSC before any WiLoR/UDP wiring exists.

    No WiLoR, no network: pure proof that mocap->arm follows rotation, the one
    thing the held-fixed stage-1 path never exercised.
    """
    t = k / max(1, n_steps)               # 0..1 over the run
    # Translation: lissajous, ~8 cm envelope.
    amp = 0.08
    dpos = np.array([
        amp * np.sin(2 * np.pi * 1.0 * t),
        amp * np.sin(2 * np.pi * 2.0 * t),
        0.5 * amp * np.sin(2 * np.pi * 1.5 * t),
    ])
    pos = home_pos + dpos
    # Orientation: angle oscillates up to ~0.5 rad about a slowly rotating axis.
    angle = 0.5 * np.sin(2 * np.pi * 1.0 * t)
    axis = np.array([np.cos(2 * np.pi * 0.5 * t), np.sin(2 * np.pi * 0.5 * t), 0.0])
    dquat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(dquat, axis, angle)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, home_quat, dquat)
    return pos, quat


def build_qpos24_scatter(model: mujoco.MjModel) -> np.ndarray:
    """Return addrs[24]: model qpos address for each Shadow joint in retargeter
    (Menagerie) order, so a streamed [24] vector scatters into a model qpos."""
    addrs = []
    for jn in _SHADOW_24_ORDER:
        jid = model.joint(f"{_JNT_PREFIX}{jn}").id
        addrs.append(int(model.jnt_qposadr[jid]))
    return np.asarray(addrs, dtype=int)


class FingerReceiver:
    """Stage-1 streaming receiver: a daemon UDP thread that holds the latest
    retargeter qpos[24] (float64, Menagerie order). Decoupled — only started
    when --recv-fingers is set; the default hand-tuned path never touches it."""

    def __init__(self, port: int = _FINGER_UDP_PORT, host: str = "127.0.0.1"):
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._stop = False
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(0.1)
        self._sock.bind((host, port))
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            try:
                data, _ = self._sock.recvfrom(4096)
                if not data:
                    continue
                q = np.frombuffer(data, dtype=np.float64)
                if q.size < 24:
                    continue
                with self._lock:
                    self._latest = q[:24].copy()
            except socket.timeout:
                continue
            except Exception:
                pass

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def close(self):
        self._stop = True
        try:
            self._sock.close()
        except Exception:
            pass


class WristReceiver:
    """Stage-2 streaming receiver: a daemon UDP thread holding the latest wrist
    pose as a 4x4 transform (12 float64 = 3x4 [R|t], the sim_teleop VIVE wire
    format). Decoupled — only started with --recv-wrist; default path holds the
    wrist welded to the home flange pose."""

    def __init__(self, port: int = _WRIST_UDP_PORT, host: str = "127.0.0.1"):
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._stop = False
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(0.1)
        self._sock.bind((host, port))
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            try:
                data, _ = self._sock.recvfrom(2048)
                if len(data) < 12 * 8:
                    continue
                pose = np.frombuffer(data, dtype=np.float64, count=12).reshape(3, 4)
                t = np.eye(4)
                t[:3, :] = pose
                with self._lock:
                    self._latest = t
            except socket.timeout:
                continue
            except Exception:
                pass

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def close(self):
        self._stop = True
        try:
            self._sock.close()
        except Exception:
            pass


def wrist_target(tracker_now: np.ndarray, tracker_start: np.ndarray,
                 ee_start: np.ndarray, r_align: np.ndarray = _R_ALIGN,
                 pose_scale: float = _WRIST_POSE_SCALE) -> tuple[np.ndarray, np.ndarray]:
    """Map a streamed wrist pose to a flange mocap target (sim_teleop VIVE scheme).

    delta = inv(start) @ now is the wrist motion since the anchor frame; r_align
    rotates that delta into the flange's axes; the result is applied to the flange
    pose captured at anchor time. Returns (pos, quat_wxyz).
    """
    delta = np.linalg.inv(tracker_start) @ tracker_now
    dR = r_align @ delta[:3, :3] @ r_align.T
    dt = pose_scale * (r_align @ delta[:3, 3])
    delta_aligned = np.eye(4)
    delta_aligned[:3, :3] = dR
    delta_aligned[:3, 3] = dt
    target = ee_start @ delta_aligned
    return target[:3, 3].copy(), _mat2quat(target[:3, :3])


def run(task_name: str = "pick_bucket", view: bool = False, n_steps: int = 1500,
        record: str | None = None, shot: str | None = None, cam: str = "front",
        sweep_wrist: bool = False, recv_fingers: bool = False,
        recv_wrist: bool = False, orient_only: bool = False,
        wait_anchor: bool = False, key_control: bool = False,
        record_poses: str | None = None, playback_poses: str | None = None):
    task = REGISTRY[task_name]
    model = build_spec(task.arena).compile()
    data = mujoco.MjData(model)

    rec = Recorder(model, cam=cam) if (record or shot) else None

    panda_dof, panda_ctrl = _panda_ids(model)
    site_id = (model.site("attachment_site") or model.site("attachment_site_right")).id
    # Panda wrist target mocap, resolved BY NAME (not index 0): hammer scenes have
    # a second mocap (the nail) that may sit at index 0.
    mocap = int(model.body("target").mocapid[0])
    fing_ids, fing_plan = build_finger_map(model)
    fing_ctrlrange = model.actuator_ctrlrange[fing_ids].copy()

    # Home the arm, settle, then weld the wrist target to the current flange pose.
    data.qpos[panda_dof] = _PANDA_HOME
    mujoco.mj_forward(model, data)
    data.mocap_pos[mocap] = data.sensor("franka/flange_pos").data.copy()
    data.mocap_quat[mocap] = data.sensor("franka/flange_quat").data.copy()
    home_pos = data.mocap_pos[mocap].copy()
    home_quat = data.mocap_quat[mocap].copy()

    # Finger close target, mapped 24 joints -> 20 ctrl, clipped to actuator range.
    q_target = finger_target_qpos(model)
    ctrl_close = np.clip(qpos_to_ctrl(q_target, fing_ids, fing_plan),
                         fing_ctrlrange[:, 0], fing_ctrlrange[:, 1])

    # Stage 1 (flagged): live retargeter qpos over UDP 5014 instead of the
    # hand-tuned close target. Scatter [24] (Menagerie order) -> model qpos.
    fing_rx = FingerReceiver() if recv_fingers else None
    q24_addrs = build_qpos24_scatter(model) if recv_fingers else None
    q_scratch = np.zeros(model.nq) if recv_fingers else None

    # Stage 2 (flagged): live wrist pose over UDP 5012 (VIVE-style relative
    # delta from the first received frame). ee_start = flange pose at anchor.
    wrist_rx = WristReceiver() if recv_wrist else None
    ee_start = _pose4x4(home_pos, home_quat) if recv_wrist else None
    tracker_start: np.ndarray | None = None

    if recv_wrist and wait_anchor:
        print("Hold your hand in the desired neutral pose, then press Enter to set anchor...")
        input()
        # Drain stale packets; next packet becomes the anchor
        if wrist_rx is not None:
            with wrist_rx._lock:
                wrist_rx._latest = None

    key_ctrl = None
    if key_control or record_poses:
        if not _PYNPUT_OK:
            print("WARNING: pynput not installed. Install with: pip install pynput")
        else:
            key_ctrl = _KeyController()
            print("Keyboard arm control active: W/S=Y  A/D=X  Q/E=Z  R=reset")

    pose_rec = _PoseRecorder(record_poses) if record_poses else None
    pose_pb  = _PosePlayback(playback_poses) if playback_poses else None

    if pose_rec:
        print("RECORD MODE: move arm with WASD/QE, press Space to save waypoint, Ctrl+C to finish.")

    task.reset()
    succeeded = False

    viewer = mujoco.viewer.launch_passive(model, data) if view else None
    try:
        for k in range(n_steps):
            # Fingers: streamed retargeter qpos (flagged) or hand-tuned ramp.
            if fing_rx is not None:
                q24 = fing_rx.latest()
                if q24 is not None:
                    q_scratch[q24_addrs] = q24
                    data.ctrl[fing_ids] = np.clip(
                        qpos_to_ctrl(q_scratch, fing_ids, fing_plan),
                        fing_ctrlrange[:, 0], fing_ctrlrange[:, 1])
                # else: no packet yet -> hold last ctrl (hand starts open at 0)
            else:
                alpha = min(1.0, k / (n_steps / 3.0))
                data.ctrl[fing_ids] = alpha * ctrl_close

            # Stage 0 (flagged): drive the mocap with a known moving+rotating
            # target to verify the arm tracks it. Default path stays held-fixed.
            if sweep_wrist:
                p, q = wrist_sweep_pose(k, n_steps, home_pos, home_quat)
                data.mocap_pos[mocap] = p
                data.mocap_quat[mocap] = q

            # Stage 2 (flagged): live wrist pose -> mocap (anchor on 1st packet).
            if wrist_rx is not None:
                tnow = wrist_rx.latest()
                if tnow is not None:
                    if tracker_start is None:
                        tracker_start = tnow
                    scale = 0.0 if orient_only else _WRIST_POSE_SCALE
                    p, q = wrist_target(tnow, tracker_start, ee_start,
                                        pose_scale=scale)
                    data.mocap_pos[mocap] = p
                    data.mocap_quat[mocap] = q

            # Keyboard arm control: shift mocap position by accumulated delta.
            if key_ctrl is not None:
                data.mocap_pos[mocap] = home_pos + key_ctrl.delta()

            # Pose recorder: Space saves current mocap pos+quat as waypoint.
            if pose_rec is not None:
                pose_rec.poll_save(data.mocap_pos[mocap].copy(),
                                   data.mocap_quat[mocap].copy())

            # Pose playback: drive mocap through recorded waypoints.
            if pose_pb is not None:
                p, q = pose_pb.current()
                data.mocap_pos[mocap] = p
                data.mocap_quat[mocap] = q
                pose_pb.step(data.mocap_pos[mocap])

            # Arm: OSC hold/track the wrist target (stage 1 = no wrist motion).
            tau = opspace(
                model=model, data=data, site_id=site_id, dof_ids=panda_dof,
                pos=data.mocap_pos[mocap], ori=data.mocap_quat[mocap], joint=_PANDA_HOME,
                gravity_comp=True, pos_gains=(400.0, 400.0, 400.0), damping_ratio=4,
            )
            data.ctrl[panda_ctrl] = tau

            mujoco.mj_step(model, data)
            succeeded = task.update(model, data) or succeeded
            if rec is not None and record:
                rec.maybe_capture(data, k)
            if viewer is not None:
                viewer.sync()
    finally:
        if viewer is not None:
            viewer.close()
        if fing_rx is not None:
            fing_rx.close()
        if wrist_rx is not None:
            wrist_rx.close()
        if key_ctrl is not None:
            key_ctrl.close()
        if pose_rec is not None:
            pose_rec.save()
            pose_rec.close()

    if rec is not None:
        if shot:
            rec.shot(data, shot)
            print(f"shot saved: {shot}")
        if record:
            rec.save_video(record)
            print(f"video saved: {record}")
        rec.close()

    print(f"task={task_name} steps={n_steps}")
    print(f"succeed (DexJoCo metric): {succeeded}")
    return succeeded


_VALUE_FLAGS = ("--record", "--shot", "--cam", "--n-steps",
                "--record-poses", "--playback-poses")


def _parse(argv):
    """Return (positionals, flags_with_values). --view is a bare flag."""
    pos, flags, i = [], {}, 0
    while i < len(argv):
        a = argv[i]
        if a in _VALUE_FLAGS and i + 1 < len(argv):
            flags[a] = argv[i + 1]
            i += 2
        elif a.startswith("--"):
            flags[a] = True
            i += 1
        else:
            pos.append(a)
            i += 1
    return pos, flags


if __name__ == "__main__":
    pos, flags = _parse(sys.argv[1:])
    task_name = pos[0] if pos else "pick_bucket"
    if "--view" in flags:
        import mujoco.viewer  # noqa: F401
    run(
        task_name=task_name,
        view="--view" in flags,
        n_steps=int(flags.get("--n-steps", 1500)),
        record=flags.get("--record"),
        shot=flags.get("--shot"),
        cam=flags.get("--cam", "front"),
        sweep_wrist="--sweep-wrist" in flags,
        recv_fingers="--recv-fingers" in flags,
        recv_wrist="--recv-wrist" in flags,
        orient_only="--orient-only" in flags,
        wait_anchor="--wait-anchor" in flags,
        key_control="--key-control" in flags,
        record_poses=flags.get("--record-poses"),
        playback_poses=flags.get("--playback-poses"),
    )
