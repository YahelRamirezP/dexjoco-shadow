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

Trial window: a live session (--view, or --recv-fingers/--recv-wrist) runs
paced to wall-clock time, for `--duration` seconds (default 60.0), not a raw
step count -- one CLI invocation = one trial, ending in success or timeout.
--n-steps still overrides with a raw step budget for offline/scripted runs
that don't need wall-clock pacing.
    python -m shadow_ext.teleop_driver pick_bucket --view --recv-fingers --recv-wrist --duration 60
"""
from __future__ import annotations
import csv
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
import numpy as np
import mujoco
import yaml

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

_KEY_STEP = 0.005      # metres per keypress
_ROT_STEP = 0.05       # radians per keypress
_APERTURE_STEP = 0.05  # grasp aperture fraction per keypress (0..1 range)

# Canonical grasp: pose_close from the Feix-taxonomy YAML, applied directly
# (no retargeter/teleop path). Menagerie 24-DOF order matches _SHADOW_24_ORDER.
# Lives in the sibling AIST-hand repo, not this one.
_CANONICAL_YAML = (Path(__file__).resolve().parents[2] / "AIST-hand"
                    / "robot" / "hands" / "shadow_hand"
                    / "shadow_hand_canonical_v5_grasp.yaml")
_DEFAULT_GRASP_CLASS = "Parallel Extension"


def load_canonical_open_close_qpos24(class_name: str,
                                      yaml_path: Path = _CANONICAL_YAML,
                                      ) -> tuple[np.ndarray, np.ndarray]:
    """(pose_open[24], pose_close[24]), Menagerie order, for one Feix class.
    Aperture convention from the YAML's own _meta: qpos = (1-a)*open + a*close,
    a in 0..1 (0=open, 1=closed) -- lets the grasp be dialed continuously
    instead of a hard on/off toggle."""
    data = yaml.safe_load(yaml_path.read_text())
    for entry in data.values():
        if isinstance(entry, dict) and entry.get("class_name") == class_name:
            return (np.asarray(entry["pose_open"], dtype=np.float64),
                    np.asarray(entry["pose_close"], dtype=np.float64))
    known = sorted(e["class_name"] for e in data.values() if isinstance(e, dict))
    raise KeyError(f"grasp class {class_name!r} not found. Known: {known}")


_DEFAULT_POSE_FILE = Path(__file__).resolve().parent / "saved_key_pose.json"


def _nonclobber_path(path: Path) -> Path:
    """If `path` already exists, insert a timestamp before the extension so
    a save NEVER overwrites a previous one. Returns `path` unchanged if
    nothing there yet. NOTE: only for one-shot recordings (waypoint
    sequences) where losing a prior file is costly -- NOT for
    save_key_pose(), which is deliberately overwrite-in-place (that's the
    whole point of "resume last position": next run's load_key_pose() must
    see the newest save, not a stale timestamped copy)."""
    if not path.exists():
        return path
    ts = time.strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.stem}_{ts}{path.suffix}")


def save_key_pose(path: Path, pos: np.ndarray, quat: np.ndarray, aperture: float):
    path.write_text(json.dumps({
        "pos": pos.tolist(), "quat": quat.tolist(), "aperture": aperture,
    }, indent=2))


def load_key_pose(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    # "aperture" (continuous 0..1) supersedes the old binary "grasp_closed"
    # field; accept old saves too (True/False -> 1.0/0.0).
    aperture = d["aperture"] if "aperture" in d else float(bool(d.get("grasp_closed")))
    return {
        "pos": np.asarray(d["pos"], dtype=np.float64),
        "quat": np.asarray(d["quat"], dtype=np.float64),
        "aperture": float(aperture),
    }


# Fixed camera-axes -> flange-local-axes rotation for the TRANSLATION channel only.
# Root cause (2026-07-04): using the anchor's own live wrist orientation (r0w at
# --wait-anchor time) to rotate the raw cam_t delta made reach direction depend on
# how the hand happened to be held at calibration -- verified with real WiLoR
# captures (see docs/estado_wrist_txty_dead_2026-07-04.md and the reach-axis
# analysis in that session): calibrating palm-frontal put the depth signal on the
# wrist's local z0 (palm-normal) axis, which maps to the flange's local Z (real
# world DOWN at _PANDA_HOME, verified via franka/flange_quat) -> reach felt like
# up/down. Calibrating edge-on ("canto") put the depth signal on local x0 (hand-
# forward), which maps to flange local X (real world horizontal reach at
# _PANDA_HOME) -> reach felt correct. This constant freezes the wrist frame from
# that working edge-on capture as a fixed camera->flange reference for
# translation, so reach no longer depends on anchor hand pose. Derived from ONE
# session's capture (not a proper physical camera-mounting calibration) --
# revisit if reach direction/gain feels off after re-testing.
_R_CAM_TO_FLANGE = np.array([
    [0.43893, -0.89664, -0.058056],
    [0.068165, -0.031198, 0.99719],
    [-0.89593, -0.44165, 0.047427],
]).T

# Grasp diagnostic (--measure-grip): body name of the graspable object per task,
# for hand-object contact force + actuator saturation reporting while a human
# positions/closes the grasp live (via --key-control / --recv-fingers). Only
# mapped for tasks actually verified 2026-07-04; unmapped tasks just skip the
# object-specific readout (no crash).
_GRASP_OBJECT_BODY = {
    "pick_bucket": "boxed_food_0",
}


class _KeyController:
    """Keyboard arm-position/orientation control via the MuJoCo viewer's own
    key_callback (GLFW key events on the viewer window) -- NOT pynput. pynput's
    global hook relies on the X Record extension, which WSLg does not implement,
    so it silently never receives keystrokes there. GLFW window-focused events
    work under WSLg because they go through the same compositor that renders
    the viewer. Requires --view (keys are only delivered while the viewer window
    has focus); pass `on_key` as `key_callback=` to `mujoco.viewer.launch_passive`.

    Keys use the NUMPAD, not letters -- ALL 26 letters are bound to a native
    MuJoCo render/visualization toggle (wireframe, transparency, joints,
    inertia, camera, etc. -- see mjVISSTRING/mjRNDSTRING in
    src/engine/engine_vis_init.c). `key_callback` fires ADDITIONALLY
    alongside that native handling, it never suppresses it, so any letter we
    pick also flips a MuJoCo view flag (confirmed live 2026-09-06: W toggled
    wireframe, D hid static bodies, etc. -- literally every letter collides).
    Numpad keycodes (320-335) aren't in either table, so they're free.

    Keys (viewer window must have focus, NumLock on):
        8/2  -> +Y / -Y   (forward / back)
        4/6  -> -X / +X   (left / right)
        7/9  -> +Z / -Z   (up / down)
        1/3  -> pitch +/- (tilt palm forward / back, local X axis)
        /(kp)/*(kp) -> yaw +/-   (turn palm left / right, local Z axis)
        -(kp)/+(kp) -> roll +/-  (spin palm CW / CCW, local Y axis)
        UP/DOWN -> close/open the canonical grasp continuously (aperture 0..1)
        0 (kp)     -> snap grasp fully open (0.0) / fully closed (1.0), toggles
        . (kp)     -> save current wrist pose + grasp aperture to --pose-file
        RIGHT/5(kp) -> advance to next --playback-poses waypoint, NOW (manual;
                      grasping takes as long as it takes -- no auto-advance
                      by position/dwell timer while key-control is active)
        LEFT        -> back to previous --playback-poses waypoint (review/redo)
        Enter (kp) -> reset position + orientation delta to zero, hand open
    """

    _KP = {  # GLFW numpad keycodes, named for readability below
        '0': 320, '1': 321, '2': 322, '3': 323, '4': 324, '5': 325,
        '6': 326, '7': 327, '8': 328, '9': 329,
        'DECIMAL': 330, 'DIVIDE': 331, 'MULTIPLY': 332,
        'SUBTRACT': 333, 'ADD': 334, 'ENTER': 335,
    }

    _MAP = {
        _KP['8']: np.array([ 0,  1,  0], dtype=float),
        _KP['2']: np.array([ 0, -1,  0], dtype=float),
        _KP['4']: np.array([-1,  0,  0], dtype=float),
        _KP['6']: np.array([ 1,  0,  0], dtype=float),
        _KP['7']: np.array([ 0,  0,  1], dtype=float),
        _KP['9']: np.array([ 0,  0, -1], dtype=float),
    }
    # axis + sign per rotation key, applied in the flange's local frame.
    _ROT_MAP = {
        _KP['1']: (np.array([1., 0., 0.]),  1.0),
        _KP['3']: (np.array([1., 0., 0.]), -1.0),
        _KP['DIVIDE']: (np.array([0., 0., 1.]),  1.0),
        _KP['MULTIPLY']: (np.array([0., 0., 1.]), -1.0),
        _KP['SUBTRACT']: (np.array([0., 1., 0.]),  1.0),
        _KP['ADD']: (np.array([0., 1., 0.]), -1.0),
    }
    _KEY_UP, _KEY_DOWN = 265, 264   # GLFW_KEY_UP / GLFW_KEY_DOWN (arrows: also free)
    _KEY_RIGHT, _KEY_LEFT = 262, 263  # GLFW_KEY_RIGHT / GLFW_KEY_LEFT -- waypoint nav

    def __init__(self, step: float = _KEY_STEP, rot_step: float = _ROT_STEP,
                 aperture_step: float = _APERTURE_STEP):
        self._step = step
        self._rot_step = rot_step
        self._aperture_step = aperture_step
        self._lock = threading.Lock()
        self._delta = np.zeros(3)
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])  # accumulated rotation, wxyz
        self._aperture = 0.0  # 0=open, 1=fully closed (canonical pose_close)
        self._save_requested = False
        self._advance_requested = False
        self._retreat_requested = False

    def on_key(self, keycode: int):
        """`key_callback` for `mujoco.viewer.launch_passive` -- runs on the
        viewer's render thread, hence the lock (main loop reads concurrently)."""
        if keycode == self._KP['ENTER']:
            with self._lock:
                self._delta[:] = 0.0
                self._quat[:] = [1.0, 0.0, 0.0, 0.0]
                self._aperture = 0.0
        elif keycode == self._KP['0']:
            with self._lock:
                self._aperture = 0.0 if self._aperture > 0.5 else 1.0
        elif keycode == self._KEY_UP:
            with self._lock:
                self._aperture = min(1.0, self._aperture + self._aperture_step)
        elif keycode == self._KEY_DOWN:
            with self._lock:
                self._aperture = max(0.0, self._aperture - self._aperture_step)
        elif keycode == self._KP['DECIMAL']:
            with self._lock:
                self._save_requested = True
        elif keycode == self._KP['5'] or keycode == self._KEY_RIGHT:
            with self._lock:
                self._advance_requested = True
        elif keycode == self._KEY_LEFT:
            with self._lock:
                self._retreat_requested = True
        elif keycode in self._MAP:
            with self._lock:
                self._delta += self._MAP[keycode] * self._step
        elif keycode in self._ROT_MAP:
            axis, sign = self._ROT_MAP[keycode]
            dquat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dquat, axis, sign * self._rot_step)
            with self._lock:
                new_quat = np.zeros(4)
                mujoco.mju_mulQuat(new_quat, self._quat, dquat)
                self._quat[:] = new_quat

    def delta(self) -> np.ndarray:
        with self._lock:
            return self._delta.copy()

    def quat_delta(self) -> np.ndarray:
        with self._lock:
            return self._quat.copy()

    def aperture(self) -> float:
        with self._lock:
            return self._aperture

    def pop_save_request(self) -> bool:
        with self._lock:
            req, self._save_requested = self._save_requested, False
            return req

    def pop_advance_request(self) -> bool:
        with self._lock:
            req, self._advance_requested = self._advance_requested, False
            return req

    def pop_retreat_request(self) -> bool:
        with self._lock:
            req, self._retreat_requested = self._retreat_requested, False
            return req

    def set_aperture(self, val: float):
        with self._lock:
            self._aperture = val

    def close(self):
        pass  # no listener thread to stop -- callback lifetime is the viewer's


_WAYPOINT_DWELL = 0.5   # seconds to hold each waypoint before moving to next


class _PoseRecorder:
    """Record arm waypoints interactively (P = save current pos, Ctrl+C = done).

    Not Space: Space is reserved globally for _ShotController (camera-side
    capture in the emitter process fires on Space too, so both processes
    react to the same physical keypress instead of needing two different
    keys pressed near-simultaneously).
    """

    def __init__(self, path: str):
        self._path = path
        self._waypoints: list[dict] = []
        self._pending_save = False
        self._lock = threading.Lock()
        if _PYNPUT_OK:
            self._listener = _kb.Listener(on_press=self._on_press)
            self._listener.start()

    def _on_press(self, key):
        try:
            ch = key.char.lower() if hasattr(key, 'char') and key.char else None
        except Exception:
            ch = None
        if ch == 'p':
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
        path = _nonclobber_path(Path(self._path))
        with open(path, "w") as f:
            json.dump(self._waypoints, f, indent=2)
        print(f"Saved {len(self._waypoints)} waypoints -> {path}")

    def close(self):
        if _PYNPUT_OK:
            self._listener.stop()


class _PosePlayback:
    """Play back recorded waypoints, dwelling at each for DWELL seconds.

    On advance()/retreat(), the TARGET itself ramps smoothly from wherever it
    currently is to the new waypoint over _TRANSIT_TIME seconds (position
    lerp + orientation nlerp, sign-fixed for quaternion double cover -- same
    technique already used for wrist_quat_filter above). Jumping the raw
    mocap target instantly and letting the stiff OSC (kp=400) chase a
    suddenly-far target produced a fast, overshooting "arrives with inertia
    and crashes into things" motion (confirmed live 2026-09-06, every
    waypoint). Auto-advance (no --key-control, via step()) still snaps
    straight to the raw waypoint -- unattended playback has no one to react
    to a crash, so a fixed-time ramp isn't obviously safer there either.
    """

    _TRANSIT_TIME = 2.0  # seconds to ramp between waypoints on manual advance/retreat

    def __init__(self, path: str, dwell: float = _WAYPOINT_DWELL):
        with open(path) as f:
            data = json.load(f)
        self._waypoints = [(np.array(w["pos"]), np.array(w["quat"])) for w in data]
        self._dwell = dwell
        self._idx = 0
        self._t_arrived: float | None = None
        self._transit_from: tuple[np.ndarray, np.ndarray] = self._waypoints[0]
        self._transit_start: float | None = None
        print(f"Loaded {len(self._waypoints)} waypoints from {path}")

    def current(self) -> tuple[np.ndarray, np.ndarray]:
        target_pos, target_quat = self._waypoints[min(self._idx, len(self._waypoints) - 1)]
        if self._transit_start is None:
            return target_pos, target_quat
        t = (time.time() - self._transit_start) / self._TRANSIT_TIME
        if t >= 1.0:
            self._transit_start = None
            return target_pos, target_quat
        # smoothstep (3t^2 - 2t^3), not raw t: a LINEAR ramp has the target's
        # velocity hold constant then drop to zero instantly at arrival --
        # the arm has real momentum matching that speed and overshoots into
        # whatever's there right as the ramp stops (confirmed live
        # 2026-09-06, "rebote"/slap on arrival). Smoothstep has zero velocity
        # at both t=0 and t=1, so there's no sudden stop for the arm to
        # overshoot past.
        s = 3.0 * t * t - 2.0 * t * t * t
        from_pos, from_quat = self._transit_from
        pos = (1.0 - s) * from_pos + s * target_pos
        q1 = target_quat if np.dot(from_quat, target_quat) >= 0.0 else -target_quat
        quat = (1.0 - s) * from_quat + s * q1
        quat /= np.linalg.norm(quat)
        return pos, quat

    def _start_transit(self):
        # ramp FROM wherever we are right now, mid-transit-safe (double
        # advance()/retreat() before the previous ramp finished won't jump).
        self._transit_from = self.current()
        self._transit_start = time.time()

    def advance(self):
        """Force-advance to the next waypoint now, ignoring dwell/distance --
        for manual control (grasping takes as long as it takes; auto-advance
        by position alone doesn't know whether the grasp actually succeeded)."""
        self._start_transit()
        self._idx = min(self._idx + 1, len(self._waypoints) - 1)
        self._t_arrived = None

    def retreat(self):
        """Force-retreat to the previous waypoint now (manual review/redo)."""
        self._start_transit()
        self._idx = max(self._idx - 1, 0)
        self._t_arrived = None

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


def _replay_state_log(model, data, rec, record: str | None, task, replay_log: str):
    """Re-render a --state-log trace: pure playback, no dynamics, no pacing.

    Each saved frame is a full data.qpos snapshot from a real (already-completed)
    live run, so setting qpos + mj_forward reproduces that exact instant -- fingers,
    arm, and object all at once, deterministically -- regardless of how the frame
    was originally driven (webcam retargeter, keyboard, or both). This is what
    makes it safe to record the live session with NO --record (full speed, no
    render lag) and only pay the ~15-23x-realtime render cost afterwards, offline,
    once per desired camera set/resolution.
    """
    npz = np.load(replay_log)
    qpos_frames = npz["qpos"]
    task.reset()
    succeeded = False
    t0 = time.time()
    for k in range(qpos_frames.shape[0]):
        data.qpos[:] = qpos_frames[k]
        mujoco.mj_forward(model, data)
        just_succeeded = task.update(model, data)
        succeeded = succeeded or just_succeeded
        if rec is not None and record:
            rec.maybe_capture(data, k)
    elapsed = time.time() - t0
    print(f"[replay] {qpos_frames.shape[0]} frames from {replay_log} in {elapsed:.1f}s wall")
    if rec is not None:
        if record:
            paths = rec.save_video(record)
            print(f"video saved: {paths}")
        rec.close()
    print(f"task={task.name} frames={qpos_frames.shape[0]} (replay)")
    print(f"succeed (DexJoCo metric): {succeeded}")
    return succeeded


class _ShotController:
    """Global spacebar listener for interactive --shot captures (fires
    regardless of window focus, same as _KeyController/_PoseRecorder). Each
    press is consumed once; poll_capture() returns True exactly once per
    physical press. Same physical Space press also fires cv2's capture in
    the emitter process (live_retarget.py), so one keypress = both photos.
    """

    def __init__(self):
        self._pending = False
        self._lock = threading.Lock()
        if _PYNPUT_OK:
            self._listener = _kb.Listener(on_press=self._on_press)
            self._listener.start()

    def _on_press(self, key):
        if key == _kb.Key.space:
            with self._lock:
                self._pending = True

    def poll_capture(self) -> bool:
        with self._lock:
            if self._pending:
                self._pending = False
                return True
        return False

    def close(self):
        if _PYNPUT_OK:
            self._listener.stop()


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


class _OneEuroFilter:
    """One Euro Filter (Casiez et al. 2012): adaptive low-pass -- heavy smoothing
    when the signal is nearly static (rejects jitter), light smoothing when it
    moves fast (low lag for real motion). Used on the wrist target because raw
    WiLoR per-frame orientation has measured jitter of several degrees even
    during pure-translation motion (verified 2026-07-04, no real rotation
    intended in the test capture) -- see docs/estado_wrist_txty_dead_2026-07-04.md.
    Operates componentwise on a flat vector (3 for position; 4 for a quaternion,
    caller must fix antipodal sign before filtering).
    """

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.3, dcutoff: float = 1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._t_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x.copy()
        dt = max(t - self._t_prev, 1e-3)
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.mincutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


def _slerp(qa: np.ndarray, qb: np.ndarray, alpha: float) -> np.ndarray:
    """Quaternion SLERP (wxyz). qb is flipped to qa's hemisphere first (double cover)."""
    if np.dot(qa, qb) < 0.0:
        qb = -qb
    dot = np.clip(np.dot(qa, qb), -1.0, 1.0)
    if dot > 0.9995:
        out = qa + alpha * (qb - qa)
        return out / np.linalg.norm(out)
    theta0 = np.arccos(dot)
    theta = theta0 * alpha
    qc = qb - qa * dot
    qc = qc / np.linalg.norm(qc)
    return qa * np.cos(theta) + qc * np.sin(theta)


class _WristInterpolator:
    """Reconstructs continuous motion between sparse real wrist samples (WiLoR
    updates arrive at ~2.5-4Hz over the network; the physics loop runs far
    faster and would otherwise hold each sample constant until the next one --
    verified 2026-07-04 this reads as a snap-then-freeze staircase once the arm
    is fast enough to fully settle within one hold period). Keeps the last two
    REAL samples and their REAL wall-clock arrival times, and at each physics
    tick LERPs position / SLERPs orientation between them using elapsed time as
    a fraction of the ACTUAL measured inter-arrival gap -- adapts to whatever
    the network is actually doing, no assumed fixed rate. See
    docs/estado_wrist_txty_dead_2026-07-04.md.
    """

    def __init__(self):
        self._p_prev: np.ndarray | None = None
        self._q_prev: np.ndarray | None = None
        self._t_prev: float | None = None
        self._p_new: np.ndarray | None = None
        self._q_new: np.ndarray | None = None
        self._t_new: float | None = None

    def reset(self) -> None:
        self.__init__()

    def push(self, p: np.ndarray, q: np.ndarray, t: float) -> None:
        """Register a genuinely new real sample (call only when the raw tracker
        pose actually changed, not every physics tick)."""
        if self._t_new is None:
            self._p_prev, self._q_prev, self._t_prev = p, q, t
        else:
            self._p_prev, self._q_prev, self._t_prev = self._p_new, self._q_new, self._t_new
        self._p_new, self._q_new, self._t_new = p, q, t

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Current interpolated (pos, quat) at wall-clock time t."""
        span = self._t_new - self._t_prev
        if span <= 1e-6:
            return self._p_new, self._q_new
        alpha = float(np.clip((t - self._t_new) / span, 0.0, 1.0))
        p = self._p_prev + alpha * (self._p_new - self._p_prev)
        q = _slerp(self._q_prev, self._q_new, alpha)
        return p, q


def wrist_target(tracker_now: np.ndarray, tracker_start: np.ndarray,
                 ee_start: np.ndarray, r_align: np.ndarray = _R_ALIGN,
                 pose_scale: float = _WRIST_POSE_SCALE,
                 r_cam_to_flange: np.ndarray = _R_CAM_TO_FLANGE) -> tuple[np.ndarray, np.ndarray]:
    """Map a streamed wrist pose to a flange mocap target (sim_teleop VIVE scheme).

    Orientation: delta = inv(start) @ now is the wrist rotation since the anchor
    frame; r_align rotates that delta into the flange's axes. This is rotation-only
    and frame-invariant, so it doesn't care what the anchor's own orientation was.

    Translation: NOT derived from delta (which would rotate the raw cam_t motion by
    the anchor's own live wrist orientation -- verified 2026-07-04 to make reach
    direction depend on how the hand was held at --wait-anchor time, since a
    frontal-palm anchor sends depth motion down the flange's local Z, which is
    world-DOWN at _PANDA_HOME). Instead, the raw camera-frame displacement is
    rotated by a FIXED camera->flange reference (_R_CAM_TO_FLANGE) so reach maps to
    the same flange axis regardless of anchor hand pose.

    Returns (pos, quat_wxyz).
    """
    delta = np.linalg.inv(tracker_start) @ tracker_now
    dR = r_align @ delta[:3, :3] @ r_align.T
    raw_disp_cam = tracker_now[:3, 3] - tracker_start[:3, 3]
    dt = pose_scale * (r_cam_to_flange @ raw_disp_cam)
    delta_aligned = np.eye(4)
    delta_aligned[:3, :3] = dR
    delta_aligned[:3, 3] = dt
    target = ee_start @ delta_aligned
    return target[:3, 3].copy(), _mat2quat(target[:3, :3])


def run(task_name: str = "pick_bucket", view: bool = False, n_steps: int | None = None,
        duration: float = 60.0,
        record: str | None = None, shot: str | None = None, cam: str = "front",
        sweep_wrist: bool = False, recv_fingers: bool = False,
        recv_wrist: bool = False, orient_only: bool = False,
        wait_anchor: bool = False, key_control: bool = False,
        record_poses: str | None = None, playback_poses: str | None = None,
        grasp_class: str = _DEFAULT_GRASP_CLASS,
        pose_file: str | Path = _DEFAULT_POSE_FILE,
        measure_grip: bool = False, log_grip: str | None = None,
        slip_compensate: bool = False, float_object: bool = False,
        state_log: str | None = None, replay_log: str | None = None):
    task = REGISTRY[task_name]
    model = build_spec(task.arena).compile()
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)
    # For live human teleop, the trial window is a wall-clock duration, not a
    # step count: --n-steps overrides for the scripted/offline paths that still
    # want a fixed step budget, but the default path derives steps from
    # `duration` seconds so one trial = `duration` seconds of real time.
    if n_steps is None:
        n_steps = int(round(duration / dt))

    rec = Recorder(model, cam=cam) if (record or shot) else None

    if replay_log:
        # Offline re-render of a --state-log trace: no live driving, no wall-clock
        # pacing, no receivers -- just set qpos and forward-kinematics it per logged
        # frame, so a live take recorded once (fast, unrendered) can be re-rendered
        # at any camera/resolution/frame-skip afterwards without the live render
        # lag (~15-23x realtime measured 2026-09-07) ever touching the live session.
        return _replay_state_log(model, data, rec, record, task, replay_log)

    panda_dof, panda_ctrl = _panda_ids(model)
    site_id = (model.site("attachment_site") or model.site("attachment_site_right")).id
    # Panda wrist target mocap, resolved BY NAME (not index 0): hammer scenes have
    # a second mocap (the nail) that may sit at index 0.
    mocap = int(model.body("target").mocapid[0])
    fing_ids, fing_plan = build_finger_map(model)
    fing_ctrlrange = model.actuator_ctrlrange[fing_ids].copy()

    # Reactive slip compensation (--slip-compensate): pure proportional (P) control,
    # per finger, on the Coulomb friction-cone margin ratio = tangential/normal
    # contact force (see docs/estado_teleop_grasp_stability_2026-07-04.md +
    # /tmp/grip_slip_log_pick_bucket.csv -- ratio pins at ~1.0=mu right before a
    # contact is lost, confirmed both by the value and by ms-scale contact
    # chattering at that value, the stick-slip signature). No I term (would need
    # an arbitrary reset rule, windup risk); no D term (the signal itself
    # chatters at the limit, differentiating amplifies that noise). Error is
    # one-sided (rectified at 0) so a healthy grip (ratio comfortably < setpoint)
    # gets zero correction and a resolved slip relaxes the bias back to zero on
    # its own, every step, with no memory/reset logic needed.
    _SLIP_RATIO_SETPOINT = 0.85     # start reacting before mu=1.0, not at it
    # Kp=0.05 measured dead in the first live test: max observed error is ~0.15
    # (ratio saturates at exactly mu=1.0, solver clamps it there), so old bias =
    # 0.05*0.15 = 0.0075rad = 0.43deg on a 180deg joint -- physically invisible.
    # 2.0 gives ~17deg at that same worst-case error, still well under bias_max
    # (27deg) so there's headroom left to tune further.
    _SLIP_KP = 2.0                  # rad of ctrl bias per unit of (ratio - setpoint)
    _SLIP_BIAS_MAX = 0.15           # cap: fraction of that actuator's ctrlrange span
    _FINGER_DISTAL_ACT = {"ff": "FFJ0", "mf": "MFJ0", "rf": "RFJ0",
                           "lf": "LFJ0", "th": "THJ1"}
    _slip_ctrl_idx = {}   # finger key -> index into fing_ids/data.ctrl[fing_ids]
    if slip_compensate:
        act_names = [model.actuator(int(a)).name for a in fing_ids]
        for fkey, suffix in _FINGER_DISTAL_ACT.items():
            for idx, name in enumerate(act_names):
                if name.endswith(suffix):
                    _slip_ctrl_idx[fkey] = idx
                    break
    slip_bias = np.zeros(len(fing_ids))

    def _slip_compensate_bias():
        """Return per-actuator ctrl bias (fing_ids-indexed) from the current
        (previous-step) contact state. One-step-delayed feedback: reads
        data.contact as left by the last mj_step, applied before the next one --
        standard discrete control loop, not a bug."""
        bias = np.zeros(len(fing_ids))
        if grip_obj_bodyid is None:
            return bias
        finger_ratio = {}  # finger key -> worst (max) ratio this step
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if grip_obj_bodyid not in (b1, b2):
                continue
            hand_b = b1 if b1 in grip_rh_bodyids else (b2 if b2 in grip_rh_bodyids else None)
            if hand_b is None:
                continue
            bname = model.body(hand_b).name  # e.g. "rh-rh_ffdistal"
            fkey = next((k for k in _FINGER_DISTAL_ACT if bname.endswith(k + "distal")), None)
            if fkey is None or fkey not in _slip_ctrl_idx:
                continue
            f6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f6)
            normal = f6[0]
            if normal < 1e-4:
                continue
            ratio = float(np.hypot(f6[1], f6[2]) / normal)
            finger_ratio[fkey] = max(finger_ratio.get(fkey, 0.0), ratio)
        for fkey, ratio in finger_ratio.items():
            err = max(ratio - _SLIP_RATIO_SETPOINT, 0.0)
            idx = _slip_ctrl_idx[fkey]
            span = fing_ctrlrange[idx, 1] - fing_ctrlrange[idx, 0]
            bias[idx] = min(_SLIP_KP * err, _SLIP_BIAS_MAX * span)
        return bias

    # Grasp diagnostic (--measure-grip): a human positions/closes the grasp live
    # (--key-control for the arm, --recv-fingers for the hand); this just reports
    # hand-object contact force + actuator saturation while they do it, instead
    # of a blind scripted approach (see docs/estado_wrist_txty_dead_2026-07-04.md
    # for why the scripted approach kept producing single-point collisions --
    # the approach orientation itself was wrong, not something a script should
    # guess at when a human can just look at the viewer).
    _grip_needed = measure_grip or slip_compensate or float_object
    grip_obj_name = _GRASP_OBJECT_BODY.get(task_name)
    grip_obj_bodyid = model.body(grip_obj_name).id if (_grip_needed and grip_obj_name) else None
    grip_rh_bodyids = (set(model.body(i).id for i in range(model.nbody) if model.body(i).name.startswith("rh-"))
                       if _grip_needed else None)
    grip_obj_z0 = None
    grip_prev_npts = 0
    grip_max_npts = 0
    grip_pose_saved = False
    grip_gravity_released = False
    grip_save_path = f"/tmp/grip_pose_{task_name}.npz"

    # Float the target object (body_gravcomp=1 cancels ONLY its own weight --
    # global model.opt.gravity untouched, arm/other bodies unaffected) until a
    # real multi-point grip forms, then restore normal gravity for it. Reuses
    # the same npts>=3 contact trigger already validated for grip_pose_saved.
    if float_object and grip_obj_bodyid is not None:
        model.body_gravcomp[grip_obj_bodyid] = 1.0
        print(f"[float] object '{grip_obj_name}' floating (gravcomp=1.0) until grasped")

    # Per-contact slip diagnostic log (--log-grip): raw normal/tangential force
    # split + contact.dist per real hand-object contact per step, so a slip event
    # can be attributed to either (a) Coulomb friction-cone violation (tangential
    # force approaches/exceeds mu*normal) or (b) soft-contact constraint creep
    # (dist drifts toward separation while comfortably inside the friction cone)
    # -- these need two different fixes (reactive grip control vs solref/solimp
    # tuning), so this distinguishes them instead of guessing. total_force above
    # (norm of all 6 components) can't do this: it collapses normal and
    # tangential together and throws away the one split that matters.
    grip_log_path = log_grip if (measure_grip and log_grip) else None
    grip_log_file = open(grip_log_path, "w", newline="") if grip_log_path else None
    grip_log_writer = None
    if grip_log_file is not None:
        grip_log_writer = csv.writer(grip_log_file)
        grip_log_writer.writerow(
            ["step", "t", "body1", "body2", "normal", "tang1", "tang2",
             "tors", "roll1", "roll2", "dist"])

    def _grip_diagnostic(step_k: int):
        if grip_obj_bodyid is None:
            return
        total_force, npts = 0.0, 0
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if grip_obj_bodyid in (b1, b2) and (b1 in grip_rh_bodyids or b2 in grip_rh_bodyids):
                f6 = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, f6)
                total_force += float(np.linalg.norm(f6[:3]))
                npts += 1
                if grip_log_writer is not None:
                    grip_log_writer.writerow(
                        [step_k, f"{data.time:.4f}",
                         model.body(b1).name, model.body(b2).name,
                         *[f"{v:.6f}" for v in f6], f"{c.dist:.6f}"])
        nonlocal grip_obj_z0, grip_prev_npts, grip_max_npts, grip_pose_saved, grip_gravity_released
        obj_z = float(data.xpos[grip_obj_bodyid, 2])
        if grip_obj_z0 is None:
            grip_obj_z0 = obj_z
        fmag = np.abs(data.actuator_force[fing_ids])
        flim = model.actuator_forcerange[fing_ids, 1]
        sat_frac = fmag / np.maximum(flim, 1e-6)
        worst = int(np.argmax(sat_frac))

        # Event-driven: print immediately when the contact point COUNT changes
        # (not every 30 steps -- a throttled sample can miss short-lived real
        # contact between samples, verified 2026-07-04 this buried real events).
        changed = npts != grip_prev_npts
        if changed or step_k % 30 == 0:
            print(f"[grip] contact={total_force:6.1f}N pts={npts}  obj_dz={obj_z-grip_obj_z0:+.3f}m  "
                  f"worst_actuator_sat={sat_frac[worst]*100:5.1f}% ({model.actuator(fing_ids[worst]).name})"
                  + ("  <- NEW MAX pts" if npts > grip_max_npts else ""))
        grip_prev_npts = npts

        # Auto-save the wrist pose + finger ctrl the first time a real multi-
        # point grip forms (>=3 simultaneous contacts), so a good grasp found
        # live (human positioning) can be replayed exactly by a script later --
        # no more guessing hand orientation/approach from scratch.
        if npts > grip_max_npts:
            grip_max_npts = npts
        if npts >= 3 and float_object and not grip_gravity_released:
            model.body_gravcomp[grip_obj_bodyid] = 0.0
            grip_gravity_released = True
            print(f"[float] contact detected (npts={npts}) -> gravity restored for '{grip_obj_name}'")
        if npts >= 3 and not grip_pose_saved:
            np.savez(grip_save_path,
                     mocap_pos=data.mocap_pos[mocap].copy(),
                     mocap_quat=data.mocap_quat[mocap].copy(),
                     finger_ctrl=data.ctrl[fing_ids].copy(),
                     npts=npts, contact_force=total_force)
            grip_pose_saved = True
            print(f"[grip] SAVED pose with {npts} contact points to {grip_save_path}")

    # Home the arm, settle, then weld the wrist target to the current flange pose.
    data.qpos[panda_dof] = _PANDA_HOME
    mujoco.mj_forward(model, data)
    data.mocap_pos[mocap] = data.sensor("franka/flange_pos").data.copy()
    data.mocap_quat[mocap] = data.sensor("franka/flange_quat").data.copy()
    home_pos = data.mocap_pos[mocap].copy()
    home_quat = data.mocap_quat[mocap].copy()

    # Manual mode: resume from a saved wrist pose instead of the arena's default
    # home, so pressing R (reset) or restarting the script doesn't lose your
    # positioning work. Saved via 'P' below into --pose-file.
    pose_file = Path(pose_file)
    loaded_aperture = 0.0
    if key_control:
        saved = load_key_pose(pose_file)
        if saved is not None:
            home_pos, home_quat = saved["pos"], saved["quat"]
            loaded_aperture = saved["aperture"]
            print(f"Resumed saved pose from {pose_file} (aperture={loaded_aperture:.2f})")

    # Finger close target, mapped 24 joints -> 20 ctrl, clipped to actuator range.
    q_target = finger_target_qpos(model)
    ctrl_close = np.clip(qpos_to_ctrl(q_target, fing_ids, fing_plan),
                         fing_ctrlrange[:, 0], fing_ctrlrange[:, 1])

    # Manual mode (--key-control): fingers dial continuously between the
    # canonical class's pose_open (aperture=0) and pose_close (aperture=1) via
    # UP/DOWN. No retargeter -- direct qpos interpolation, then the same
    # 24->20 ctrl mapping used everywhere else.
    q24_addrs_manual = build_qpos24_scatter(model)
    _, _pose_close24 = load_canonical_open_close_qpos24(grasp_class)
    # The YAML's pose_open is the medoid of the LEAST-closed grasps in the
    # dataset (Dexonomy/HOGraspNet has zero true open-hand frames), not a flat
    # extended hand -- e.g. Parallel Extension's pose_open still has several
    # joints > 1.0 rad. Use qpos=0 (full extension) as the real open endpoint.
    _pose_open24 = np.zeros(24)

    def _ctrl_for_aperture(a: float) -> np.ndarray:
        q24 = (1.0 - a) * _pose_open24 + a * _pose_close24
        q_scratch = np.zeros(model.nq)
        q_scratch[q24_addrs_manual] = q24
        return np.clip(qpos_to_ctrl(q_scratch, fing_ids, fing_plan),
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
    # Smooths the raw wrist target before it reaches the OSC: verified 2026-07-04
    # that raw WiLoR per-frame orientation has ~3-4deg (up to ~8deg) jitter even
    # during pure-translation motion (no rotation intended) -- with the arm's
    # damping_ratio now fast (1.0, was 4.0), it faithfully chased that per-frame
    # noise, which read as erratic spinning. See
    # docs/estado_wrist_txty_dead_2026-07-04.md.
    wrist_pos_filter = _OneEuroFilter(mincutoff=1.0, beta=0.3) if recv_wrist else None
    wrist_quat_filter = _OneEuroFilter(mincutoff=1.0, beta=0.3) if recv_wrist else None
    wrist_interp = _WristInterpolator() if recv_wrist else None
    _last_raw_tnow: np.ndarray | None = None

    if recv_wrist and wait_anchor:
        print("Hold your hand in the desired neutral pose, then press Enter to set anchor...")
        input()
        # Drain stale packets; next packet becomes the anchor
        if wrist_rx is not None:
            with wrist_rx._lock:
                wrist_rx._latest = None

    key_ctrl = None
    if key_control or record_poses:
        if key_control and not view:
            print("WARNING: --key-control needs --view (keys are delivered to "
                  "the MuJoCo viewer window, no viewer = no input).")
        key_ctrl = _KeyController()
        key_ctrl.set_aperture(loaded_aperture)
        if key_control:
            print("Keyboard control active (viewer window must have focus, NumLock ON): "
                  "kp8/kp2=Y kp4/kp6=X kp7/kp9=Z (move) kp1/kp3=pitch kp/,kp*=yaw kp-/kp+=roll (rotate) "
                  f"UP/DOWN=grasp aperture ({grasp_class}) kp0=snap open/close "
                  "kp.=save pose kpEnter=reset")
        if record_poses and not _PYNPUT_OK:
            print("WARNING: pynput not installed -- needed for --record-poses "
                  "Space-to-save. Install with: pip install pynput")

    pose_rec = _PoseRecorder(record_poses) if record_poses else None
    pose_pb  = _PosePlayback(playback_poses) if playback_poses else None

    if pose_rec:
        print("RECORD MODE: move arm with WASD/QE, press P to save waypoint, Ctrl+C to finish.")

    # Interactive shot capture: Space saves a numbered still to <shot>_NNN.png
    # (shot doubles as a path prefix). Same physical Space press also fires
    # the emitter's cv2 capture, so one keypress = camera photo + robot photo.
    # Not /tmp: wiped on reboot, and these are real thesis-figure source data.
    if shot:
        os.makedirs(os.path.dirname(os.path.abspath(shot)), exist_ok=True)
    shot_ctrl = _ShotController() if (shot and _PYNPUT_OK) else None
    shot_count = 0
    if shot and not _PYNPUT_OK:
        print("WARNING: pynput not installed, --shot capture on Space unavailable.")
    elif shot:
        print(f"SHOT MODE: press Space anytime to save a still from every camera -> {shot}_NNN_<cam>.png")

    # --state-log: full qpos every physics step, for a later offline
    # _replay_state_log() pass. No rendering happens here, so this adds no
    # wall-clock cost to the live drive -- the whole point is decoupling
    # "drive it well" (fast, live) from "render it pretty" (slow, offline,
    # see recorder.py's measured ~15-23x realtime at 720p/3-cam).
    state_frames = [] if state_log else None

    task.reset()
    succeeded = False
    # Live sessions (viewer open, or a human streaming fingers/wrist over UDP)
    # must run at wall-clock speed: mj_step() alone is far faster than dt, so
    # without pacing the whole n_steps budget elapses before a human's motion
    # ever reaches the sim (root cause of the "closes before I can do anything"
    # symptom). Offline/scripted runs (no viewer, no live input) skip pacing.
    live = view or recv_fingers or recv_wrist

    # Hide the left/right UI panels when driving via key_ctrl: MuJoCo's own
    # visualization/rendering checkboxes have single-letter shortcuts (contact
    # points, joints, transparency, etc.) that fire ALONGSIDE our key_callback,
    # not instead of it -- with the panels visible, WASD/IJKLUO also toggle
    # those layers on/off, confusing what's actually happening in the scene.
    viewer = mujoco.viewer.launch_passive(
        model, data, key_callback=key_ctrl.on_key if key_ctrl is not None else None,
        show_left_ui=key_ctrl is None, show_right_ui=key_ctrl is None,
    ) if view else None
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
            elif key_ctrl is not None:
                data.ctrl[fing_ids] = _ctrl_for_aperture(key_ctrl.aperture())
            else:
                alpha = min(1.0, k / (n_steps / 3.0))
                data.ctrl[fing_ids] = alpha * ctrl_close

            if slip_compensate:
                data.ctrl[fing_ids] = np.clip(data.ctrl[fing_ids] + slip_bias,
                                              fing_ctrlrange[:, 0], fing_ctrlrange[:, 1])

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
                        wrist_pos_filter.reset()
                        wrist_quat_filter.reset()
                        wrist_interp.reset()
                        _last_raw_tnow = None
                    now = time.time()
                    # Only recompute/filter/push when a genuinely NEW packet arrived
                    # (wrist_rx.latest() otherwise keeps returning the same stale
                    # array every physics tick, far faster than the network updates).
                    if _last_raw_tnow is None or not np.array_equal(tnow, _last_raw_tnow):
                        _last_raw_tnow = tnow
                        scale = 0.0 if orient_only else _WRIST_POSE_SCALE
                        p_raw, q_raw = wrist_target(tnow, tracker_start, ee_start,
                                                    pose_scale=scale)
                        p_f = wrist_pos_filter(p_raw, now)
                        # Quaternions have double cover (q and -q are the same
                        # rotation); flip to the previous hemisphere before
                        # filtering so the filter doesn't average antipodal values.
                        q_prev = wrist_quat_filter._x_prev
                        if q_prev is not None and np.dot(q_raw, q_prev) < 0.0:
                            q_raw = -q_raw
                        q_f = wrist_quat_filter(q_raw, now)
                        q_f = q_f / np.linalg.norm(q_f)
                        wrist_interp.push(p_f, q_f, now)
                    p, q = wrist_interp.sample(now)
                    data.mocap_pos[mocap] = p
                    data.mocap_quat[mocap] = q

            # Pose playback: waypoint becomes the reference pose (was: unconditionally
            # overwriting mocap AFTER key_ctrl below, which silently discarded every
            # keypress -- froze the arm when both flags were on together, confirmed
            # live 2026-07-07). Key control now deltas on top of this base instead of
            # home_pos, so --playback-poses + --key-control compose: playback parks
            # the arm at the saved approach pose, keys fine-adjust/lower/lift from there.
            base_pos, base_quat = None, None
            if pose_pb is not None:
                base_pos, base_quat = pose_pb.current()
                if key_ctrl is None:
                    # hands-off playback: auto-advance by position + dwell.
                    pose_pb.step(base_pos)
                elif key_ctrl.pop_advance_request():
                    # manual mode: only advance when told to (kp5) -- grasping
                    # takes as long as it takes, position/dwell alone can't
                    # tell whether the grasp actually succeeded.
                    pose_pb.advance()
                    print(f"Advanced to waypoint {pose_pb._idx + 1}/{len(pose_pb._waypoints)}")
                elif key_ctrl.pop_retreat_request():
                    pose_pb.retreat()
                    print(f"Back to waypoint {pose_pb._idx + 1}/{len(pose_pb._waypoints)}")

            # Keyboard arm control: shift mocap position + rotate mocap orientation
            # by accumulated deltas (I/K pitch, J/L yaw, U/O roll about home/base axes).
            if key_ctrl is not None:
                ref_pos = base_pos if base_pos is not None else home_pos
                ref_quat = base_quat if base_quat is not None else home_quat
                data.mocap_pos[mocap] = ref_pos + key_ctrl.delta()
                new_quat = np.zeros(4)
                mujoco.mju_mulQuat(new_quat, key_ctrl.quat_delta(), ref_quat)
                data.mocap_quat[mocap] = new_quat
                if key_ctrl.pop_save_request():
                    save_key_pose(pose_file, data.mocap_pos[mocap].copy(),
                                  data.mocap_quat[mocap].copy(), key_ctrl.aperture())
                    print(f"Saved pose -> {pose_file} (aperture={key_ctrl.aperture():.2f})")
            elif pose_pb is not None:
                data.mocap_pos[mocap] = base_pos
                data.mocap_quat[mocap] = base_quat
            # else: neither flag active -- leave mocap as set by wrist_rx/sweep_wrist/default above.

            # Pose recorder: P saves current mocap pos+quat as waypoint.
            if pose_rec is not None:
                pose_rec.poll_save(data.mocap_pos[mocap].copy(),
                                   data.mocap_quat[mocap].copy())

            # Arm: OSC hold/track the wrist target. damping_ratio=1.0 (critical) instead
            # of the old 4 (heavily overdamped, tuned when the wrist target never moved
            # live) -- with kp=400, omega_n=20 rad/s, so critical damping settles in
            # ~150-200ms vs ~400ms at ratio=4 (see docs/estado_wrist_txty_dead_2026-07-04.md
            # for the derivation). Matched to the ~260ms perception (WiLoR/Colab) floor,
            # not tuned faster than that. Verify live for oscillation/overshoot.
            tau = opspace(
                model=model, data=data, site_id=site_id, dof_ids=panda_dof,
                pos=data.mocap_pos[mocap], ori=data.mocap_quat[mocap], joint=_PANDA_HOME,
                gravity_comp=True, pos_gains=(400.0, 400.0, 400.0), damping_ratio=1.0,
            )
            data.ctrl[panda_ctrl] = tau

            mujoco.mj_step(model, data)
            if state_frames is not None:
                state_frames.append(data.qpos.copy())
            if measure_grip or float_object:
                _grip_diagnostic(k)
            if slip_compensate:
                slip_bias = _slip_compensate_bias()
                active = np.nonzero(slip_bias > 1e-4)[0]
                if len(active) and k % 15 == 0:
                    names = [model.actuator(int(fing_ids[i])).name for i in active]
                    degs = [f"{np.degrees(slip_bias[i]):.1f}deg" for i in active]
                    print(f"[slip] bias active: {list(zip(names, degs))}")
            just_succeeded = task.update(model, data)
            succeeded = just_succeeded or succeeded
            if rec is not None and record:
                rec.maybe_capture(data, k)
            if shot_ctrl is not None and shot_ctrl.poll_capture():
                shot_count += 1
                saved = rec.shot_all(data, f"{shot}_{shot_count:03d}")
                print(f"[shot] saved {len(saved)} camera(s): {saved}")
            if viewer is not None:
                viewer.sync()
            if live:
                time.sleep(dt)
            if just_succeeded:
                print(f"succeeded at t={k * dt:.1f}s (of {n_steps * dt:.0f}s budget) -> ending trial early")
                break
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
        if shot_ctrl is not None:
            shot_ctrl.close()
        if grip_log_file is not None:
            grip_log_file.close()
            print(f"[grip] slip log saved to {grip_log_path}")
        if state_frames is not None:
            # Inside finally: Ctrl+C during a live take must not lose the trace --
            # that's the one thing --state-log exists to protect against (confirmed
            # live 2026-09-07: an interrupted run silently dropped it when this save
            # sat after the try/finally instead of in it).
            path = _nonclobber_path(Path(state_log))
            np.savez(path, qpos=np.array(state_frames), dt=dt, task=task_name)
            print(f"[state-log] {len(state_frames)} frames -> {path}")

    if rec is not None:
        if record:
            paths = rec.save_video(record)
            print(f"video saved: {paths}")
        rec.close()
    if shot:
        print(f"[shot] {shot_count} press(es) this run, files {shot}_NNN_<cam>.png")

    print(f"task={task_name} steps={n_steps}")
    print(f"succeed (DexJoCo metric): {succeeded}")
    return succeeded


_VALUE_FLAGS = ("--record", "--shot", "--cam", "--n-steps", "--duration",
                "--record-poses", "--playback-poses", "--grasp-class", "--pose-file",
                "--log-grip", "--state-log", "--replay-log")


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
        n_steps=int(flags["--n-steps"]) if "--n-steps" in flags else None,
        duration=float(flags.get("--duration", 60.0)),
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
        grasp_class=flags.get("--grasp-class", _DEFAULT_GRASP_CLASS),
        pose_file=flags.get("--pose-file", _DEFAULT_POSE_FILE),
        measure_grip="--measure-grip" in flags,
        log_grip=flags.get("--log-grip"),
        slip_compensate="--slip-compensate" in flags,
        float_object="--float-object" in flags,
        state_log=flags.get("--state-log"),
        replay_log=flags.get("--replay-log"),
    )
