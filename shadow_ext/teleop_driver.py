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
import sys
import numpy as np
import mujoco

from dexjoco.sim.controllers import opspace

from .build import build_spec
from .mapping import build_finger_map, qpos_to_ctrl
from .tasks import REGISTRY

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


def run(task_name: str = "pick_bucket", view: bool = False, n_steps: int = 1500):
    task = REGISTRY[task_name]
    model = build_spec(task.arena).compile()
    data = mujoco.MjData(model)

    panda_dof, panda_ctrl = _panda_ids(model)
    site_id = (model.site("attachment_site") or model.site("attachment_site_right")).id
    fing_ids, fing_plan = build_finger_map(model)
    fing_ctrlrange = model.actuator_ctrlrange[fing_ids].copy()

    # Home the arm, settle, then weld the wrist target to the current flange pose.
    data.qpos[panda_dof] = _PANDA_HOME
    mujoco.mj_forward(model, data)
    data.mocap_pos[0] = data.sensor("franka/flange_pos").data.copy()
    data.mocap_quat[0] = data.sensor("franka/flange_quat").data.copy()

    # Finger close target, mapped 24 joints -> 20 ctrl, clipped to actuator range.
    q_target = finger_target_qpos(model)
    ctrl_close = np.clip(qpos_to_ctrl(q_target, fing_ids, fing_plan),
                         fing_ctrlrange[:, 0], fing_ctrlrange[:, 1])

    task.reset()
    succeeded = False

    viewer = mujoco.viewer.launch_passive(model, data) if view else None
    try:
        for k in range(n_steps):
            # Fingers: ramp open->close over first ~1/3, then hold closed.
            alpha = min(1.0, k / (n_steps / 3.0))
            data.ctrl[fing_ids] = alpha * ctrl_close

            # Arm: OSC hold at the fixed wrist target (stage 1 = no wrist motion).
            tau = opspace(
                model=model, data=data, site_id=site_id, dof_ids=panda_dof,
                pos=data.mocap_pos[0], ori=data.mocap_quat[0], joint=_PANDA_HOME,
                gravity_comp=True, pos_gains=(400.0, 400.0, 400.0), damping_ratio=4,
            )
            data.ctrl[panda_ctrl] = tau

            mujoco.mj_step(model, data)
            succeeded = task.update(model, data) or succeeded
            if viewer is not None:
                viewer.sync()
    finally:
        if viewer is not None:
            viewer.close()

    print(f"task={task_name} steps={n_steps}")
    print(f"succeed (DexJoCo metric): {succeeded}")
    return succeeded


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    task_name = args[0] if args else "pick_bucket"
    if "--view" in sys.argv:
        import mujoco.viewer  # noqa: F401
    run(task_name=task_name, view="--view" in sys.argv)
