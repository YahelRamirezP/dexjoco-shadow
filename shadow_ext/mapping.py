"""Map retargeter qpos (24 Shadow joints, Menagerie order) -> ctrl (20 actuators).

The only non-1:1 part is the coupled distal tendons. In the Shadow MJCF each of
FF/MF/RF/LF has a `fixed` tendon FFJ0 = 1*FFJ2 + 1*FFJ1 driven by one position
actuator A_FFJ0. So the actuator command for a coupled tendon is the SUM of the
two distal joint angles -- this is fixed by the model (coef 1/1), not a choice.

Everything else (abduction, MCP, thumb, the 2 wrist joints which the retargeter
already zeros) maps one joint -> one actuator.

The mapping is built by NAME from the compiled model, so it survives index
reordering and the "rh-" attach prefix.
"""
from __future__ import annotations
import numpy as np
import mujoco

_ATT_PREFIX = "rh-"          # added by build_spec().attach_body(..., "rh-", "")
_ACT_PREFIX = "rh-rh_A_"     # compiled actuator names, e.g. "rh-rh_A_FFJ0"
_JNT_PREFIX = "rh-rh_"       # compiled joint names,    e.g. "rh-rh_FFJ2"


def build_finger_map(model: mujoco.MjModel, hand: str = "shadow"):
    """Return (ctrl_ids[20], plan) where plan[k] = list of joint qpos addresses
    to SUM for actuator ctrl_ids[k]. Single-joint actuators have a 1-element list.
    """
    if hand == "allegro":
        names = [f"{finger}{i}" for finger in ("ff", "mf", "rf", "th") for i in range(4)]
        ctrl = [model.actuator(n[:-1] + "a" + n[-1]).id for n in names]
        plan = [[int(model.jnt_qposadr[model.joint(n[:-1] + "j" + n[-1]).id])]
                for n in names]
        return np.asarray(ctrl, dtype=int), plan
    if hand != "shadow":
        raise ValueError(f"Unsupported hand: {hand}")
    ctrl_ids: list[int] = []
    plan: list[list[int]] = []
    for i in range(model.nu):
        name = model.actuator(i).name
        if not name.startswith(_ACT_PREFIX):
            continue
        suffix = name[len(_ACT_PREFIX):]          # e.g. "FFJ0", "THJ4", "WRJ1"
        finger, jnum = suffix[:-1], int(suffix[-1])  # ("FFJ", 0)
        if jnum == 0:                              # coupled distal tendon
            jnames = [f"{finger}2", f"{finger}1"]  # FFJ2 + FFJ1
        else:
            jnames = [f"{finger}{jnum}"]           # 1:1
        addrs = []
        for jn in jnames:
            jid = model.joint(f"{_JNT_PREFIX}{jn}").id
            addrs.append(int(model.jnt_qposadr[jid]))
        ctrl_ids.append(i)
        plan.append(addrs)
    return np.asarray(ctrl_ids, dtype=int), plan


def qpos_to_ctrl(q_full: np.ndarray, ctrl_ids, plan) -> np.ndarray:
    """q_full: model-sized qpos (or any array indexed by jnt_qposadr).
    Returns ctrl values for ctrl_ids (coupled actuators = sum of their joints).
    """
    return np.asarray([sum(q_full[a] for a in addrs) for addrs in plan],
                      dtype=np.float64)
