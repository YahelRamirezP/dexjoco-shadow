"""Stage-1 de-risk: does ctrl -> Shadow fingers close cleanly in the attached model?

Isolated test, no env class, no retargeter, no teleop. Loads the Panda+Shadow
build_spec() model, disables gravity (so the un-actuated Franka arm doesn't fall
and distract), then ramps the Shadow position actuators from an open pose to a
hand-tuned close pose and animates it in the viewer.

Goal: eyeball whether the 20 Shadow position actuators (incl. the 4 coupled
distal tendons rh_A_*J0, range [0, pi]) produce a natural fist, or whether
fingers twist/penetrate. This validates ctrl->fingers before any 24->20 mapping
or env refactor.

Usage:
    python -m shadow_ext.finger_close_test            # ramp + viewer
    python -m shadow_ext.finger_close_test --print    # just dump actuator table
"""
from __future__ import annotations
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

from .build import build_spec

# Hand-tuned close targets, keyed by the Shadow actuator SUFFIX (after the
# "rh-rh_A_" attach prefix). Anything not listed -> 0.0 (neutral/open).
# Per finger: J4=abduction, J3=MCP flex, J0=coupled PIP+DIP tendon ([0,pi]).
# Thumb: THJ5 opposition rotate, THJ4 flex toward palm, THJ2/THJ1 curl.
_CLOSE = {
    "FFJ3": 1.40, "FFJ0": 2.60,
    "MFJ3": 1.40, "MFJ0": 2.60,
    "RFJ3": 1.40, "RFJ0": 2.60,
    "LFJ3": 1.40, "LFJ0": 2.60,
    "THJ5": 0.90, "THJ4": 1.10, "THJ2": 0.50, "THJ1": 1.20,
}

_PREFIX = "rh-rh_A_"


def _hand_actuators(m: mujoco.MjModel):
    """Return [(actuator_id, suffix, ctrlrange)] for the Shadow hand only."""
    out = []
    for i in range(m.nu):
        name = m.actuator(i).name
        if name.startswith(_PREFIX):
            out.append((i, name[len(_PREFIX):], m.actuator_ctrlrange[i].copy()))
    return out


def _clip(val, lo, hi):
    return float(np.clip(val, lo, hi))


def main():
    just_print = "--print" in sys.argv

    spec = build_spec()
    m = spec.compile()
    d = mujoco.MjData(m)

    hand = _hand_actuators(m)
    njoint_hand = sum(1 for j in range(m.njnt) if m.joint(j).name.startswith("rh-"))
    print(f"compiled: nq={m.nq} nu={m.nu} nbody={m.nbody}")
    print(f"shadow: {len(hand)} actuators, {njoint_hand} joints "
          f"(coupling => {njoint_hand}->{len(hand)})")
    print("--- shadow actuators ---")
    for aid, suf, cr in hand:
        tgt = _CLOSE.get(suf, 0.0)
        clipped = _clip(tgt, cr[0], cr[1])
        flag = "  <-- target clipped!" if clipped != tgt else ""
        print(f"id={aid:2d} {suf:6s} range=[{cr[0]:+.3f},{cr[1]:+.3f}] "
              f"close={clipped:+.3f}{flag}")

    if just_print:
        return

    # Build open (current = 0) and close ctrl vectors over the hand actuators.
    open_ctrl = {aid: 0.0 for aid, _, _ in hand}
    close_ctrl = {aid: _clip(_CLOSE.get(suf, 0.0), cr[0], cr[1])
                  for aid, suf, cr in hand}

    # Isolate fingers: kill gravity so the OSC-less Franka arm stays put.
    m.opt.gravity[:] = 0.0

    print("\nlaunching viewer: ramp open->close over ~3s, hold, loop. Ctrl-C to quit.")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        t0 = time.time()
        period = 6.0  # full open->close->open cycle seconds
        while viewer.is_running():
            phase = ((time.time() - t0) % period) / period
            # triangle wave 0->1->0
            alpha = 1.0 - abs(2.0 * phase - 1.0)
            for aid in open_ctrl:
                d.ctrl[aid] = (1 - alpha) * open_ctrl[aid] + alpha * close_ctrl[aid]
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(m.opt.timestep)


if __name__ == "__main__":
    main()
