"""Measure real attempt duration independent of the F5 evaluation marker.

F5 (evaluation start) is operator-controlled and includes a variable amount of
real-world setup time before it (or, when mistimed, none at all -- see
verify_task_reference.py). Neither the wall time from session start (dominated
by camera/positioning overhead) nor the wall time from F5 is a reliable
"how long did the grasp attempt take" measurement across every trial.

This script derives that duration directly from the raw state trace: it finds
the first frame where the hand's grasp site has moved >= `--threshold` meters
from its resting pose (the operator visibly starting to reach/act, not sensor
or controller-settling jitter -- 1cm is stable; below ~0.5cm this fires on
startup settling within the first 0.1s) and the first frame where the task
succeeds using the true (frame-0) physical reference (see
verify_task_reference.py for why that matters). The interval between the two
is comparable across every trial regardless of when, or whether meaningfully,
F5 was pressed.

Usage: python -m shadow_ext.attempt_timing <session_dir> [--threshold 0.01]
    <session_dir> is the trial directory (containing sim/metadata.json).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .state_recording import TraceReader


def measure(session_dir, threshold_m=0.01):
    session_dir = Path(session_dir)
    reader = TraceReader(session_dir / "sim")
    model, data = reader.model, reader.data

    grasp_site = model.site("rh-grasp_site").id
    bottom_ids = [model.site(f"bucket_ref_{i}").id for i in (0, 2, 4, 6)]
    corner_ids = [model.site(f"bucket_ref_{i}").id for i in range(8)]

    reader.restore(0)
    hand_p0 = data.site_xpos[grasp_site].copy()
    bottom_z0 = data.site_xpos[bottom_ids, 2].copy()

    motion_i = success_i = None
    for i in range(len(reader.timestamps)):
        reader.restore(i)
        if motion_i is None and np.linalg.norm(data.site_xpos[grasp_site] - hand_p0) >= threshold_m:
            motion_i = i
        if success_i is None:
            box_pos = np.array(data.sensor("boxed_food_0_pos").data, float)
            corners = np.array(data.site_xpos[corner_ids], float)
            inside = np.all(box_pos >= corners.min(0)) and np.all(box_pos <= corners.max(0))
            bottom_z = data.site_xpos[bottom_ids, 2]
            lifted = np.all(bottom_z - bottom_z0 >= 0.15)
            if inside and lifted:
                success_i = i
        if motion_i is not None and success_i is not None:
            break

    result = {"session": session_dir.name, "threshold_m": threshold_m,
              "first_motion_sim_time": None, "first_success_sim_time": None,
              "time_to_success_from_motion_wall_s": None}
    if motion_i is not None:
        t_motion_ns = int(reader.timestamps[motion_i])
        reader.restore(motion_i)
        result["first_motion_sim_time"] = float(data.time)
    if success_i is not None:
        t_success_ns = int(reader.timestamps[success_i])
        reader.restore(success_i)
        result["first_success_sim_time"] = float(data.time)
    if motion_i is not None and success_i is not None:
        result["time_to_success_from_motion_wall_s"] = (t_success_ns - t_motion_ns) / 1e9

    reader.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir")
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()
    print(json.dumps(measure(args.session_dir, args.threshold), indent=2))


if __name__ == "__main__":
    main()
