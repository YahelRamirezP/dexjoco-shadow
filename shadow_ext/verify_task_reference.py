"""Audit a recorded PickBucket session for a corrupted table-height reference.

Bug (fixed 2026-09-09, see tasks.py PickBucket.reset): before the fix, pressing
F5 (evaluation start) re-captured the bucket "resting height" baseline from
whatever the bucket's CURRENT height was at that instant. If F5 was pressed
after the bucket was already lifted, every lift measurement for the rest of
the trial was computed against an airborne baseline, making a real success
read as `lifted: false`.

This script replays the session's raw MuJoCo state trace (saved every step,
independent of the recorded final_task_metrics) and recomputes PickBucket
success using the TRUE baseline: the bucket height at the very first recorded
frame (sim_time ~0, before any operator input). It reports whether the
recorded reference was corrupted and, if so, what the corrected outcome is.

Usage: python -m shadow_ext.verify_task_reference <session_dir>
    <session_dir> is the trial directory, e.g. recordings/shadow-only-formal-008
    (containing sim/metadata.json), not the sim/ subdirectory itself.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .state_recording import TraceReader


def audit(session_dir):
    session_dir = Path(session_dir)
    reader = TraceReader(session_dir / "sim")
    model, data, meta = reader.model, reader.data, reader.metadata

    bottom_ids = [model.site(f"bucket_ref_{i}").id for i in (0, 2, 4, 6)]
    corner_ids = [model.site(f"bucket_ref_{i}").id for i in range(8)]

    reader.restore(0)
    true_z0 = data.site_xpos[bottom_ids, 2].copy()
    recorded_z0 = np.array(meta.get("final_task_metrics", {}).get("bottom_reference_z_m") or [])

    def check(i):
        reader.restore(i)
        box_pos = np.array(data.sensor("boxed_food_0_pos").data, float)
        corners = np.array(data.site_xpos[corner_ids], float)
        inside = bool(np.all(box_pos >= corners.min(0)) and np.all(box_pos <= corners.max(0)))
        bottom_z = data.site_xpos[bottom_ids, 2]
        lift = bottom_z - true_z0
        return inside, bool(np.all(lift >= 0.15)), float(np.min(lift)), float(data.time)

    evaluation = meta.get("evaluation") or {}
    eval_start_ns = evaluation.get("start_monotonic_ns")
    corrupted = bool(len(recorded_z0)) and float(np.max(np.abs(recorded_z0 - true_z0))) > 0.01

    result = {
        "session": session_dir.name,
        "true_bottom_reference_z_m": true_z0.tolist(),
        "recorded_bottom_reference_z_m": recorded_z0.tolist() if len(recorded_z0) else None,
        "reference_corrupted": corrupted,
    }

    if eval_start_ns is not None:
        idx_eval_start = reader.index_at(eval_start_ns)
        first_in_window = None
        for i in range(idx_eval_start, len(reader.timestamps)):
            inside, lifted, min_lift, t = check(i)
            if inside and lifted:
                first_in_window = t
                break
        result["corrected_success_within_eval_window"] = first_in_window is not None
        result["corrected_first_success_sim_time"] = first_in_window

    reader.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir")
    args = parser.parse_args()
    print(json.dumps(audit(args.session_dir), indent=2))


if __name__ == "__main__":
    main()
