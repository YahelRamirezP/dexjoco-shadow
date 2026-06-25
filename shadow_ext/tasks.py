"""Task registry for Shadow functional eval (Eval 2).

Each task knows (a) which DexJoCo arena to load and (b) how to score success.
The success logic is a FAITHFUL COPY of the corresponding DexJoCo env's
`_compute_success` -- we do not invent metrics, we replicate DexJoCo's so the
numbers are comparable to the native Allegro benchmark. Source env + line refs
are noted per task. Some tasks are stateful (counters / trigger latches), so a
task is an object with `.reset()` and `.update(model, data) -> bool succeeded`.

We only READ DexJoCo (its envs, scenes, success formulas). Nothing in DexJoCo is
modified. The Shadow swap happens via shadow_ext.build.build_spec(arena).
"""
from __future__ import annotations
import numpy as np


class _Task:
    arena: str
    name: str

    def reset(self) -> None:
        pass

    def update(self, model, data) -> bool:
        """Advance the success check one step; return current success bool."""
        raise NotImplementedError


class PickBucket(_Task):
    """Copy of panda_pick_bucket_env._compute_success (env lines 586-597).
    Success = boxed food inside the bucket footprint AND lifted >= 0.15m."""
    arena = "arena_arm_hand_bucket_pick.xml"
    name = "pick_bucket"

    def reset(self):
        self._bottom_z0 = None

    def update(self, model, data) -> bool:
        bottom_ids = [model.site(f"bucket_ref_{i}").id for i in (0, 2, 4, 6)]
        if self._bottom_z0 is None:
            self._bottom_z0 = data.site_xpos[bottom_ids, 2].copy()
        box_pos = np.array(data.sensor("boxed_food_0_pos").data, float)
        corner_ids = [model.site(f"bucket_ref_{i}").id for i in range(8)]
        corners = np.array(data.site_xpos[corner_ids], float)
        inside = np.all(box_pos >= corners.min(0)) and np.all(box_pos <= corners.max(0))
        bottom_z = data.site_xpos[bottom_ids, 2]
        lifted = np.all(bottom_z - self._bottom_z0 >= 0.15)
        return bool(inside and lifted)


class WaterPlant(_Task):
    """Copy of panda_water_plant_env._compute_success (env lines 557-575).
    Success = spray ref_point within a cylinder (R=0.2, HALF_H=0.2) of the plant
    AND the spray trigger pulled, held for 30 consecutive steps."""
    arena = "arena_arm_hand_plant.xml"
    name = "water_plant"
    _R = 0.2
    _HALF_H = 0.2
    _TRIGGER_RELEASE = 0.25
    _TRIGGER_PULL = 0.34
    _STEPS_REQUIRED = 30

    def reset(self):
        self._trigger_pulled = False
        self._counter = 0

    def update(self, model, data) -> bool:
        p = data.site_xpos[model.site("ref_point").id]
        plant = data.body("plant").xpos
        dx, dy, dz = p - plant
        inside = (dx * dx + dy * dy <= self._R * self._R) and (-self._HALF_H <= dz <= self._HALF_H)

        trig = float(data.sensor("spray_joint_0_pos").data)
        if trig < self._TRIGGER_RELEASE:
            self._trigger_pulled = False
        elif trig > self._TRIGGER_PULL:
            self._trigger_pulled = True

        if inside and self._trigger_pulled:
            self._counter += 1
        else:
            self._counter = 0
        return self._counter >= self._STEPS_REQUIRED


class PinchTongs(_Task):
    """Copy of panda_pinch_tongs_env (_compute_success + _update_pinch_count).
    Success = tongs lifted >= 0.1m above the table AND >= 3 pinches (open->close
    cycles of the tongs joint), held for 30 consecutive steps.

    Note: the env's lift threshold is table_z + 0.1; here we proxy table_z with
    the tongs' resting z captured on the first step (tongs start on the table)."""
    arena = "arena_arm_hand_table_tongs.xml"
    name = "pinch_tongs"
    _CLOSE_THRESH = -0.07
    _OPEN_THRESH = 0.1
    _REQUIRED_PINCHES = 3
    _LIFT_HEIGHT = 0.1
    _STEPS_REQUIRED = 30

    def reset(self):
        self._lift_z = None
        self._pinch_count = 0
        self._pinch_in_progress = False
        self._success_counter = 0

    def update(self, model, data) -> bool:
        tongs_pos = data.sensor("tongs_pos").data
        if self._lift_z is None:
            self._lift_z = float(tongs_pos[2]) + self._LIFT_HEIGHT

        # pinch counter: open->close edge increments
        jp = float(data.sensor("tongs_joint_0_pos").data)
        if jp <= self._CLOSE_THRESH and self._pinch_in_progress:
            self._pinch_in_progress = False
            self._pinch_count += 1
        elif jp >= self._OPEN_THRESH and not self._pinch_in_progress:
            self._pinch_in_progress = True

        lifted = tongs_pos[2] >= self._lift_z
        triggered = lifted and self._pinch_count >= self._REQUIRED_PINCHES
        if triggered:
            self._success_counter += 1
        else:
            self._success_counter = 0
        return self._success_counter >= self._STEPS_REQUIRED


# Registry. hammer_nail pending: its _nail_depth is env-internal physics
# (computed in step), not a pure read -> needs the nail-drive block copied or a
# task swap (e.g. click_mouse / fold_glasses).
REGISTRY: dict[str, _Task] = {
    "pick_bucket": PickBucket(),
    "water_plant": WaterPlant(),
    "pinch_tongs": PinchTongs(),
}
