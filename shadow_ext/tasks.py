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


class HammerNail(_Task):
    """Copy of panda_hammer_nail_env nail-drive physics + success.
    The nail is a mocap body driven DOWN on each valid hammer impact (this task
    WRITES data.mocap_pos[nail], unlike the read-only tasks). Success = nail
    depth >= 0.04m. Defaults copied from the env __init__.

    Impact model (env lines 555-651): track the hammer 'face' geom z-velocity in
    a 12-sample buffer; on a hammer<->nail contact, if the pre-impact downward
    velocity exceeds the threshold, advance depth by impact_step*speed_scale."""
    arena = "arena_arm_hand_hammer_nail.xml"
    name = "hammer_nail"
    _SUCCESS_DEPTH = 0.04
    _IMPACT_STEP = 0.008
    _VEL_THRESH = 0.02
    _MAX_DEPTH = 0.0726

    def reset(self):
        self._init = False
        self._nail_depth = 0.0
        self._prev_face_z = None
        self._vz_buf = []

    def _setup(self, model, data):
        nail_body = model.body("nail")
        self._nail_mocap_id = int(nail_body.mocapid[0])
        self._nail_init_pos = model.body_pos[nail_body.id].copy()
        self._nail_init_quat = model.body_quat[nail_body.id].copy()
        self._hammer_gids = {model.geom(n).id for n in ("face", "head", "neck", "claw")
                             if _has_geom(model, n)}
        self._nail_gids = {model.geom(n).id for n in ("nail_head", "nail_shaft")
                           if _has_geom(model, n)}
        self._face_gid = model.geom("face").id if _has_geom(model, "face") else -1
        self._dt = float(model.opt.timestep)
        self._init = True

    def update(self, model, data) -> bool:
        if not self._init:
            self._setup(model, data)

        # track hammer-face z velocity (12-sample buffer)
        if self._face_gid >= 0:
            face_z = float(data.geom_xpos[self._face_gid][2])
            if self._prev_face_z is None:
                self._prev_face_z = face_z
            self._vz_buf.append((face_z - self._prev_face_z) / self._dt)
            self._prev_face_z = face_z
            if len(self._vz_buf) > 12:
                self._vz_buf.pop(0)

        # contact scan: any hammer geom touching any nail geom
        hit = False
        for i in range(int(data.ncon)):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self._hammer_gids and g2 in self._nail_gids) or \
               (g2 in self._hammer_gids and g1 in self._nail_gids):
                hit = True
                break

        if hit:
            preimpact_vz = min(self._vz_buf) if self._vz_buf else 0.0
            if preimpact_vz < -self._VEL_THRESH:
                scale = min(3.0, abs(preimpact_vz) / max(self._VEL_THRESH, 1e-6))
                new_depth = min(self._MAX_DEPTH, self._nail_depth + self._IMPACT_STEP * scale)
                if new_depth > self._nail_depth:
                    self._nail_depth = new_depth
                    pos = self._nail_init_pos.copy()
                    pos[2] = self._nail_init_pos[2] - new_depth
                    data.mocap_pos[self._nail_mocap_id] = pos
                    data.mocap_quat[self._nail_mocap_id] = self._nail_init_quat

        return self._nail_depth >= self._SUCCESS_DEPTH


def _has_geom(model, name) -> bool:
    try:
        model.geom(name)
        return True
    except Exception:
        return False


REGISTRY: dict[str, _Task] = {
    "pick_bucket": PickBucket(),
    "water_plant": WaterPlant(),
    "pinch_tongs": PinchTongs(),
    "hammer_nail": HammerNail(),
}
