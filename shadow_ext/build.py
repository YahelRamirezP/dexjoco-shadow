"""Build Panda+Shadow models from DexJoCo Allegro scenes (in-memory MjSpec).

Swaps the Allegro hand for a Shadow Hand at the panda wrist attachment site.
Paths are resolved relative to this file so the fork is self-contained
(Shadow Hand vendored under sim/envs/xmls/shadow_hand/, from MuJoCo Menagerie).

Usage:
    python -m shadow_ext.build [arena_xml_name]   # launches viewer
"""
from __future__ import annotations
import os, sys
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
XMLS = os.path.join(_HERE, "..", "dexjoco", "dexjoco", "sim", "envs", "xmls")
SHADOW_DIR = os.path.join(XMLS, "shadow_hand")
SHADOW = os.path.join(SHADOW_DIR, "right_hand.xml")

_HAND_TAGS = ("ffa", "mfa", "rfa", "tha", "ffj", "mfj", "rfj", "thj", "allegro")
_ALLEGRO_BODIES = ("allegro_palm", "allegro_attachment",
                   "allegro_palm_right", "allegro_attachment_right")


def build_spec(arena_name: str = "arena_arm_hand_bucket_pick.xml") -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(os.path.join(XMLS, arena_name))

    # remove allegro: excludes -> actuators -> sensors -> palm subtree
    for ex in list(getattr(spec, "excludes", [])):
        s = ((getattr(ex, "bodyname1", "") or "") + (getattr(ex, "bodyname2", "") or "")).lower()
        if any(t in s for t in ("allegro", "ff_base", "mf_base", "rf_base", "th_base")):
            spec.delete(ex)
    for it in list(spec.actuators) + list(spec.sensors):
        if any(t in (it.name or "").lower() for t in _HAND_TAGS):
            spec.delete(it)
    for bn in _ALLEGRO_BODIES:
        b = spec.body(bn)
        if b is not None:
            spec.delete(b)

    # attach shadow at the panda wrist site (name differs by source xml)
    site = spec.site("attachment_site") or spec.site("attachment_site_right")
    shadow = mujoco.MjSpec.from_file(SHADOW)
    site.attach_body(shadow.body("rh_forearm"), "rh-", "")

    # resolve mesh/texture paths to absolute (runtime only; not serialized)
    roots = [XMLS, os.path.join(SHADOW_DIR, "assets"), SHADOW_DIR]
    for m in spec.meshes:
        if os.path.isabs(m.file) and os.path.exists(m.file):
            continue
        for r in roots:
            c = os.path.join(r, m.file)
            if os.path.exists(c):
                m.file = c
                break
    for t in spec.textures:
        if t.file and not os.path.isabs(t.file):
            c = os.path.join(XMLS, t.file)
            if os.path.exists(c):
                t.file = c
    spec.meshdir = ""
    spec.texturedir = ""
    return spec


if __name__ == "__main__":
    arena = sys.argv[1] if len(sys.argv) > 1 else "arena_arm_hand_bucket_pick.xml"
    spec = build_spec(arena)
    model = spec.compile()
    print(f"OK {arena}: nq={model.nq} nu={model.nu} nbody={model.nbody}")
    import mujoco.viewer
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)
