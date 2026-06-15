#!/usr/bin/env python3
"""
Export every robot part (9 TRON 1 links + 18 BraveBot components) with its exact
world pose, color, and metadata to web/parts.json — the data the interactive
exploded-view 3D interface (web/index.html) loads.

    python scripts/export_parts.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from bravebot_sim import registry as R  # noqa: E402

# Friendly names + groups for the 9 real LimX TRON 1 links (mesh-key -> info).
TRON1 = {
    "base":  ("TRON 1 base / torso", "chassis", "The LimX WF_TRON1A base — battery, mainboard, IMU and the wheel/leg actuator stack."),
    "abadL": ("Left ab/ad actuator", "leg", "Left hip ab/adduction joint — swings the leg laterally."),
    "abadR": ("Right ab/ad actuator", "leg", "Right hip ab/adduction joint — swings the leg laterally."),
    "hipL":  ("Left hip / thigh", "leg", "Left hip-pitch link (thigh) — drives the leg forward/back."),
    "hipR":  ("Right hip / thigh", "leg", "Right hip-pitch link (thigh) — drives the leg forward/back."),
    "kneeL": ("Left knee / shank", "leg", "Left knee link (shank) — bends to set stance height."),
    "kneeR": ("Right knee / shank", "leg", "Right knee link (shank) — bends to set stance height."),
    "wheelL":("Left drive wheel", "wheel", "Left hub-motor wheel — gripped tire, does the driving + balancing."),
    "wheelR":("Right drive wheel", "wheel", "Right hub-motor wheel — gripped tire, does the driving + balancing."),
}
GROUP_LABEL = {"chassis": "Chassis", "leg": "Legs", "wheel": "Wheels",
               "structural": "Structure", "compute": "Compute", "sensor": "Sensors",
               "power": "Power", "comms": "Comms", "safety": "Safety"}


def main():
    # Pose the robot in its ASSEMBLED standing stance (mirrored per-joint hip/knee
    # bend + correct base height) — NOT the zero-joint default, where the legs stick
    # out straight and float above the wheels. BraveBot.__init__ applies that stance.
    from bravebot_sim import BraveBot, model_path
    bot = BraveBot(model_path())
    m, d = bot.model, bot.data

    comp_by_id = {c.id: c for c in R.COMPONENTS}
    sensor_comp = {c.id: c.sensor for c in R.SENSOR_COMPONENTS}
    parts, centroid = [], np.zeros(3)

    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])
        if mesh_name is None or not mesh_name.startswith("m_"):
            continue
        key = mesh_name[2:]                                   # strip "m_"
        pos = d.geom_xpos[g].copy()
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, d.geom_xmat[g])   # [w,x,y,z]
        rgba = m.geom_rgba[g].copy()

        if key in TRON1:
            name, group, desc = TRON1[key]
            stl = f"../description/meshes/tron1/{R.TRON1_MESH[key]}.stl"
            mass = sensor = None
        elif key in comp_by_id:
            c = comp_by_id[key]
            name, group, desc = c.name, c.group, c.summary
            stl = f"../description/meshes/bravebot/{key}.stl"
            s = sensor_comp.get(key)
            sensor = (None if s is None else
                      {"modality": s.modality, "range_m": s.range_m, "fov_deg": s.fov_deg})
            mass = None
        else:
            continue

        parts.append(dict(
            id=key, name=name, group=group, group_label=GROUP_LABEL.get(group, group.title()),
            file=stl, pos=[round(float(x), 5) for x in pos],
            quat=[round(float(x), 6) for x in quat],
            color=[round(float(x), 3) for x in rgba[:3]],
            desc=desc, sensor=sensor))
        centroid += pos
    centroid /= max(1, len(parts))

    out = dict(
        robot="BraveBot — wheel-legged autonomous inspection robot (LimX WF_TRON1A base)",
        centroid=[round(float(x), 5) for x in centroid],
        n_parts=len(parts),
        groups=sorted({p["group_label"] for p in parts}),
        parts=sorted(parts, key=lambda p: (p["group"], p["id"])),
    )
    os.makedirs(os.path.join(ROOT, "web"), exist_ok=True)
    path = os.path.join(ROOT, "web", "parts.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}: {len(parts)} parts, groups {out['groups']}")


if __name__ == "__main__":
    main()
