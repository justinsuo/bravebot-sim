#!/usr/bin/env python3
"""
Export the BraveBot component library to machine- and human-readable manifests:

  * description/config/components.json   -- full structured registry
  * docs/BILL_OF_MATERIALS.md            -- readable component + sensor tables

Run:  python scripts/export_manifest.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from bravebot_sim import registry as R  # noqa: E402

JSON_PATH = os.path.join(ROOT, "description", "config", "components.json")
BOM_PATH = os.path.join(ROOT, "docs", "BILL_OF_MATERIALS.md")


def export_json():
    data = {
        "robot": "BraveBot",
        "base_platform": R.SPECS["base_platform"],
        "frame": "URDF-local: +x forward, +y left, +z up; origin at TRON 1 base link",
        "specs": R.SPECS,
        "experts": list(R.EXPERTS),
        "tron1_links": {
            k: {"pos_m": list(v), "mesh": f"tron1/{R.TRON1_MESH[k]}.stl"}
            for k, v in R.TRON1_LINK_POS.items()},
        "leg_joints": [
            {"name": j.name, "parent": j.parent, "child": j.child,
             "type": j.type, "origin_m": list(j.origin), "axis": list(j.axis),
             "range_rad": [j.lower, j.upper] if j.type == "revolute" else None}
            for j in R.LEG_JOINTS],
        "components": [
            {"id": c.id, "name": c.name, "group": c.group, "mount_m": list(c.pos),
             "mesh": f"bravebot/{c.id}.stl", "summary": c.summary,
             "sensor": None if not c.sensor else {
                 "modality": c.sensor.modality, "spec": c.sensor.spec,
                 "fov_deg": c.sensor.fov_deg, "range_m": c.sensor.range_m,
                 "detects": list(c.sensor.detects)}}
            for c in R.COMPONENTS],
    }
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print("wrote", os.path.relpath(JSON_PATH, ROOT))


def export_bom():
    lines = ["# BraveBot — Bill of Materials\n",
             "Auto-generated from `bravebot_sim/registry.py` "
             "(`python scripts/export_manifest.py`).\n",
             f"**Base platform:** {R.SPECS['base_platform']}  ",
             "**Frame:** URDF-local — +x forward, +y left, +z up, origin at the "
             "TRON 1 base link.\n",
             "## LimX TRON 1 base (real meshes, Apache-2.0)\n",
             "| Link | Mount (m) | Mesh |",
             "|------|-----------|------|"]
    for k, v in R.TRON1_LINK_POS.items():
        lines.append(f"| {k} | ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}) | "
                     f"`tron1/{R.TRON1_MESH[k]}.stl` |")

    lines += ["\n## BraveBot modification components (original geometry)\n",
              "| # | Component | Group | Mount (m) | Mesh | Notes |",
              "|---|-----------|-------|-----------|------|-------|"]
    for i, c in enumerate(R.COMPONENTS, 1):
        lines.append(f"| {i} | {c.name} | {c.group} | "
                     f"({c.pos[0]:.3f}, {c.pos[1]:.3f}, {c.pos[2]:.3f}) | "
                     f"`bravebot/{c.id}.stl` | {c.summary} |")

    lines += ["\n## Four-sensor inspection stack\n",
              "| Sensor | Modality | FOV (deg) | Range (m) | Spec |",
              "|--------|----------|-----------|-----------|------|"]
    for c in R.SENSOR_COMPONENTS:
        s = c.sensor
        lines.append(f"| {c.name} | {s.modality} | {s.fov_deg:.0f} | "
                     f"{s.range_m:.0f} | {s.spec} |")

    lines += ["\n## Headline specifications\n", "| Spec | Value |", "|------|-------|"]
    for k, v in R.SPECS.items():
        lines.append(f"| {k.replace('_', ' ')} | {v} |")

    os.makedirs(os.path.dirname(BOM_PATH), exist_ok=True)
    with open(BOM_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.relpath(BOM_PATH, ROOT))


if __name__ == "__main__":
    export_json()
    export_bom()
