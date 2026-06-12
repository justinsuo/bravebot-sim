"""
Facility scene generation — a small AI data-center aisle for the BraveBot to
patrol, plus the anomalies it is meant to catch (drawn from the product site's
threat scenarios).

`build_scene_xml()` writes a MuJoCo scene that <include>s the robot model and
adds server-rack rows, an aisle, a charging dock and anomaly markers. The
anomaly list is returned separately so the sim can score sensor coverage.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

from .sim import Anomaly

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
MJCF_DIR = os.path.join(ROOT, "description", "mjcf")
SCENE_PATH = os.path.join(MJCF_DIR, "bravebot_scene.xml")
SCENE_PHYS_PATH = os.path.join(MJCF_DIR, "bravebot_scene_physics.xml")

RACK = (0.6, 1.0, 2.0)        # depth, width, height (m)
AISLE_HALF = 1.4              # half-width of the patrol aisle
N_RACKS = 6                   # racks per row


# Anomalies positioned against the rack rows (from the site threat scenarios).
ANOMALIES: list[Anomaly] = [
    Anomaly("Rack 4B coolant micro-leak", "acoustic", (3.6, 1.0, 0.9),
            "ultrasonic jet at a liquid-cooling connector"),
    Anomaly("PDU 7 joint hotspot", "thermal", (5.4, -1.0, 1.2),
            "PDU termination trending above baseline"),
    Anomaly("UPS String C off-gas", "gas", (1.8, -1.0, 0.6),
            "VRLA off-gas precursor near the UPS cabinet"),
    Anomaly("Cold Aisle 2 containment door open", "visual", (2.7, 1.0, 1.1),
            "containment door left open"),
    Anomaly("Switchgear partial discharge", "acoustic", (6.6, 1.0, 1.4),
            "intermittent ultrasonic discharge at switchgear"),
]

# Patrol route waypoints (x, y) down the aisle and back to the dock.
WAYPOINTS = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.5, 0.0),
             (6.5, 0.0), (4.0, 0.0), (2.0, 0.0), (0.0, 0.0)]

DOCK = (-1.0, 0.0)


def _rgba(*v):
    return " ".join(f"{x:g}" for x in v)


def build_scene_xml(path: str = SCENE_PATH, robot: str = "bravebot.xml") -> str:
    mj = ET.Element("mujoco", model="bravebot_scene")
    ET.SubElement(mj, "include", file=robot)

    world = ET.SubElement(mj, "worldbody")

    # two rows of server racks flanking the aisle
    for row, sgn in (("L", 1.0), ("R", -1.0)):
        for i in range(N_RACKS):
            x = 1.2 + i * (RACK[0] + 0.25)
            y = sgn * (AISLE_HALF + RACK[1] / 2)
            b = ET.SubElement(world, "body", name=f"rack_{row}{i}", pos=f"{x:g} {y:g} {RACK[2]/2:g}")
            ET.SubElement(b, "geom", type="box",
                          size=f"{RACK[0]/2:g} {RACK[1]/2:g} {RACK[2]/2:g}",
                          rgba=_rgba(0.10, 0.11, 0.14, 1), contype="1", conaffinity="1")
            # blue status strip on the aisle-facing side
            ET.SubElement(b, "geom", type="box", pos=f"{RACK[0]/2:g} 0 0",
                          size=f"0.01 {RACK[1]/2*0.8:g} {RACK[2]/2*0.85:g}",
                          rgba=_rgba(0.20, 0.45, 0.95, 1), contype="0", conaffinity="0")

    # charging dock
    d = ET.SubElement(world, "body", name="dock", pos=f"{DOCK[0]:g} {DOCK[1]:g} 0.15")
    ET.SubElement(d, "geom", type="box", size="0.25 0.5 0.15",
                  rgba=_rgba(0.18, 0.20, 0.24, 1), contype="0", conaffinity="0")
    ET.SubElement(d, "geom", type="box", pos="0 0 0.18", size="0.22 0.45 0.02",
                  rgba=_rgba(0.20, 0.85, 0.45, 1), contype="0", conaffinity="0")

    # anomaly markers (emissive spheres color-coded by first-detecting sensor)
    colors = {"acoustic": (0.36, 0.85, 1.0), "thermal": (1.0, 0.45, 0.2),
              "gas": (0.5, 0.95, 0.5), "visual": (1.0, 0.85, 0.35)}
    for k, a in enumerate(ANOMALIES):
        c = colors[a.modality]
        b = ET.SubElement(world, "body", name=f"anom_{k}", pos=_rgba(*a.pos))
        ET.SubElement(b, "geom", type="sphere", size="0.09",
                      rgba=_rgba(*c, 0.9), contype="0", conaffinity="0")

    raw = ET.tostring(mj, "utf-8")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(pretty)
    return path


if __name__ == "__main__":
    print("wrote", build_scene_xml())
