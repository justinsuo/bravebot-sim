#!/usr/bin/env python3
"""
Generate the BraveBot robot description from the component registry:

  * description/mjcf/bravebot.xml   -- MuJoCo model for the interactive sim
  * description/urdf/bravebot.urdf  -- URDF for ROS 2 / Gazebo / RViz

Both reference the same meshes:
  * real LimX TRON 1 links  -> description/meshes/tron1/*.stl   (Apache-2.0)
  * BraveBot modifications  -> description/meshes/bravebot/*.stl (generated)

Run:  python scripts/build_model.py   (regenerates meshes first)
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from bravebot_sim import registry as R  # noqa: E402
from bravebot_sim import meshgen          # noqa: E402

MESH_DIR = os.path.join(ROOT, "description", "meshes")
MJCF_PATH = os.path.join(ROOT, "description", "mjcf", "bravebot.xml")
MJCF_PHYS_PATH = os.path.join(ROOT, "description", "mjcf", "bravebot_physics.xml")
URDF_PATH = os.path.join(ROOT, "description", "urdf", "bravebot.urdf")

# Real link densities are unknown; approximate masses (kg) for physics mode.
LINK_MASS = {"base": 9.0, "abad": 1.4, "hip": 1.6, "knee": 1.2, "wheel": 1.1}
COMP_MASS = {  # BraveBot modification masses (kg) — sum ≈ 15 kg payload
    "payload": 2.2, "torso": 4.0, "battery": 2.6, "edgeai": 0.9, "display": 0.5,
    "rear": 0.5, "estop": 0.1, "mast": 0.7, "head": 0.6, "acoustic": 0.4,
    "thermal": 0.3, "hdcam": 0.3, "gas": 0.3, "nav": 0.4, "antennaL": 0.05,
    "antennaR": 0.05, "railL": 0.3, "railR": 0.3,
}

GROUP_RGBA = {
    "structural": "0.86 0.88 0.92 1", "power": "0.95 0.55 0.18 1",
    "compute": "0.34 0.38 0.46 1", "sensor": "0.30 0.75 0.95 1",
    "safety": "0.90 0.25 0.20 1", "comms": "0.55 0.58 0.64 1",
}
SENSOR_RGBA = {"acoustic": "0.36 0.85 1 1", "thermal": "1 0.55 0.30 1",
               "gas": "0.55 0.95 0.55 1", "visual": "1 0.85 0.35 1",
               "nav": "0.70 0.55 1 1"}


def _v(t) -> str:
    return f"{t[0]:.6g} {t[1]:.6g} {t[2]:.6g}"


def _mesh_bbox(rel_path: str):
    m = trimesh.load(os.path.join(MESH_DIR, rel_path), force="mesh")
    return m.extents, m.bounds


def _mesh_props(rel_path: str):
    """Return (extents, centroid) of a mesh — used for physics inertials."""
    m = trimesh.load(os.path.join(MESH_DIR, rel_path), force="mesh")
    return np.asarray(m.extents), np.asarray((m.bounds[0] + m.bounds[1]) / 2.0)


def _add_inertial(body, mass: float, rel_mesh: str, min_dim: float = 0.05):
    """Explicit solid-box inertial from a known mass + the mesh bounding box.

    Robust for the non-watertight concatenated component meshes (their volume is
    unreliable, so we never let MuJoCo infer mass from geom density in physics
    mode). COM sits at the mesh centroid; inertia is the solid-box tensor.
    """
    ext, cen = _mesh_props(rel_mesh)
    x, y, z = (max(float(e), min_dim) for e in ext)
    ixx = mass / 12.0 * (y * y + z * z)
    iyy = mass / 12.0 * (x * x + z * z)
    izz = mass / 12.0 * (x * x + y * y)
    ET.SubElement(body, "inertial", pos=_v(cen), mass=f"{mass:.4f}",
                  diaginertia=f"{ixx:.5f} {iyy:.5f} {izz:.5f}")


# --------------------------------------------------------------------------- #
#  Base standing height                                                        #
# --------------------------------------------------------------------------- #
# At the home leg pose the wheel centre sits at z = wheelL.z relative to base.
# Drop the base so the wheels just touch the z=0 floor.
BASE_Z = -R.TRON1_LINK_POS["wheelL"][2] + R.WHEEL_RADIUS


# --------------------------------------------------------------------------- #
#  MJCF                                                                         #
# --------------------------------------------------------------------------- #

# Leg stance held by stiff position servos in physics mode (hip, knee, abad).
# kv adds actuator damping so the legs don't sag (~9° sag without it), keeping
# the CoM height L constant so the balance plant stays well-tuned.
PHYS_STANCE = {"hip": 0.30, "knee": -0.60, "abad": 0.0}
LEG_SERVO = {"abad": dict(kp=500, kv=25, damping=6, fr=120),
             "hip":  dict(kp=900, kv=45, damping=10, fr=160),
             "knee": dict(kp=900, kv=45, damping=10, fr=160)}
WHEEL_PEAK_TORQUE = 60.0   # N·m, TRON 1 peak


def build_mjcf(physics: bool = False) -> str:
    out_path = MJCF_PHYS_PATH if physics else MJCF_PATH
    mj = ET.Element("mujoco", model="bravebot_physics" if physics else "bravebot")
    ET.SubElement(mj, "compiler", meshdir=os.path.relpath(MESH_DIR, os.path.dirname(out_path)),
                  angle="radian", autolimits="true")
    if physics:
        ET.SubElement(mj, "option", timestep="0.002", gravity="0 0 -9.81",
                      integrator="implicitfast", cone="elliptic", impratio="3")
    else:
        ET.SubElement(mj, "option", timestep="0.004", gravity="0 0 -9.81",
                      integrator="implicitfast")

    visual = ET.SubElement(mj, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080", azimuth="130", elevation="-18")
    ET.SubElement(visual, "quality", shadowsize="4096", numslices="28")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.4", diffuse="0.5 0.5 0.55",
                  specular="0.1 0.1 0.1")
    ET.SubElement(visual, "map", znear="0.01", zfar="50")

    default = ET.SubElement(mj, "default")
    ET.SubElement(default, "geom", contype="0", conaffinity="0", group="2",
                  density="500", friction="1 0.05 0.01")
    ET.SubElement(default, "site", size="0.012", rgba="1 1 0 1", group="3")

    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1="0.16 0.18 0.22", rgb2="0.04 0.05 0.07", width="512", height="512")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                  rgb1="0.13 0.14 0.17", rgb2="0.17 0.19 0.23", width="512", height="512")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="40 40",
                  reflectance="0.05")
    # mesh assets
    for link, mesh in R.TRON1_MESH.items():
        ET.SubElement(asset, "mesh", name=f"m_{link}", file=f"tron1/{mesh}.stl")
    for comp in R.COMPONENTS:
        ET.SubElement(asset, "mesh", name=f"m_{comp.id}", file=f"bravebot/{comp.id}.stl")

    world = ET.SubElement(mj, "worldbody")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", directional="true",
                  diffuse="0.7 0.7 0.7")
    ET.SubElement(world, "light", pos="3 3 3", dir="-1 -1 -1", diffuse="0.4 0.4 0.45")
    ET.SubElement(world, "geom", name="floor", type="plane", size="40 40 0.1",
                  material="grid", contype="1", conaffinity="1")

    base = ET.SubElement(world, "body", name="base_link", pos=_v((0, 0, BASE_Z)))
    ET.SubElement(base, "freejoint", name="root")
    ET.SubElement(base, "geom", type="mesh", mesh="m_base", rgba="0.82 0.84 0.88 1")
    ET.SubElement(base, "site", name="imu", pos="0 0 0", rgba="1 1 1 0.3")
    if physics:
        _add_inertial(base, LINK_MASS["base"], "tron1/base_Link.stl")
    else:
        ET.SubElement(base, "inertial", pos="0 0 -0.1", mass=str(LINK_MASS["base"]),
                      diaginertia="0.25 0.25 0.18")

    # BraveBot modification bodies (fixed to the base)
    for comp in R.COMPONENTS:
        body = ET.SubElement(base, "body", name=f"{comp.id}_link", pos=_v(comp.pos))
        rgba = SENSOR_RGBA[comp.sensor.modality] if comp.sensor else GROUP_RGBA[comp.group]
        ET.SubElement(body, "geom", type="mesh", mesh=f"m_{comp.id}", rgba=rgba)
        if physics:
            _add_inertial(body, COMP_MASS[comp.id], f"bravebot/{comp.id}.stl")
        if comp.sensor:
            # site frame: +x of the body is the sensor look axis (matches geometry)
            ET.SubElement(body, "site", name=f"s_{comp.id}", pos="0 0 0",
                          rgba=SENSOR_RGBA[comp.sensor.modality])

    # Leg chains
    bodies = {"base": base}
    for j in R.LEG_JOINTS:
        parent_body = bodies[_canonical(j.parent)]
        b = ET.SubElement(parent_body, "body", name=f"{j.child}_link", pos=_v(j.origin))
        link_key = _link_kind(j.child)
        attrs = {"name": j.name, "type": "hinge", "axis": _v(j.axis), "pos": "0 0 0"}
        if j.type == "revolute":
            attrs["range"] = f"{j.lower} {j.upper}"
            if physics:
                attrs["damping"] = str(LEG_SERVO[link_key]["damping"])
        else:
            attrs["limited"] = "false"
            if physics:
                attrs["damping"] = "0.05"          # tiny wheel bearing friction
        ET.SubElement(b, "joint", **attrs)
        is_wheel = "wheel" in j.child
        rgba = "0.30 0.32 0.36 1" if is_wheel else "0.62 0.66 0.72 1"
        gattrs = dict(type="mesh", mesh=f"m_{_canonical_mesh(j.child)}", rgba=rgba)
        if is_wheel:
            gattrs.update(contype="1", conaffinity="1", condim="4",
                          friction="1.4 0.02 0.001", solref="0.008 1", solimp="0.95 0.99 0.001",
                          priority="2")
        ET.SubElement(b, "geom", **gattrs)
        if physics:
            _add_inertial(b, LINK_MASS[link_key],
                          f"tron1/{R.TRON1_MESH[j.child]}.stl", min_dim=0.03)
        else:
            ET.SubElement(b, "inertial", pos="0 0 0", mass=str(LINK_MASS[link_key]),
                          diaginertia="0.01 0.01 0.01")
        bodies[j.child] = b

    # Sensors (physics mode) — clean state read for the balance controller
    if physics:
        sens = ET.SubElement(mj, "sensor")
        ET.SubElement(sens, "framequat", name="base_quat", objtype="site", objname="imu")
        ET.SubElement(sens, "framezaxis", name="base_zaxis", objtype="site", objname="imu")
        ET.SubElement(sens, "gyro", name="base_gyro", site="imu")
        ET.SubElement(sens, "velocimeter", name="base_vel", site="imu")
        for side in ("L", "R"):
            ET.SubElement(sens, "jointvel", name=f"wheel_{side}_w", joint=f"wheel_{side}")
            ET.SubElement(sens, "jointpos", name=f"wheel_{side}_q", joint=f"wheel_{side}")

    # Actuators
    act = ET.SubElement(mj, "actuator")
    if physics:
        # legs: stiff position servos hold the stance; wheels: torque motors
        for j in R.LEG_JOINTS:
            if j.type != "revolute":
                continue
            s = LEG_SERVO[_link_kind(j.child)]
            ET.SubElement(act, "position", name=f"{j.name}_pos", joint=j.name,
                          kp=str(s["kp"]), kv=str(s["kv"]),
                          forcerange=f"-{s['fr']} {s['fr']}")
        for side in ("L", "R"):
            ET.SubElement(act, "motor", name=f"wheel_{side}_mot", joint=f"wheel_{side}",
                          gear="1", ctrlrange=f"-{WHEEL_PEAK_TORQUE} {WHEEL_PEAK_TORQUE}",
                          forcerange=f"-{WHEEL_PEAK_TORQUE} {WHEEL_PEAK_TORQUE}")
    else:
        for side in ("L", "R"):
            ET.SubElement(act, "velocity", name=f"wheel_{side}_vel", joint=f"wheel_{side}",
                          kv="8", ctrlrange="-40 40")

    return _pretty(mj)


def _canonical(link: str) -> str:
    return "base" if link == "base" else link


def _canonical_mesh(child: str) -> str:
    # child like abadL/hipR/wheelL -> mesh key abadL etc. (registry keys match)
    return child


def _link_kind(child: str) -> str:
    for k in ("abad", "hip", "knee", "wheel"):
        if child.startswith(k):
            return k
    return "base"


# --------------------------------------------------------------------------- #
#  URDF                                                                         #
# --------------------------------------------------------------------------- #

def _inertial(parent, mass, ext):
    x, y, z = (float(e) for e in ext)
    ixx = mass / 12.0 * (y * y + z * z)
    iyy = mass / 12.0 * (x * x + z * z)
    izz = mass / 12.0 * (x * x + y * y)
    inertial = ET.SubElement(parent, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.4f}")
    ET.SubElement(inertial, "inertia", ixx=f"{ixx:.5f}", iyy=f"{iyy:.5f}", izz=f"{izz:.5f}",
                  ixy="0", ixz="0", iyz="0")


def _link(robot, name, mesh_rel, mass, rgba="0.8 0.82 0.86 1"):
    ext, _ = _mesh_bbox(mesh_rel)
    link = ET.SubElement(robot, "link", name=name)
    for tag in ("visual", "collision"):
        node = ET.SubElement(link, tag)
        ET.SubElement(node, "origin", xyz="0 0 0", rpy="0 0 0")
        geo = ET.SubElement(node, "geometry")
        ET.SubElement(geo, "mesh", filename=f"../meshes/{mesh_rel}")
        if tag == "visual":
            mat = ET.SubElement(node, "material", name=f"{name}_mat")
            ET.SubElement(mat, "color", rgba=rgba)
    _inertial(link, mass, ext)
    return link


def build_urdf() -> str:
    robot = ET.Element("robot", name="bravebot")

    _link(robot, "base_link", "tron1/base_Link.stl", LINK_MASS["base"])

    # leg chain
    for j in R.LEG_JOINTS:
        child_link = f"{j.child}_link"
        mesh = f"tron1/{R.TRON1_MESH[j.child]}.stl"
        rgba = "0.25 0.27 0.30 1" if "wheel" in j.child else "0.6 0.64 0.7 1"
        _link(robot, child_link, mesh, LINK_MASS[_link_kind(j.child)], rgba)
        joint = ET.SubElement(robot, "joint", name=j.name,
                              type="continuous" if j.type == "continuous" else "revolute")
        ET.SubElement(joint, "parent", link=f"{_canonical(j.parent)}_link"
                      if j.parent != "base" else "base_link")
        ET.SubElement(joint, "child", link=child_link)
        ET.SubElement(joint, "origin", xyz=_v(j.origin), rpy="0 0 0")
        ET.SubElement(joint, "axis", xyz=_v(j.axis))
        if j.type == "revolute":
            ET.SubElement(joint, "limit", lower=str(j.lower), upper=str(j.upper),
                          effort="60", velocity="15")

    # BraveBot modification links (fixed joints to base)
    for comp in R.COMPONENTS:
        rgba = SENSOR_RGBA[comp.sensor.modality] if comp.sensor else GROUP_RGBA[comp.group]
        _link(robot, f"{comp.id}_link", f"bravebot/{comp.id}.stl",
              COMP_MASS[comp.id], rgba)
        joint = ET.SubElement(robot, "joint", name=f"{comp.id}_mount", type="fixed")
        ET.SubElement(joint, "parent", link="base_link")
        ET.SubElement(joint, "child", link=f"{comp.id}_link")
        ET.SubElement(joint, "origin", xyz=_v(comp.pos), rpy="0 0 0")

    return _pretty(robot)


def _pretty(elem) -> str:
    raw = ET.tostring(elem, "utf-8")
    return minidom.parseString(raw).toprettyxml(indent="  ")


def main():
    print("1/4  generating component meshes ...")
    meshgen.build_all(verbose=False)
    print("2/4  writing kinematic MJCF ...")
    os.makedirs(os.path.dirname(MJCF_PATH), exist_ok=True)
    with open(MJCF_PATH, "w") as f:
        f.write(build_mjcf(physics=False))
    print(f"       -> {os.path.relpath(MJCF_PATH, ROOT)}")
    print("3/4  writing physics MJCF ...")
    with open(MJCF_PHYS_PATH, "w") as f:
        f.write(build_mjcf(physics=True))
    print(f"       -> {os.path.relpath(MJCF_PHYS_PATH, ROOT)}")
    print("4/5  writing URDF ...")
    os.makedirs(os.path.dirname(URDF_PATH), exist_ok=True)
    with open(URDF_PATH, "w") as f:
        f.write(build_urdf())
    print(f"       -> {os.path.relpath(URDF_PATH, ROOT)}")
    print("5/5  regenerating facility scenes ...")
    from bravebot_sim import facility
    facility.build_scene_xml(facility.SCENE_PATH)
    facility.build_scene_xml(facility.SCENE_PHYS_PATH, robot="bravebot_physics.xml")
    print(f"       -> {os.path.relpath(facility.SCENE_PATH, ROOT)}, "
          f"{os.path.relpath(facility.SCENE_PHYS_PATH, ROOT)}")
    print(f"base standing height: {BASE_Z:.4f} m")


if __name__ == "__main__":
    main()
