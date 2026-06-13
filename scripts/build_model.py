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

    Used for the BraveBot payload (original parts, no published inertials): their
    concatenated meshes have unreliable volume, so we never let MuJoCo infer mass
    from density. COM sits at the mesh centroid; inertia is the solid-box tensor.
    """
    ext, cen = _mesh_props(rel_mesh)
    x, y, z = (max(float(e), min_dim) for e in ext)
    ixx = mass / 12.0 * (y * y + z * z)
    iyy = mass / 12.0 * (x * x + z * z)
    izz = mass / 12.0 * (x * x + y * y)
    ET.SubElement(body, "inertial", pos=_v(cen), mass=f"{mass:.4f}",
                  diaginertia=f"{ixx:.5f} {iyy:.5f} {izz:.5f}")


def _add_real_inertial(body, key: str):
    """Exact LimX WF_TRON1A inertial (full tensor + COM) for a real link."""
    d = R.LIMX_INERTIAL[key]
    f = d["full"]   # ixx iyy izz ixy ixz iyz
    ET.SubElement(body, "inertial", pos=_v(d["com"]), mass=f"{d['m']:.4f}",
                  fullinertia=f"{f[0]:.6f} {f[1]:.6f} {f[2]:.6f} {f[3]:.6f} {f[4]:.6f} {f[5]:.6f}")


# --------------------------------------------------------------------------- #
#  Base standing height                                                        #
# --------------------------------------------------------------------------- #
# At the home leg pose the wheel centre sits at z = wheelL.z relative to base.
# Drop the base so the wheels just touch the z=0 floor.
BASE_Z = -R.TRON1_LINK_POS["wheelL"][2] + R.WHEEL_RADIUS


# --------------------------------------------------------------------------- #
#  MJCF                                                                         #
# --------------------------------------------------------------------------- #

# Leg position-servo gains (the joint controller). forcerange is the REAL LimX
# actuator effort (80 N·m legs). kv adds damping so the legs hold the stance.
LEG_SERVO = {"abad": dict(kp=600, kv=25, damping=4, friction=0.2, armature=0.02),
             "hip":  dict(kp=900, kv=45, damping=6, friction=0.2, armature=0.02),
             "knee": dict(kp=900, kv=45, damping=6, friction=0.2, armature=0.02)}
LEG_EFFORT = 80.0          # N·m, real LimX leg actuators
WHEEL_PEAK_TORQUE = 40.0   # N·m, real LimX wheel actuators
WHEEL_HALF_WIDTH = 0.024   # m, tire half-width (from wheel mesh extent)
# BraveBot active roll-stabilization: an actuated waist roll joint carrying the
# payload, so the policy can lean the upper body to control roll without splaying.
WAIST_PIVOT = (0.02, 0.0, 0.06)   # low pivot -> strong CoM-shift roll authority
WAIST_RANGE = 0.9                 # rad
WAIST_EFFORT = 120.0              # N·m (heavy payload)


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
    if physics:
        # Every robot geom collides with the floor (robot contype bit 2 meets the
        # floor's conaffinity bit 1) but NOT with other robot geoms (no shared
        # contype/conaffinity bit) — so the whole body is physically present and
        # cannot fall through the ground, with self-collision filtered out.
        ET.SubElement(default, "geom", contype="2", conaffinity="1", group="2",
                      density="500", friction="1 0.05 0.01", margin="0.001")
    else:
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
        _add_real_inertial(base, "base")
    else:
        ET.SubElement(base, "inertial", pos="0 0 -0.1", mass=str(LINK_MASS["base"]),
                      diaginertia="0.25 0.25 0.18")

    # BraveBot payload. In PHYSICS mode the whole payload hangs off an ACTUATED
    # waist roll joint, so the policy can lean the heavy upper body to actively
    # control roll (instead of splaying the legs) — its missing roll actuator.
    if physics:
        torso = ET.SubElement(base, "body", name="torso_roll", pos=_v(WAIST_PIVOT))
        ET.SubElement(torso, "joint", name="torso_roll", type="hinge", axis="1 0 0",
                      range=f"-{WAIST_RANGE} {WAIST_RANGE}", damping="3",
                      armature="0.05", frictionloss="0.1")
        ET.SubElement(torso, "inertial", pos="0 0 0.05", mass="0.3",
                      diaginertia="0.002 0.002 0.002")   # connector; mass is the children
        payload_parent, off = torso, WAIST_PIVOT
    else:
        payload_parent, off = base, (0.0, 0.0, 0.0)

    for comp in R.COMPONENTS:
        pos = (comp.pos[0] - off[0], comp.pos[1] - off[1], comp.pos[2] - off[2])
        body = ET.SubElement(payload_parent, "body", name=f"{comp.id}_link", pos=_v(pos))
        rgba = SENSOR_RGBA[comp.sensor.modality] if comp.sensor else GROUP_RGBA[comp.group]
        ET.SubElement(body, "geom", type="mesh", mesh=f"m_{comp.id}", rgba=rgba)
        if physics:
            _add_inertial(body, COMP_MASS[comp.id], f"bravebot/{comp.id}.stl")
        if comp.sensor:
            ET.SubElement(body, "site", name=f"s_{comp.id}", pos="0 0 0",
                          rgba=SENSOR_RGBA[comp.sensor.modality])

    # Leg chains
    bodies = {"base": base}
    for j in R.LEG_JOINTS:
        parent_body = bodies[_canonical(j.parent)]
        b = ET.SubElement(parent_body, "body", name=f"{j.child}_link", pos=_v(j.origin))
        link_key = _link_kind(j.child)
        is_wheel = "wheel" in j.child
        attrs = {"name": j.name, "type": "hinge", "axis": _v(j.axis), "pos": "0 0 0"}
        if j.type == "revolute":
            attrs["range"] = f"{j.lower} {j.upper}"        # real LimX joint limits
            if physics:
                s = LEG_SERVO[link_key]
                attrs.update(damping=str(s["damping"]), frictionloss=str(s["friction"]),
                             armature=str(s["armature"]))
        else:
            attrs["limited"] = "false"
            if physics:
                attrs.update(damping="0.05", frictionloss="0.01", armature="0.01")  # real wheel
        ET.SubElement(b, "joint", **attrs)

        rgba = "0.20 0.21 0.24 1" if is_wheel else "0.62 0.66 0.72 1"
        gattrs = dict(type="mesh", mesh=f"m_{_canonical_mesh(j.child)}", rgba=rgba)
        if is_wheel and physics:
            # tire = clean cylinder collider on the spin axis (y) so it ROLLS with
            # real grip instead of the faceted convex-hull mesh; high friction.
            gattrs.update(contype="0", conaffinity="0")    # mesh is visual only
            ET.SubElement(b, "geom", **gattrs)
            hw = WHEEL_HALF_WIDTH
            ET.SubElement(b, "geom", name=f"{j.child}_tire", type="cylinder",
                          fromto=f"0 {-hw} 0 0 {hw} 0", size=str(R.WHEEL_RADIUS),
                          contype="2", conaffinity="1", condim="3", priority="3",
                          friction="2.2 0.05 0.002", solref="0.01 1",
                          solimp="0.97 0.99 0.001", rgba="0.08 0.08 0.10 1")
        else:
            if is_wheel:
                gattrs.update(contype="1", conaffinity="1", condim="4", friction="1.4 0.02 0.001")
            elif not physics:
                gattrs.update(contype="0", conaffinity="0")   # kinematic: no collision
            ET.SubElement(b, "geom", **gattrs)

        if physics:
            _add_real_inertial(b, j.child)                 # exact LimX link inertial
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
        ET.SubElement(sens, "jointpos", name="torso_roll_q", joint="torso_roll")
        ET.SubElement(sens, "jointvel", name="torso_roll_w", joint="torso_roll")

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
                          forcerange=f"-{LEG_EFFORT} {LEG_EFFORT}")   # real 80 N·m
        for side in ("L", "R"):
            ET.SubElement(act, "motor", name=f"wheel_{side}_mot", joint=f"wheel_{side}",
                          gear="1", ctrlrange=f"-{WHEEL_PEAK_TORQUE} {WHEEL_PEAK_TORQUE}",
                          forcerange=f"-{WHEEL_PEAK_TORQUE} {WHEEL_PEAK_TORQUE}")   # real 40 N·m
        # waist roll: position servo (RL commands a lean target; controllers hold 0)
        ET.SubElement(act, "position", name="torso_roll_pos", joint="torso_roll",
                      kp="350", kv="20", ctrlrange=f"-{WAIST_RANGE} {WAIST_RANGE}",
                      forcerange=f"-{WAIST_EFFORT} {WAIST_EFFORT}")
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


def _real_inertial_urdf(link, key):
    d = R.LIMX_INERTIAL[key]
    f = d["full"]
    ine = ET.SubElement(link, "inertial")
    ET.SubElement(ine, "origin", xyz=_v(d["com"]), rpy="0 0 0")
    ET.SubElement(ine, "mass", value=f"{d['m']:.4f}")
    ET.SubElement(ine, "inertia", ixx=f"{f[0]:.6f}", iyy=f"{f[1]:.6f}", izz=f"{f[2]:.6f}",
                  ixy=f"{f[3]:.6f}", ixz=f"{f[4]:.6f}", iyz=f"{f[5]:.6f}")


def _link(robot, name, mesh_rel, mass, rgba="0.8 0.82 0.86 1", real_key=None):
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
    if real_key:
        _real_inertial_urdf(link, real_key)   # exact LimX inertial
    else:
        _inertial(link, mass, ext)
    return link


def build_urdf() -> str:
    robot = ET.Element("robot", name="bravebot")

    _link(robot, "base_link", "tron1/base_Link.stl", LINK_MASS["base"], real_key="base")

    # leg chain — real LimX axes, limits, inertials and actuator efforts
    for j in R.LEG_JOINTS:
        child_link = f"{j.child}_link"
        mesh = f"tron1/{R.TRON1_MESH[j.child]}.stl"
        rgba = "0.25 0.27 0.30 1" if "wheel" in j.child else "0.6 0.64 0.7 1"
        _link(robot, child_link, mesh, LINK_MASS[_link_kind(j.child)], rgba, real_key=j.child)
        jd = R.LIMX_JOINT[j.child]
        joint = ET.SubElement(robot, "joint", name=j.name,
                              type="continuous" if j.type == "continuous" else "revolute")
        ET.SubElement(joint, "parent", link=f"{_canonical(j.parent)}_link"
                      if j.parent != "base" else "base_link")
        ET.SubElement(joint, "child", link=child_link)
        ET.SubElement(joint, "origin", xyz=_v(j.origin), rpy="0 0 0")
        ET.SubElement(joint, "axis", xyz=_v(j.axis))
        if j.type == "revolute":
            ET.SubElement(joint, "limit", lower=str(j.lower), upper=str(j.upper),
                          effort=str(jd["effort"]), velocity=str(jd["vel"]))
        else:
            ET.SubElement(joint, "limit", effort=str(jd["effort"]), velocity=str(jd["vel"]))

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
