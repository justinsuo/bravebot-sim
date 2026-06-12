"""
BraveBot component registry — the single source of truth for the simulation.

Every component, its mounting frame (URDF-local: +x forward, +y left, +z up,
origin at the TRON 1 base link), and its engineering metadata are defined here.
Data is transcribed from the BraveBot product site (`bravebot-site/src/data/
bravebot.ts` and `src/config/landing.ts`).

The BraveBot is a wheel-legged autonomous inspection robot built on a modified
LimX Dynamics TRON 1 (WF_TRON1A) base. The base/leg/wheel links use the real
LimX meshes (Apache-2.0); every BraveBot modification is original geometry
generated procedurally in `meshgen.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Vec3 = tuple[float, float, float]


# --------------------------------------------------------------------------- #
#  LimX TRON 1 kinematic chain (real meshes)                                  #
# --------------------------------------------------------------------------- #
# Home-pose link positions in base frame, measured from the site's RobotScene.
# Each STL is authored link-local, so a joint origin is the child position minus
# the parent position (all frames are axis-aligned at the home pose).

TRON1_LINK_POS: dict[str, Vec3] = {
    "base":   (0.0,      0.0,     0.0),
    "abadL":  (0.05556,  0.105,  -0.2602),
    "abadR":  (0.05556, -0.105,  -0.2602),
    "hipL":   (-0.02144, 0.1255, -0.2602),
    "hipR":   (-0.02144, -0.1255, -0.2602),
    "kneeL":  (-0.17144, 0.105,  -0.52001),
    "kneeR":  (-0.17144, -0.105, -0.52001),
    "wheelL": (-0.02144, 0.1485, -0.77982),
    "wheelR": (-0.02144, -0.1485, -0.77982),
}

# Mesh file (without extension) for each real link.
TRON1_MESH = {
    "base": "base_Link", "abadL": "abad_L_Link", "abadR": "abad_R_Link",
    "hipL": "hip_L_Link", "hipR": "hip_R_Link", "kneeL": "knee_L_Link",
    "kneeR": "knee_R_Link", "wheelL": "wheel_L_Link", "wheelR": "wheel_R_Link",
}

WHEEL_RADIUS = 0.1297       # measured from wheel_*_Link.stl
WHEEL_SEPARATION = 0.297    # |wheelL.y - wheelR.y| = 2 * 0.1485


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    origin: Vec3
    axis: Vec3
    type: Literal["revolute", "continuous", "fixed"]
    lower: float = 0.0
    upper: float = 0.0
    home: float = 0.0


def _diff(child: str, parent: str) -> Vec3:
    c, p = TRON1_LINK_POS[child], TRON1_LINK_POS[parent]
    return (c[0] - p[0], c[1] - p[1], c[2] - p[2])


# --------------------------------------------------------------------------- #
#  Real LimX WF_TRON1A data (from limxdynamics/robot-description robot.urdf)    #
#  Inertials are full tensors; joint axes are MIRRORED on the right leg, exactly#
#  as the real robot. Actuators: 80 N·m legs / 40 N·m wheels, 15 / 40 rad/s.    #
# --------------------------------------------------------------------------- #

LIMX_INERTIAL = {
    'base':   dict(m=9.5850, com=(0.04576, 0.00014, -0.16398),  full=(0.140110, 0.110641, 0.098945, 0.000535, 0.028184, -0.000027)),
    'abadL':  dict(m=1.4690, com=(-0.06977, 0.04479, 0.00057),  full=(0.001555, 0.002359, 0.002081, 0.000398, -0.000013, -0.000001)),
    'hipL':   dict(m=2.3000, com=(-0.02869, -0.04770, -0.03992), full=(0.016937, 0.022853, 0.009334, 0.001647, -0.009233, 0.002202)),
    'kneeL':  dict(m=1.4900, com=(0.11913, 0.01106, -0.20363),  full=(0.013233, 0.017661, 0.005017, -0.000435, 0.006936, 0.000791)),
    'wheelL': dict(m=1.0800, com=(0.00003, 0.00807, -0.00002),  full=(0.005155, 0.009743, 0.005154, 0.0, -0.000001, 0.0)),
    'abadR':  dict(m=1.4690, com=(-0.06977, -0.04479, 0.00057), full=(0.001555, 0.002359, 0.002081, -0.000398, -0.000013, 0.000001)),
    'hipR':   dict(m=2.3000, com=(-0.02869, 0.04770, -0.03992), full=(0.016937, 0.022853, 0.009334, -0.001647, -0.009233, -0.002202)),
    'kneeR':  dict(m=1.4900, com=(0.11913, -0.01106, -0.20363), full=(0.013233, 0.017661, 0.005017, 0.000435, 0.006936, -0.000791)),
    'wheelR': dict(m=1.0800, com=(0.00003, -0.00807, -0.00002), full=(0.005155, 0.009743, 0.005154, 0.0, -0.000001, 0.0)),
}

LIMX_JOINT = {
    'abadL':  dict(axis=(1, 0, 0),  lower=-0.38397, upper=1.39626, effort=80, vel=15, continuous=False),
    'hipL':   dict(axis=(0, 1, 0),  lower=-1.01229, upper=1.39626, effort=80, vel=15, continuous=False),
    'kneeL':  dict(axis=(0, -1, 0), lower=-0.87267, upper=1.36136, effort=80, vel=15, continuous=False),
    'wheelL': dict(axis=(0, 1, 0),  lower=0.0, upper=0.0, effort=40, vel=40, continuous=True),
    'abadR':  dict(axis=(1, 0, 0),  lower=-1.39626, upper=0.38397, effort=80, vel=15, continuous=False),
    'hipR':   dict(axis=(0, -1, 0), lower=-1.39626, upper=1.01229, effort=80, vel=15, continuous=False),
    'kneeR':  dict(axis=(0, 1, 0),  lower=-1.36136, upper=0.87267, effort=80, vel=15, continuous=False),
    'wheelR': dict(axis=(0, 1, 0),  lower=0.0, upper=0.0, effort=40, vel=40, continuous=True),
}

# Standing stance in the real (mirrored-axis) convention: the same physical
# bent-leg pose as before, with the joint signs the LimX axes require.
STANCE = {"abad_L": 0.0, "hip_L": 0.30, "knee_L": 0.60,
          "abad_R": 0.0, "hip_R": -0.30, "knee_R": -0.60}

# The WF_TRON1A leg: abad (roll) -> hip (pitch) -> knee (pitch) -> wheel (drive).
LEG_JOINTS: list[Joint] = []
for side in ("L", "R"):
    for base in ("abad", "hip", "knee", "wheel"):
        child = f"{base}{side}"
        parent = "base" if base == "abad" else (
            f"abad{side}" if base == "hip" else f"hip{side}" if base == "knee" else f"knee{side}")
        jd = LIMX_JOINT[child]
        name = f"{base}_{side}"
        LEG_JOINTS.append(Joint(
            name, parent, child, _diff(child, "base" if base == "abad" else parent),
            jd["axis"], "continuous" if jd["continuous"] else "revolute",
            jd["lower"], jd["upper"], STANCE.get(name, 0.0)))


# --------------------------------------------------------------------------- #
#  BraveBot modification components (original geometry)                        #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Component:
    id: str                 # matches a builder in meshgen.PART_PRIMITIVES
    name: str
    group: str              # structural | power | compute | sensor | safety | comms
    pos: Vec3               # mount frame in base-link coordinates (home pose)
    summary: str
    sensor: "Sensor | None" = None


@dataclass(frozen=True)
class Sensor:
    modality: Literal["acoustic", "thermal", "gas", "visual", "nav"]
    spec: str
    fov_deg: float          # cone half-angle for the simulated field of view
    range_m: float
    look: Vec3 = (1, 0, 0)  # forward axis of the sensor in its own frame
    detects: tuple[str, ...] = ()


SENSORS: dict[str, Sensor] = {
    "acoustic": Sensor(
        "acoustic", "Phased acoustic imaging array, ultrasonic up to 100 kHz", 35.0, 8.0,
        detects=("coolant micro-leak", "partial discharge", "arcing", "bearing wear")),
    "thermal": Sensor(
        "thermal", "Radiometric infrared camera, per-zone baseline learning", 24.0, 12.0,
        detects=("GPU hotspot", "loose connection", "thermal runaway", "aisle imbalance")),
    "gas": Sensor(
        "gas", "VOC / CO / H2 / off-gas; optional OGI / TDLAS (OTC)", 45.0, 4.0,
        detects=("battery off-gas", "refrigerant leak", "flammable gas", "methane")),
    "visual": Sensor(
        "visual", "HD camera with on-board gauge OCR, object & scene detection", 30.0, 10.0,
        detects=("open containment", "missing seal", "asset drift", "gauge reading")),
    "nav": Sensor(
        "nav", "LiDAR + depth navigation cluster: mapping, obstacle avoidance", 60.0, 15.0,
        detects=("obstacle", "free space", "localization")),
}

# pos values are the home-pose mount frames from the site's RobotScene part list.
COMPONENTS: list[Component] = [
    # --- structural ---------------------------------------------------------
    Component("payload", "Payload adapter frame", "structural", (0.02, 0.0, 0.045),
              "Custom mounting layer bolted on top of the TRON 1 base."),
    Component("torso", "Rugged inspection torso", "structural", (0.02, 0.0, 0.215),
              "Sealed IP66 enclosure housing inspection electronics and edge compute."),
    Component("railL", "Protective rail (left)", "structural", (0.03, 0.176, -0.1),
              "Impact rail cage and field handling point."),
    Component("railR", "Protective rail (right)", "structural", (0.03, -0.176, -0.1),
              "Impact rail cage and field handling point."),
    Component("head", "Sensor head housing", "structural", (0.02, 0.0, 0.57),
              "Houses the four-sensor stack at the top of the mast."),
    Component("mast", "Sensor mast", "structural", (0.02, 0.0, 0.43),
              "Raises the sensor head for an elevated vantage over racks and aisles."),
    # --- power --------------------------------------------------------------
    Component("battery", "Hot-swappable battery", "power", (0.02, 0.0, 0.12),
              "2-4 h runtime, ~10 s swap for continuous 24/7 patrol."),
    # --- compute ------------------------------------------------------------
    Component("edgeai", "Edge AI core", "compute", (-0.03, 0.0, 0.22),
              "Local multimodal reasoning: fusion, diagnosis, risk scoring. No cloud."),
    Component("display", "Front AI display panel", "compute", (0.165, 0.0, 0.215),
              "Local status, patrol mode and operator feedback."),
    Component("rear", "Rear service panel", "structural", (-0.12, 0.0, 0.215),
              "Tool-free maintenance access and modular service layout."),
    # --- sensors ------------------------------------------------------------
    Component("acoustic", "Acoustic imaging array", "sensor", (0.14, 0.075, 0.58),
              "Detects ultrasonic leaks, partial discharge and bearing wear.",
              SENSORS["acoustic"]),
    Component("thermal", "Thermal camera", "sensor", (0.145, 0.0, 0.6),
              "Maps hotspots across UPS, PDU, cable joints, GPUs and racks.",
              SENSORS["thermal"]),
    Component("hdcam", "HD visual AI camera", "sensor", (0.145, -0.075, 0.58),
              "Reads gauges, assets, doors, seals and indicator lights.",
              SENSORS["visual"]),
    Component("gas", "Gas / laser telemetry sensor", "sensor", (0.14, 0.0, 0.535),
              "VOC / CO / H2 off-gas; optional OGI / TDLAS for industrial gas.",
              SENSORS["gas"]),
    Component("nav", "Navigation sensor cluster", "sensor", (0.145, 0.0, 0.48),
              "Mapping, obstacle avoidance and route following.",
              SENSORS["nav"]),
    # --- comms & safety -----------------------------------------------------
    Component("antennaL", "Communication antenna (left)", "comms", (0.0, 0.07, 0.71),
              "Network sync and command / communication link."),
    Component("antennaR", "Communication antenna (right)", "comms", (0.0, -0.07, 0.71),
              "Network sync and command / communication link."),
    Component("estop", "Emergency stop", "safety", (-0.1, 0.11, 0.31),
              "Hardware safety cut-off for operation around people and equipment."),
]

COMPONENT_BY_ID = {c.id: c for c in COMPONENTS}
SENSOR_COMPONENTS = [c for c in COMPONENTS if c.sensor is not None]


# --------------------------------------------------------------------------- #
#  Headline specifications (from the site specs table)                         #
# --------------------------------------------------------------------------- #

SPECS: dict[str, str] = {
    "robot_type": "Wheel-legged autonomous inspection robot",
    "base_platform": "Modified LimX Dynamics TRON 1 (WF_TRON1A) architecture",
    "chassis_footprint": "~9 in x 11 in",
    "chassis_weight_kg": "21.8 (48 lb body)",
    "sensor_payload_kg": "15.0 (33 lb stack, config-dependent)",
    "mobility": "Wheel-legged, stair / slope capable",
    "slope_capability": "up to 30 deg",
    "step_capability": "up to ~10 in (estimated)",
    "ruggedness": "IP66 (target)",
    "runtime": "2-4 h per charge (estimated)",
    "battery": "Hot-swappable, ~10 s swap",
    "sensor_stack": "Acoustic / thermal / gas / visual",
    "ai": "Edge-resident multimodal AI / local reasoning",
    "integrations": "OPC UA / MQTT / REST / BMS / DCIM / CMMS",
    "wheel_radius_m": f"{WHEEL_RADIUS:.4f}",
    "wheel_separation_m": f"{WHEEL_SEPARATION:.4f}",
    "max_speed_mps": "3.0 (wheeled mode)",
}

# Edge-AI mixture-of-experts fusion layer (from the site's edge-AI section).
EXPERTS = (
    "Acoustic expert", "Thermal expert", "Gas / safety expert",
    "Visual inspection expert", "Maintenance / work-order expert",
)
