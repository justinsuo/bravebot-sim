"""
BraveBot kinematic simulation on top of MuJoCo.

The default control mode is *kinematic patrol*: the floating base is driven by a
unicycle model (linear + angular velocity) and the wheels/legs are posed for
display. This is deterministic and never tips over, which is exactly what an
inspection-robot patrol / sensor-coverage simulation needs. Full rigid-body
dynamics (wheel actuators) are available for balance research but are not the
default — keeping a wheeled biped upright is the real-robot RL problem and is
out of scope here.

Coordinate frame: world +x forward, +y left, +z up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from . import registry as R


# stance height presets as (hip, knee) MAGNITUDES. Applied per-joint with the
# real LimX mirrored-axis signs (registry.STANCE: hip_L +, hip_R -, knee_L +,
# knee_R -) so BOTH legs bend symmetrically and both wheels seat on the floor
# together — a single raw (hip, knee) pair sent to both legs makes them rotate in
# opposite physical directions (one wheel floats ~15 cm).
STANCES = {
    "tall":    (0.15, 0.30),
    "nominal": (0.30, 0.60),
    "low":     (0.55, 1.10),
}


def _stance_pose(name: str) -> dict:
    """Per-leg-joint angle dict for a height preset, honoring the mirrored axes."""
    hip_mag, knee_mag = STANCES[name]
    pose = {}
    for j in R.LEG_JOINTS:
        n = j.name
        if n.startswith("hip"):
            pose[n] = math.copysign(hip_mag, R.STANCE[n])
        elif n.startswith("knee"):
            pose[n] = math.copysign(knee_mag, R.STANCE[n])
        elif n.startswith("abad"):
            pose[n] = 0.0
    return pose


@dataclass
class SensorReading:
    sensor_id: str
    modality: str
    target: str | None
    distance: float
    confidence: float
    note: str


class BraveBot:
    def __init__(self, model_path: str):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self._root_qadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")]
        self._leg_qadr = {
            j.name: self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j.name)]
            for j in R.LEG_JOINTS}

        # planar state
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.wheel_angle = 0.0
        self.stance = "nominal"
        self.v = 0.0
        self.omega = 0.0
        self.set_stance("nominal")

    # ------------------------------------------------------------------ #
    #  motion
    # ------------------------------------------------------------------ #
    MAX_V = 3.0       # m/s, wheeled-mode top speed from the spec
    MAX_OMEGA = 1.5   # rad/s

    def drive(self, v: float, omega: float):
        """Set commanded linear (m/s) and angular (rad/s) velocity."""
        self.v = float(np.clip(v, -self.MAX_V, self.MAX_V))
        self.omega = float(np.clip(omega, -self.MAX_OMEGA, self.MAX_OMEGA))

    def step(self, dt: float):
        """Advance the unicycle model and pose the wheels/legs."""
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw = _wrap(self.yaw + self.omega * dt)
        # wheels roll proportional to forward speed
        self.wheel_angle = _wrap(self.wheel_angle + (self.v / R.WHEEL_RADIUS) * dt)
        self.apply()

    def set_stance(self, name: str):
        if name not in STANCES:
            raise KeyError(f"unknown stance {name!r}; choose from {list(STANCES)}")
        self.stance = name
        self.apply()

    def apply(self):
        """Write planar pose + joint angles into MuJoCo and run FK."""
        q = self.data.qpos
        # base free joint: [x y z, qw qx qy qz]
        a = self._root_qadr
        q[a + 0] = self.x
        q[a + 1] = self.y
        q[a + 2] = self._stance_base_z()
        cz, sz = math.cos(self.yaw / 2), math.sin(self.yaw / 2)
        q[a + 3], q[a + 4], q[a + 5], q[a + 6] = cz, 0.0, 0.0, sz

        pose = _stance_pose(self.stance)
        for j in R.LEG_JOINTS:
            adr = self._leg_qadr[j.name]
            if j.name.startswith("wheel"):
                q[adr] = self.wheel_angle
            else:
                q[adr] = pose[j.name]
        mujoco.mj_forward(self.model, self.data)

    def _stance_base_z(self) -> float:
        # height of the wheel centre below base for the current stance (FK once).
        return _stance_height(self.model, self.data, self._root_qadr,
                              self._leg_qadr, _stance_pose(self.stance))

    # ------------------------------------------------------------------ #
    #  sensing
    # ------------------------------------------------------------------ #
    def sensor_frame(self, sensor_id: str):
        """World position and forward (look) unit vector of a sensor."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"s_{sensor_id}")
        pos = self.data.site_xpos[sid].copy()
        xmat = self.data.site_xmat[sid].reshape(3, 3)
        forward = xmat @ np.array([1.0, 0.0, 0.0])   # +x of sensor body
        return pos, forward / (np.linalg.norm(forward) + 1e-9)

    def scan(self, anomalies: "list[Anomaly]") -> list[SensorReading]:
        """Return readings for anomalies that fall in a sensor's FOV + range."""
        return scan_anomalies(self.sensor_frame, anomalies)


@dataclass
class Anomaly:
    name: str
    modality: str          # acoustic | thermal | gas | visual
    pos: tuple[float, float, float]
    note: str


def scan_anomalies(frame_fn, anomalies: "list[Anomaly]") -> list[SensorReading]:
    """Shared FOV/range scan used by both the kinematic and physics models.

    `frame_fn(sensor_id)` returns (world_pos, forward_unit_vector).
    """
    out: list[SensorReading] = []
    for comp in R.SENSOR_COMPONENTS:
        sensor = comp.sensor
        pos, fwd = frame_fn(comp.id)
        for anom in anomalies:
            if anom.modality != sensor.modality:
                continue
            rel = np.array(anom.pos) - pos
            dist = float(np.linalg.norm(rel))
            if dist > sensor.range_m or dist < 1e-3:
                continue
            cosang = float(np.dot(rel / dist, fwd))
            if cosang < math.cos(math.radians(sensor.fov_deg)):
                continue
            conf = max(0.0, min(0.99, (1.0 - dist / sensor.range_m) * (cosang ** 0.5)))
            if conf < 0.25:
                continue
            out.append(SensorReading(comp.id, sensor.modality, anom.name,
                                     dist, conf, anom.note))
    out.sort(key=lambda r: -r.confidence)
    return out


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _stance_height(model, data, root_qadr, leg_qadr, pose) -> float:
    """FK the legs at this stance pose, return base z so the LOWEST wheel touches z=0."""
    q = data.qpos
    q[root_qadr + 2] = 1.0   # provisional
    q[root_qadr + 3:root_qadr + 7] = [1, 0, 0, 0]
    for name, adr in leg_qadr.items():
        if name.startswith("wheel"):
            continue
        q[adr] = pose.get(name, 0.0)
    mujoco.mj_forward(model, data)
    wl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheelL_link")
    wr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheelR_link")
    wheel_z = min(data.xpos[wl][2], data.xpos[wr][2])
    drop = wheel_z - R.WHEEL_RADIUS     # how far the lowest wheel bottom is above 0 now
    return 1.0 - drop
