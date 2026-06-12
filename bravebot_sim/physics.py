"""
Real-physics BraveBot: full MuJoCo rigid-body dynamics with the balance
controller keeping the wheel-legged biped upright while it drives.

Unlike the kinematic `BraveBot` (which poses the robot), `PhysicsBraveBot` runs
`mj_step` — gravity, contacts and actuator torques are all real. The two wheel
motors balance and drive the robot via `bravebot_sim.balance.BalanceController`;
the legs are held at the stance by stiff position servos.

Exposes the same `drive(v, omega)` / `step(dt)` / `sensor_frame` / `scan`
interface as the kinematic model, so the viewer and patrol code work unchanged.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import registry as R
from .balance import BalanceController, Gains, settle_upright
from .sim import scan_anomalies


class PhysicsBraveBot:
    # command limits exposed for the viewer (the controller also enforces these)
    MAX_V = BalanceController.MAX_V
    MAX_OMEGA = BalanceController.MAX_W

    def __init__(self, model_path: str, gains: Gains | None = None):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.ctrl = BalanceController(self.model, self.data, gains or Gains.load())
        self._timestep = self.model.opt.timestep
        self._v = 0.0
        self._w = 0.0
        self.reset_upright()

    # ---- pose / odometry (read from the free joint) ----
    @property
    def _root(self):
        return self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")]

    @property
    def x(self):
        return float(self.data.qpos[self._root + 0])

    @property
    def y(self):
        return float(self.data.qpos[self._root + 1])

    @property
    def yaw(self):
        w, x, y, z = self.data.qpos[self._root + 3:self._root + 7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    @property
    def stance(self):
        return "nominal"

    def reset_upright(self):
        settle_upright(self.model, self.data, self.ctrl)
        self._v = self._w = 0.0

    # ---- control ----
    def drive(self, v: float, omega: float):
        self._v, self._w = float(v), float(omega)

    def step(self, dt: float):
        """Advance real dynamics by dt, running the 500 Hz balance loop."""
        n = max(1, int(round(dt / self._timestep)))
        for _ in range(n):
            self.ctrl.control(self._v, self._w)
            mujoco.mj_step(self.model, self.data)

    def state(self):
        return self.ctrl.read_state()

    @property
    def fell(self):
        return self.ctrl.read_state().fell

    # ---- sensing (same interface as the kinematic model) ----
    def sensor_frame(self, sensor_id: str):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"s_{sensor_id}")
        pos = self.data.site_xpos[sid].copy()
        fwd = self.data.site_xmat[sid].reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
        return pos, fwd / (np.linalg.norm(fwd) + 1e-9)

    def scan(self, anomalies):
        return scan_anomalies(self.sensor_frame, anomalies)
