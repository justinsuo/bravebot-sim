"""
Real-physics balance + drive controller for the BraveBot wheel-legged biped.

The robot is a Segway-class wheeled inverted pendulum: the two wheels share one
axle, so it is passively stable in roll but unstable in pitch, with a high CoM
(~0.69 m above the axle, ~34 kg). The legs are held rigidly at a stance by stiff
position servos (in the MJCF); the two wheel torque motors do all the balancing,
driving and steering.

Control law (cascaded state feedback, signs calibrated against the model):
    pitch      = asin(base_zaxis_x)          # + = leaning forward (+x)
    pitch_rate = gyro_y                       # same sign as d(pitch)/dt
    v          = base_vel_x                    # forward ground speed
    yaw_rate   = gyro_z

    pitch_ref  = clip(k_pv * (v_cmd - v), -lean_max, lean_max)   # lean to accel
    tau_bal    = +(k_p*(pitch - pitch_ref) + k_d*pitch_rate)     # see sign note below
    tau_turn   = k_yaw*(omega_cmd - yaw_rate)
    tau_L = tau_bal - tau_turn ;  tau_R = tau_bal + tau_turn

Sign note (verified empirically against bravebot_physics.xml): a positive wheel
torque produces a reaction torque on the body that pitches it BACKWARD, which is
what corrects a forward lean — so tau_bal carries a leading PLUS sign. (Measuring
base velocity at the high IMU site is misleading here because it mixes the
reaction-torque tip with translation; the closed-loop test is authoritative —
+sign stays upright >30 s, -sign falls in 0.2 s.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import registry as R

# Stance the leg servos hold — per-joint in the real LimX (mirrored-axis)
# convention (registry.STANCE), e.g. hip_L=+0.30, knee_L=+0.60, hip_R=-0.30,
# knee_R=-0.60: the same physical bent-leg pose the real robot stands in.
STANCE = R.STANCE


# Default gains: calm/stable baseline (scripts/eval_balance.py --tune to retune).
@dataclass
class Gains:
    k_p: float = 372.0      # pitch proportional   (N·m / rad)
    k_d: float = 18.7       # pitch derivative     (N·m / (rad/s))
    k_pv: float = 0.16      # velocity -> lean ref (rad / (m/s))
    lean_max: float = 0.20  # max commanded lean   (rad) — modest for control
    k_yaw: float = 14.0     # yaw-rate tracking    (N·m / (rad/s))
    k_iv: float = 0.45      # velocity-error integral -> lean (station-keeping)

    # as_vector/from_vector cover the 5 tuned gains; k_iv is fixed (not tuned).
    def as_vector(self) -> np.ndarray:
        return np.array([self.k_p, self.k_d, self.k_pv, self.lean_max, self.k_yaw])

    @staticmethod
    def from_vector(v) -> "Gains":
        return Gains(*[float(x) for x in v])

    @staticmethod
    def load(path: str | None = None) -> "Gains":
        """Load tuned gains from gains.json, falling back to the defaults."""
        import json
        import os
        path = path or os.path.join(os.path.dirname(__file__), "gains.json")
        try:
            with open(path) as f:
                return Gains.from_vector(json.load(f)["gains"])
        except (FileNotFoundError, KeyError, ValueError):
            return Gains()


@dataclass
class State:
    pitch: float
    pitch_rate: float
    v: float
    yaw_rate: float
    roll: float
    roll_rate: float
    height: float

    @property
    def fell(self) -> bool:
        return abs(self.pitch) > 0.6 or abs(self.roll) > 0.6 or self.height < 0.45


class BalanceController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 gains: Gains | None = None):
        self.m = model
        self.d = data
        self.gains = gains or Gains()

        self._sid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, n)
                     for n in ("base_zaxis", "base_gyro", "base_vel")}
        self._leg_act = {j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_pos")
                         for j in ("abad_L", "hip_L", "knee_L", "abad_R", "hip_R", "knee_R")}
        self._wL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_L_mot")
        self._wR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_R_mot")
        self._imu_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._tau_clip = float(model.actuator_ctrlrange[self._wL, 1])
        self._v_cmd = 0.0
        self._w_cmd = 0.0
        self._v_int = 0.0      # velocity-error integral (station-keeping)
        self._turn_used = 0.0  # cumulative-turn budget consumed
        self._turn_locked = False
        self._cooldown = 0     # steps of mandatory straight time remaining
        self.hold_stance()

    # command envelope: a high-CoM single-axle wheeled biped tips if asked to
    # turn/accelerate too hard, so cap and slew-limit the commands.
    MAX_V = 1.1          # m/s — controllable top speed
    MAX_W = 0.45         # rad/s ceiling
    V_SLEW = 2.5         # m/s per second — moderate, not jerky
    W_SLEW = 3.0         # rad/s per second
    DT_CMD = 0.002       # control period
    # Proactive turn budget + DIRECTIONAL lock. The robot has NO roll actuator, so a
    # sustained same-direction arc leans the body (sign(roll) follows the accumulated
    # turn) and the lean does not decay on its own. We spend a budget while turning
    # and lock after a burst — which stops turning EARLY (at roll ~0.028, a big tip
    # margin) and is what keeps every scenario upright. The crucial fix over a plain
    # lock: while locked we block ONLY the turn that would WORSEN the lean and always
    # let the OPPOSITE (releveling) turn through, so an opposite command can relevel
    # and re-arm — the symmetric lock instead froze turning forever (verified bug).
    TURN_BUDGET = 0.6        # rad of cumulative turn per burst (~34 deg)
    TURN_RECOVER = 0.6       # rad/s budget recovery while going straight
    TURN_COOLDOWN_S = 2.5    # mandatory straight time after a burst (roll settles)
    ROLL_SETTLED = 0.025     # rad — roll this small (+ settled) re-arms full turning
    ROLLRATE_SETTLED = 0.04  # rad/s
    ROLL_LIMIT = 0.06        # rad (~3.4 deg) — hard backstop: lock if roll exceeds it

    # ---- sensor read ----
    def _sensor(self, name: str) -> np.ndarray:
        i = self._sid[name]
        a = self.m.sensor_adr[i]
        return self.d.sensordata[a:a + self.m.sensor_dim[i]]

    def read_state(self) -> State:
        za = self._sensor("base_zaxis")
        gyro = self._sensor("base_gyro")
        vel = self._sensor("base_vel")
        pitch = math.asin(float(np.clip(za[0], -1, 1)))
        roll = math.asin(float(np.clip(-za[1], -1, 1)))
        height = float(self.d.xpos[self._imu_body][2])
        return State(pitch=pitch, pitch_rate=float(gyro[1]), v=float(vel[0]),
                     yaw_rate=float(gyro[2]), roll=roll, roll_rate=float(gyro[0]),
                     height=height)

    def reset(self):
        """Clear command-shaping and integral state (call when teleporting)."""
        self._v_cmd = self._w_cmd = self._v_int = self._turn_used = 0.0
        self._turn_locked = False
        self._cooldown = 0

    # ---- leg stance ----
    def hold_stance(self, stance: dict | None = None):
        st = stance or STANCE
        for joint, aid in self._leg_act.items():
            self.d.ctrl[aid] = st[joint]

    # ---- control ----
    @staticmethod
    def _finite(x: float) -> float:
        return x if math.isfinite(x) else 0.0

    def _shape_command(self, v_cmd: float, omega_cmd: float, roll: float, roll_rate: float):
        v_t = float(np.clip(self._finite(v_cmd), -self.MAX_V, self.MAX_V))
        w_t = float(np.clip(self._finite(omega_cmd), -self.MAX_W, self.MAX_W))
        # hard backstop: if roll exceeds the safety limit, lock turning immediately.
        if abs(roll) > self.ROLL_LIMIT and not self._turn_locked:
            self._turn_locked = True
            self._cooldown = int(self.TURN_COOLDOWN_S / self.DT_CMD)
        # spend the turn budget while turning (any direction); recover it straight.
        if abs(w_t) > 0.02 and not self._turn_locked:
            self._turn_used += abs(w_t) * (1.0 + 2.5 * abs(self._v_cmd)) * self.DT_CMD
            if self._turn_used >= self.TURN_BUDGET:
                self._turn_locked = True
                self._cooldown = int(self.TURN_COOLDOWN_S / self.DT_CMD)
        else:
            self._turn_used = max(0.0, self._turn_used - self.TURN_RECOVER * self.DT_CMD)
        if self._turn_locked:
            self._cooldown -= 1
            if self._cooldown <= 0 and self._turn_used <= 0.0 \
                    and abs(roll) < self.ROLL_SETTLED and abs(roll_rate) < self.ROLLRATE_SETTLED:
                self._turn_locked = False          # roll recovered -> full turning
            elif w_t * roll >= 0.0:
                w_t = 0.0                          # block only a lean-WORSENING turn;
                # a releveling turn (opposite sign to roll) passes -> can recover.
        dv = self.V_SLEW * self.DT_CMD
        dw = self.W_SLEW * self.DT_CMD
        self._v_cmd += float(np.clip(v_t - self._v_cmd, -dv, dv))
        self._w_cmd += float(np.clip(w_t - self._w_cmd, -dw, dw))
        return self._v_cmd, self._w_cmd

    def control(self, v_cmd: float = 0.0, omega_cmd: float = 0.0) -> State:
        g = self.gains
        s = self.read_state()
        v_cmd, omega_cmd = self._shape_command(v_cmd, omega_cmd, s.roll, s.roll_rate)
        # velocity loop with an anti-windup integral so it actually station-keeps
        # (the CoM sits ahead of the axle, needing a steady lean that pure-P can
        # only hold via a permanent velocity error -> forward creep).
        v_err = v_cmd - s.v
        self._v_int = float(np.clip(self._v_int + g.k_iv * v_err * self.DT_CMD,
                                    -g.lean_max, g.lean_max))
        pitch_ref = float(np.clip(g.k_pv * v_err + self._v_int, -g.lean_max, g.lean_max))
        tau_bal = +(g.k_p * (s.pitch - pitch_ref) + g.k_d * s.pitch_rate)
        tau_turn = g.k_yaw * (omega_cmd - s.yaw_rate)
        c = self._tau_clip
        self.d.ctrl[self._wL] = self._finite(float(np.clip(tau_bal - tau_turn, -c, c)))
        self.d.ctrl[self._wR] = self._finite(float(np.clip(tau_bal + tau_turn, -c, c)))
        self.hold_stance()
        return s


def stance_base_height(model, data) -> float:
    """Forward-kinematics base height so the wheels rest on z=0 at the stance.

    The legs are bent at the stance in the mirrored LimX convention (hip_L 0.30 /
    knee_L 0.60; hip_R -0.30 / knee_R -0.60), which tucks the wheels up relative to
    the base — so the correct standing height differs from the straight-leg home
    pose. Get it from FK, not from a fixed constant.
    """
    qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")]
    data.qpos[qadr:qadr + 3] = [0, 0, 1.0]
    data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
    for j in R.LEG_JOINTS:
        if j.type != "revolute":
            continue
        adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.name)]
        data.qpos[adr] = STANCE[j.name]
    mujoco.mj_forward(model, data)
    wl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheelL_link")
    wheel_z = data.xpos[wl][2]
    return 1.0 - (wheel_z - R.WHEEL_RADIUS)   # drop base so wheel bottom -> 0


def settle_upright(model, data, controller: "BalanceController", clearance: float = 0.002):
    """Reset to a clean upright start pose with the legs at stance."""
    h = stance_base_height(model, data)
    mujoco.mj_resetData(model, data)
    qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")]
    data.qpos[qadr + 2] = h + clearance
    data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
    for j in R.LEG_JOINTS:
        if j.type != "revolute":
            continue
        adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.name)]
        data.qpos[adr] = STANCE[j.name]
    controller.reset()
    controller.hold_stance()
    mujoco.mj_forward(model, data)
