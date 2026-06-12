"""
Gymnasium environment for training a BraveBot locomotion policy.

This is the *proper* path to robust walking + turning: the policy learns to
coordinate the 6 leg joints and 2 wheel motors to track a commanded body
velocity (vx, vy, yaw rate) while staying upright — the same approach the real
LimX TRON 1 uses (an Isaac Gym / Isaac Lab trained policy). Hand-tuned control
can't do it (the ab/ad joints are far too weak to control roll during turns),
so this is where walking and clean rotation actually come from.

Observation (command-conditioned):
    pitch, roll, gyro(3), base lin vel(3), leg qpos(6), leg qvel(6),
    wheel speeds(2), previous action(8), command(vx, vy, yaw)(3)   -> 33 dims
Action (8, in [-1, 1]):
    6 leg position offsets around the stance + 2 wheel torques
Reward:
    velocity-command tracking + upright + alive - energy/effort; episode ends on
    a fall.

Train with `scripts/rl_train.py` (PPO). Designed to run on CPU for validation;
full convergence wants a GPU and millions of steps.
"""

from __future__ import annotations

import os

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install gymnasium stable-baselines3") from e

import mujoco

from .. import registry as R
from ..balance import STANCE, settle_upright, BalanceController, Gains

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.normpath(os.path.join(_HERE, "..", "..", "description", "mjcf", "bravebot_physics.xml"))

LEG_JOINTS = ["abad_L", "hip_L", "knee_L", "abad_R", "hip_R", "knee_R"]
LEG_RANGE = np.array([0.4, 0.6, 0.6, 0.4, 0.6, 0.6])   # action -> rad offset
WHEEL_TORQUE = 40.0   # real LimX wheel actuator
STANCE_VEC = np.array([STANCE[j] for j in LEG_JOINTS])


class BraveBotLocomotionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, control_hz: int = 50, episode_s: float = 12.0,
                 cmd_ranges=((-0.5, 0.8), (-0.3, 0.3), (-0.6, 0.6))):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(MODEL)
        self.data = mujoco.MjData(self.model)
        self.sim_dt = float(self.model.opt.timestep)
        self.decimation = max(1, int(round((1.0 / control_hz) / self.sim_dt)))
        self.max_steps = int(episode_s * control_hz)
        self.cmd_ranges = np.array(cmd_ranges, dtype=np.float32)

        nm = lambda n, o: mujoco.mj_name2id(self.model, o, n)
        self._leg_act = [nm(f"{j}_pos", mujoco.mjtObj.mjOBJ_ACTUATOR) for j in LEG_JOINTS]
        self._wheel_act = [nm("wheel_L_mot", mujoco.mjtObj.mjOBJ_ACTUATOR),
                           nm("wheel_R_mot", mujoco.mjtObj.mjOBJ_ACTUATOR)]
        self._leg_qadr = [self.model.jnt_qposadr[nm(j, mujoco.mjtObj.mjOBJ_JOINT)] for j in LEG_JOINTS]
        self._leg_vadr = [self.model.jnt_dofadr[nm(j, mujoco.mjtObj.mjOBJ_JOINT)] for j in LEG_JOINTS]
        self._wheel_vadr = [self.model.jnt_dofadr[nm(f"wheel_{s}", mujoco.mjtObj.mjOBJ_JOINT)] for s in "LR"]
        self._sid = {n: nm(n, mujoco.mjtObj.mjOBJ_SENSOR) for n in ("base_zaxis", "base_gyro", "base_vel")}
        self._imu_body = nm("base_link", mujoco.mjtObj.mjOBJ_BODY)
        self._ctrl = BalanceController(self.model, self.data, Gains())  # for settle_upright reuse

        self.action_space = spaces.Box(-1.0, 1.0, (8,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (33,), np.float32)
        self._last_action = np.zeros(8, np.float32)
        self._cmd = np.zeros(3, np.float32)
        self._t = 0

    # ------------------------------------------------------------------ #
    def _sensor(self, name):
        i = self._sid[name]
        a = self.model.sensor_adr[i]
        return self.data.sensordata[a:a + self.model.sensor_dim[i]].copy()

    def _state(self):
        za = self._sensor("base_zaxis")
        pitch = float(np.arcsin(np.clip(za[0], -1, 1)))
        roll = float(np.arcsin(np.clip(-za[1], -1, 1)))
        gyro = self._sensor("base_gyro")
        vel = self._sensor("base_vel")
        height = float(self.data.xpos[self._imu_body][2])
        return pitch, roll, gyro, vel, height

    def _obs(self):
        pitch, roll, gyro, vel, _ = self._state()
        leg_q = self.data.qpos[self._leg_qadr] - STANCE_VEC
        leg_v = self.data.qvel[self._leg_vadr]
        wheel_v = self.data.qvel[self._wheel_vadr]
        return np.concatenate([[pitch, roll], gyro, vel, leg_q, leg_v, wheel_v,
                               self._last_action, self._cmd]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        settle_upright(self.model, self.data, self._ctrl)
        lo, hi = self.cmd_ranges[:, 0], self.cmd_ranges[:, 1]
        self._cmd = self.np_random.uniform(lo, hi).astype(np.float32)
        # ~20% of episodes are "stand still" so it also learns to station-keep
        if self.np_random.random() < 0.2:
            self._cmd[:] = 0.0
        self._last_action[:] = 0.0
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        leg_targets = STANCE_VEC + action[:6] * LEG_RANGE
        wheel_torque = action[6:8] * WHEEL_TORQUE
        for aid, tgt in zip(self._leg_act, leg_targets):
            self.data.ctrl[aid] = tgt
        self.data.ctrl[self._wheel_act[0]] = wheel_torque[0]
        self.data.ctrl[self._wheel_act[1]] = wheel_torque[1]
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        pitch, roll, gyro, vel, height = self._state()
        vx_cmd, vy_cmd, yaw_cmd = self._cmd
        # reward: command tracking + upright + alive - effort
        r_track = (1.5 * np.exp(-4.0 * (vel[0] - vx_cmd) ** 2)
                   + 0.5 * np.exp(-4.0 * (vel[1] - vy_cmd) ** 2)
                   + 1.0 * np.exp(-4.0 * (gyro[2] - yaw_cmd) ** 2))
        r_upright = 0.8 * np.exp(-8.0 * (pitch ** 2 + roll ** 2))
        r_alive = 0.5
        r_effort = -0.002 * float(np.sum(action ** 2)) - 0.05 * float(np.sum(gyro ** 2))
        reward = float(r_track + r_upright + r_alive + r_effort)

        fell = abs(pitch) > 0.7 or abs(roll) > 0.7 or height < 0.45 \
            or not np.isfinite(self.data.qpos).all()
        self._t += 1
        terminated = bool(fell)
        truncated = self._t >= self.max_steps
        if fell:
            reward -= 5.0
        self._last_action = action
        return self._obs(), reward, terminated, truncated, {}


def make_env():
    return BraveBotLocomotionEnv()
