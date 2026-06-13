"""
Gymnasium environment for training a BraveBot locomotion policy (v2).

The real path to robust walking + turning: the policy learns to coordinate the
6 leg joints and 2 wheel motors to track a commanded body velocity (vx, vy, yaw)
while staying upright — like the real LimX TRON 1's Isaac-trained policy.

v2 over v1: projected-gravity orientation, a gait clock, richer reward shaping
(velocity/yaw tracking + orientation + height + smoothness + energy + alive),
DOMAIN RANDOMIZATION (mass, friction, COM, actuator strength, floor tilt) +
random pushes + action latency + observation noise for robustness and sim-to-real,
and a command / DR curriculum so it learns easy skills first.

Observation (37):
    projected_gravity(3), base gyro(3), base lin vel(3), leg qpos-stance(6),
    leg qvel(6), wheel speeds(2), gait clock sin/cos(2), prev action(8),
    command(vx, vy, yaw)(3), dr/cmd ramp(1)
Action (8, [-1,1]): 6 leg position offsets around stance + 2 wheel torques.
"""

from __future__ import annotations

import math
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
from .randomize import DomainRandomizer, ActionLatency, PushPerturber, DR_DEFAULT, PUSH_DEFAULT

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.normpath(os.path.join(_HERE, "..", "..", "description", "mjcf", "bravebot_physics.xml"))

LEG_JOINTS = ["abad_L", "hip_L", "knee_L", "abad_R", "hip_R", "knee_R"]
LEG_RANGE = np.array([0.4, 0.6, 0.6, 0.4, 0.6, 0.6])
WHEEL_TORQUE = 40.0
WHEEL_RADIUS = float(R.WHEEL_RADIUS)
STANCE_VEC = np.array([STANCE[j] for j in LEG_JOINTS])
# real LimX leg limits in LEG_JOINTS order, for a soft joint-limit barrier
LEG_LOWER = np.array([R.LIMX_JOINT[j.replace("_", "")]["lower"] for j in LEG_JOINTS])
LEG_UPPER = np.array([R.LIMX_JOINT[j.replace("_", "")]["upper"] for j in LEG_JOINTS])
ABAD_IDX = np.array([0, 3])         # ab/ad (lateral lean) DoFs
PITCHKNEE_IDX = np.array([1, 2, 4, 5])
GAIT_HZ = 1.4            # gait clock frequency
RAMP_STEPS = 200_000     # per-env control steps to ramp curriculum 0 -> 1


class BraveBotLocomotionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, control_hz: int = 50, episode_s: float = 16.0,
                 randomize: bool = True):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(MODEL)
        self.data = mujoco.MjData(self.model)
        self.sim_dt = float(self.model.opt.timestep)
        self.control_hz = control_hz
        self.decimation = max(1, int(round((1.0 / control_hz) / self.sim_dt)))
        self.max_steps = int(episode_s * control_hz)
        self.randomize = randomize

        nm = lambda n, o: mujoco.mj_name2id(self.model, o, n)
        self._leg_act = [nm(f"{j}_pos", mujoco.mjtObj.mjOBJ_ACTUATOR) for j in LEG_JOINTS]
        self._wheel_act = [nm("wheel_L_mot", mujoco.mjtObj.mjOBJ_ACTUATOR),
                           nm("wheel_R_mot", mujoco.mjtObj.mjOBJ_ACTUATOR)]
        self._leg_qadr = [self.model.jnt_qposadr[nm(j, mujoco.mjtObj.mjOBJ_JOINT)] for j in LEG_JOINTS]
        self._leg_vadr = [self.model.jnt_dofadr[nm(j, mujoco.mjtObj.mjOBJ_JOINT)] for j in LEG_JOINTS]
        self._wheel_vadr = [self.model.jnt_dofadr[nm(f"wheel_{s}", mujoco.mjtObj.mjOBJ_JOINT)] for s in "LR"]
        self._sid = {n: nm(n, mujoco.mjtObj.mjOBJ_SENSOR) for n in ("base_gyro", "base_vel")}
        self._base = nm("base_link", mujoco.mjtObj.mjOBJ_BODY)
        self._torso = nm("torso_link", mujoco.mjtObj.mjOBJ_BODY)
        self._ctrl = BalanceController(self.model, self.data, Gains())

        self._dr = DomainRandomizer(self.model, DR_DEFAULT)
        self._push = PushPerturber(self._torso, PUSH_DEFAULT)
        self._latency = ActionLatency(delay_steps=1)   # ~20 ms

        self.action_space = spaces.Box(-1.0, 1.0, (8,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (37,), np.float32)
        self._last_action = np.zeros(8, np.float32)
        self._prev_action = np.zeros(8, np.float32)
        self._cmd = np.zeros(3, np.float32)
        self._t = 0
        self._global = 0
        self._phase = 0.0
        self._stance_h = 0.78    # cached standing height (refined on first reset)
        self._cached_h = False

    # ------------------------------------------------------------------ #
    def _ramp(self):
        return float(min(1.0, self._global / RAMP_STEPS))

    def _sensor(self, name):
        i = self._sid[name]
        a = self.model.sensor_adr[i]
        return self.data.sensordata[a:a + self.model.sensor_dim[i]].copy()

    def _proj_gravity(self):
        m = self.data.xmat[self._base]      # world-from-body rotation (row-major)
        return np.array([-m[6], -m[7], -m[8]], np.float32)   # world-down in body frame

    def _obs(self, noisy=True):
        pg = self._proj_gravity()
        gyro = self._sensor("base_gyro")
        vel = self._sensor("base_vel")
        leg_q = self.data.qpos[self._leg_qadr] - STANCE_VEC
        leg_v = self.data.qvel[self._leg_vadr]
        wheel_v = self.data.qvel[self._wheel_vadr]
        clock = [math.sin(self._phase), math.cos(self._phase)]
        obs = np.concatenate([pg, gyro, vel, leg_q, leg_v, wheel_v, clock,
                              self._last_action, self._cmd, [self._ramp()]]).astype(np.float32)
        if noisy and self.randomize:
            obs[:3] += self.np_random.normal(0, 0.02, 3)      # gravity/orientation noise
            obs[3:6] += self.np_random.normal(0, 0.05, 3)     # gyro noise
            obs[6:9] += self.np_random.normal(0, 0.05, 3)     # vel noise
        return obs

    def _sample_cmd(self):
        r = self._ramp()
        # curriculum: grow command ranges as the ramp increases; vy comes in late
        vx = self.np_random.uniform(-0.4 - 0.3 * r, 0.5 + 0.4 * r)
        vy = self.np_random.uniform(-0.25, 0.25) * r
        yaw = self.np_random.uniform(-0.7, 0.7) * (0.3 + 0.7 * r)
        cmd = np.array([vx, vy, yaw], np.float32)
        if self.np_random.random() < 0.15:
            cmd[:] = 0.0       # some stand-still episodes
        return cmd

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize:
            self._dr.reset(self.np_random, scale=self._ramp())
        settle_upright(self.model, self.data, self._ctrl)
        if not self._cached_h:
            self._stance_h = float(self.data.xpos[self._base][2])
            self._cached_h = True
        self._cmd = self._sample_cmd()
        self._last_action[:] = 0.0
        self._prev_action[:] = 0.0
        self._latency.reset(8)
        self._push.reset(self.np_random, self.control_hz)
        self._t = 0
        self._phase = float(self.np_random.uniform(0, 2 * math.pi))
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        applied = self._latency(action) if self.randomize else action
        leg_targets = STANCE_VEC + applied[:6] * LEG_RANGE
        wheel_torque = applied[6:8] * WHEEL_TORQUE
        for aid, tgt in zip(self._leg_act, leg_targets):
            self.data.ctrl[aid] = tgt
        self.data.ctrl[self._wheel_act[0]] = wheel_torque[0]
        self.data.ctrl[self._wheel_act[1]] = wheel_torque[1]

        for k in range(self.decimation):
            if self.randomize:
                self._push.apply(self.data, self._t, self.np_random, self.control_hz,
                                 scale=self._ramp())
            mujoco.mj_step(self.model, self.data)
        self._phase = (self._phase + 2 * math.pi * GAIT_HZ / self.control_hz) % (2 * math.pi)

        # --- state ---
        pg = self._proj_gravity()
        gyro = self._sensor("base_gyro")
        vel = self._sensor("base_vel")
        height = float(self.data.xpos[self._base][2])
        wheel_w = self.data.qvel[self._wheel_vadr]                 # rad/s [L,R]
        leg_q = self.data.qpos[self._leg_qadr]
        leg_dev = leg_q - STANCE_VEC
        vx_cmd, vy_cmd, yaw_cmd = self._cmd

        # --- reward (wheel-legged locomotion shaping; synthesized design) ---
        # (A) command tracking
        r_vx = 1.5 * math.exp(-5.0 * (vel[0] - vx_cmd) ** 2)
        r_vy = 0.6 * math.exp(-8.0 * (vel[1] - vy_cmd) ** 2)
        r_yaw = 1.0 * math.exp(-5.0 * (gyro[2] - yaw_cmd) ** 2)
        # (B) upright via projected gravity, (C) base height, (D) alive
        r_orient = 0.8 * math.exp(-20.0 * (pg[0] ** 2 + pg[1] ** 2))
        r_height = 0.3 * math.exp(-80.0 * (height - self._stance_h) ** 2)
        r_alive = 0.25
        # (E) action-rate smoothness (biggest anti-wobble lever)
        p_arate = -0.05 * float(np.sum((action - self._prev_action) ** 2))
        # (F) energy: wheel mechanical power + small action L2
        p_energy = (-0.0006 * float(np.sum(np.abs(wheel_torque * wheel_w)))
                    - 0.0008 * float(np.sum(action ** 2)))
        # (G) posture on hips/knees, (H) soft joint-limit barrier
        p_pose = -0.15 * float(np.sum(leg_dev[PITCHKNEE_IDX] ** 2))
        over_hi = np.clip(leg_q - (LEG_UPPER - 0.1), 0, None)
        over_lo = np.clip((LEG_LOWER + 0.1) - leg_q, 0, None)
        p_limit = -2.0 * float(np.sum(over_hi ** 2 + over_lo ** 2))
        # (I) lateral-lean shaping so vy is attempted via the legs (wheels can't strafe)
        lean = float(leg_q[ABAD_IDX[0]] + leg_q[ABAD_IDX[1]])
        lean_des = float(np.clip(vy_cmd / 0.3, -1, 1)) * 0.4
        r_lean = 0.25 * math.exp(-15.0 * (lean - lean_des) ** 2)
        # (J) wheel-slip penalty (clean rolling, no skid)
        slip = WHEEL_RADIUS * float(np.mean(wheel_w)) - vel[0]
        p_slip = -0.2 * slip ** 2
        # (K) base lateral/vertical velocity damping, (L) roll/pitch rate damping
        p_bvel = -0.1 * (vel[1] - vy_cmd) ** 2 - 0.2 * float(vel[2]) ** 2
        p_gyro = -0.02 * float(gyro[0] ** 2 + gyro[1] ** 2)

        reward = (r_vx + r_vy + r_yaw + r_orient + r_height + r_alive + r_lean
                  + p_arate + p_energy + p_pose + p_limit + p_slip + p_bvel + p_gyro)

        roll = math.asin(max(-1, min(1, -self.data.xmat[self._base][7])))
        pitch = math.asin(max(-1, min(1, self.data.xmat[self._base][6])))
        fell = abs(pitch) > 0.8 or abs(roll) > 0.8 or height < 0.45 \
            or not np.isfinite(self.data.qpos).all()
        if fell:
            reward -= 5.0

        self._prev_action = action
        self._last_action = applied
        self._t += 1
        self._global += 1
        terminated = bool(fell)
        truncated = self._t >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, {}


def make_env():
    return BraveBotLocomotionEnv()
