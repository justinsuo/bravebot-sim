"""
Heading-aware locomotion env — an ADDITIVE experiment on top of the shipped 40-d
policy, kept fully separate so the champion and all its tooling are untouched.

The shipped policy tracks commanded yaw *rate* and has no absolute-heading input,
so it drifts off-heading (the root of the "rotation is bad" behavior and of the
patrol's inability to traverse cleanly). This subclass adds a heading-error
observation (sin/cos of actual heading minus the integral of commanded yaw rate)
and a reward for holding it, so the policy can learn to drive a straight, commanded
heading. Obs grows 40 -> 42; everything else is inherited unchanged.

Train it on its own track (separate policy files):
    python scripts/rl_train.py --heading --steps 30_000_000 --envs 16
"""

from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces

from .env import BraveBotLocomotionEnv


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class HeadingAwareEnv(BraveBotLocomotionEnv):
    W_HEADING = 0.8        # reward weight for holding the commanded heading

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # +2 obs: sin/cos of heading error vs the integrated yaw-rate command
        self.observation_space = spaces.Box(-np.inf, np.inf, (42,), np.float32)
        self._yaw_des = 0.0

    def _base_yaw(self) -> float:
        m = self.data.xmat[self._base]          # flat row-major 3x3
        return math.atan2(float(m[3]), float(m[0]))   # atan2(R10, R00)

    def _heading_err(self) -> float:
        return _wrap(self._base_yaw() - self._yaw_des)

    def _obs(self, noisy=True):
        base = super()._obs(noisy)              # 40-d (inherited, incl. sensor noise)
        herr = self._heading_err()
        return np.concatenate([base, [math.sin(herr), math.cos(herr)]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self._yaw_des = 0.0                     # set before super().reset() calls _obs()
        obs, info = super().reset(seed=seed, options=options)
        self._yaw_des = self._base_yaw()        # sync to the settled upright heading (~0)
        return self._obs(), info

    def step(self, action):
        # integrate the commanded yaw rate into the desired heading BEFORE stepping,
        # so _obs() (called inside super().step) sees the matching heading error.
        self._yaw_des = _wrap(self._yaw_des + float(self._cmd[2]) / self.control_hz)
        obs, rew, term, trunc, info = super().step(action)
        rew += self.W_HEADING * math.exp(-4.0 * self._heading_err() ** 2)
        return obs, rew, term, trunc, info
