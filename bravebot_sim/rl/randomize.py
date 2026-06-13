"""
Domain randomization + perturbation helpers for sim-to-real locomotion training.

Randomizing dynamics each episode forces the policy to be robust to model error,
which is what lets a sim-trained policy transfer to the real LimX TRON 1. All
helpers cache the nominal model values once and rescale around them per reset, so
randomization never compounds across episodes.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class DomainRandomizer:
    """Per-episode randomization of MuJoCo dynamics fields, ramped by a scale."""

    def __init__(self, model, cfg: dict):
        self.m = model
        self.cfg = cfg
        # cache nominal values
        self._mass0 = model.body_mass.copy()
        self._fric0 = model.geom_friction.copy()
        self._damp0 = model.dof_damping.copy()
        self._gain0 = model.actuator_gainprm.copy()
        self._bias0 = model.actuator_biasprm.copy()
        self._com0 = model.body_ipos.copy()

    def reset(self, rng: np.random.Generator, scale: float = 1.0):
        """Randomize model dynamics in place. `scale` in [0,1] ramps DR strength."""
        c = self.cfg

        # body mass (all bodies) + payload/base extra
        self.m.body_mass[:] = self._mass0 * rng.uniform(
            1 - scale * c["mass_pct"], 1 + scale * c["mass_pct"], size=self._mass0.shape)
        # COM jitter on the base/payload (meters)
        jit = scale * c["com_jitter_m"]
        self.m.body_ipos[:] = self._com0 + rng.uniform(-jit, jit, size=self._com0.shape)
        # geom friction (tangential mostly) — scale around nominal
        f = self._fric0.copy()
        f[:, 0] = self._fric0[:, 0] * rng.uniform(
            1 - scale * c["friction_pct"], 1 + scale * c["friction_pct"], size=f.shape[0])
        self.m.geom_friction[:] = f
        # joint damping
        self.m.dof_damping[:] = self._damp0 * rng.uniform(
            1 - scale * c["damping_pct"], 1 + scale * c["damping_pct"], size=self._damp0.shape)
        # actuator strength (gain + matching bias for position servos)
        g = rng.uniform(1 - scale * c["gain_pct"], 1 + scale * c["gain_pct"])
        self.m.actuator_gainprm[:] = self._gain0 * g
        self.m.actuator_biasprm[:] = self._bias0 * g
        # gravity tilt (a few degrees) to mimic an uneven floor
        tilt = scale * c["gravity_tilt_rad"]
        ang = rng.uniform(-tilt, tilt, size=2)
        self.m.opt.gravity[:] = [9.81 * np.sin(ang[0]), 9.81 * np.sin(ang[1]),
                                 -9.81 * np.cos(ang[0]) * np.cos(ang[1])]


class ActionLatency:
    """Fixed-delay action buffer (control latency / actuator lag)."""

    def __init__(self, delay_steps: int):
        self.delay = max(0, int(delay_steps))
        self.buf = deque(maxlen=self.delay + 1)

    def reset(self, action_dim: int):
        self.buf.clear()
        for _ in range(self.delay + 1):
            self.buf.append(np.zeros(action_dim, np.float32))

    def __call__(self, action):
        self.buf.append(np.asarray(action, np.float32))
        return self.buf[0]


class PushPerturber:
    """Random impulsive shoves to the torso, to train push recovery."""

    def __init__(self, body_id, cfg: dict):
        self.body_id = body_id
        self.cfg = cfg
        self._next = 0
        self._until = -1
        self._force = np.zeros(3)

    def reset(self, rng, control_hz):
        self._next = int(rng.uniform(*self.cfg["interval_s"]) * control_hz)
        self._until = -1

    def apply(self, data, step, rng, control_hz, scale=1.0):
        data.xfrc_applied[self.body_id, :3] = 0.0
        if step == self._next:
            mag = scale * rng.uniform(*self.cfg["force_n"])
            ang = rng.uniform(0, 2 * np.pi)
            self._force = np.array([mag * np.cos(ang), mag * np.sin(ang), 0.0])
            self._until = step + int(self.cfg["duration_s"] * control_hz)
            self._next = step + int(rng.uniform(*self.cfg["interval_s"]) * control_hz)
        if step <= self._until:
            data.xfrc_applied[self.body_id, :3] = self._force


# Default DR config (ramped over training); ranges grounded in the real robot.
DR_DEFAULT = dict(
    mass_pct=0.20,          # ±20% per-body mass
    com_jitter_m=0.01,      # ±1 cm COM
    friction_pct=0.40,      # ±40% friction
    damping_pct=0.50,
    gain_pct=0.20,          # ±20% actuator strength
    gravity_tilt_rad=0.05,  # ±2.9° floor tilt
)
PUSH_DEFAULT = dict(force_n=(30.0, 100.0), duration_s=0.1, interval_s=(2.0, 5.0))
