#!/usr/bin/env python3
"""
RL-policy-driven inspection patrol (physics).

The trained locomotion policy does the balancing + driving; a thin waypoint
navigator on top sets the velocity command (vx, vy, yaw) to walk the facility's
cold-aisle route, and the robot's onboard sensors scan for anomalies as it goes —
a full legged inspection round on the real rigid-body model, not the kinematic
stand-in.

The locomotion policy tracks commanded yaw *rate* (it has no absolute-heading
input), so it threads the aisle with some lateral drift rather than hugging the
centerline — area-inspection coverage, not survey-grade path tracking. The
navigator closes the heading loop externally and picks forward/reverse to avoid
an impossible 180-deg same-direction spin at the turnaround.

    python scripts/rl_patrol.py                 # headless: patrol + report (default)
    python scripts/rl_patrol.py --champion      # use policy_champion (default: policy)
    mjpython scripts/rl_patrol.py --view        # watch it live (macOS)
    python scripts/rl_patrol.py --check         # headless self-test (asserts)
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import mujoco  # noqa: E402
from bravebot_sim.rl.env import BraveBotLocomotionEnv  # noqa: E402
from bravebot_sim import facility  # noqa: E402
from bravebot_sim.sim import scan_anomalies  # noqa: E402

RL_DIR = os.path.join(ROOT, "bravebot_sim", "rl")
WP_RADIUS = 0.7            # m — "passed near" tolerance (the policy tracks yaw rate,
                           # not absolute heading, so it threads the aisle with some
                           # lateral drift; area-inspection coverage is the real goal)
V_MAX, V_REV = 0.5, -0.45  # forward / reverse command caps the policy tracks well


def load_policy(tag, use_onnx):
    if use_onnx:
        import onnxruntime as ort
        s = ort.InferenceSession(os.path.join(RL_DIR, f"{tag}.onnx"))
        return lambda o: s.run(None, {"obs": o[None].astype(np.float32)})[0][0]
    from stable_baselines3 import PPO
    m = PPO.load(os.path.join(RL_DIR, tag), device="cpu")
    return lambda o: m.predict(o, deterministic=True)[0]


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class WaypointNavigator:
    """Pure-pursuit that actively closes the HEADING loop (the policy tracks yaw
    *rate*, not absolute heading, so it drifts off-aisle without steering). At each
    step it picks forward OR reverse — whichever needs less turning — so the
    out-and-back route never needs an impossible 180 deg same-direction spin, and
    steers on absolute bearing error toward the next waypoint."""

    K_YAW = 1.5

    def __init__(self, waypoints):
        self.wp = list(waypoints)
        self.i = 0
        self.reached = 0
        self._reverse = False      # current drive mode (hysteresis vs chatter)

    @property
    def done(self):
        return self.i >= len(self.wp)

    def command(self, x, y, yaw):
        if self.done:
            return np.zeros(3, np.float32)
        tx, ty = self.wp[self.i]
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        if dist < WP_RADIUS:
            self.i += 1
            self.reached += 1
            return self.command(x, y, yaw)
        bearing = math.atan2(dy, dx)
        err_f = _wrap(bearing - yaw)            # heading error if driving forward
        err_r = _wrap(bearing - yaw - math.pi)  # ... if driving in reverse
        # pick forward/reverse by smaller turn, with hysteresis so it doesn't
        # chatter (and flip vx) when the target sits near +-90 deg abeam.
        margin = 0.35 if self._reverse else -0.35
        self._reverse = abs(err_r) + margin < abs(err_f)
        err = err_r if self._reverse else err_f
        yaw_cmd = float(np.clip(self.K_YAW * err, -0.4, 0.4))
        speed = max(0.2, math.cos(err)) * min(1.0, dist)   # ease off until aligned
        vx = (V_REV if self._reverse else V_MAX) * speed
        return np.array([float(np.clip(vx, V_REV, V_MAX)), 0.0, yaw_cmd], np.float32)


def _pose(env):
    p = env.data.xpos[env._base]
    R = env.data.xmat[env._base].reshape(3, 3)
    return float(p[0]), float(p[1]), math.atan2(float(R[1, 0]), float(R[0, 0]))


def _scan_frame_fn(env):
    def frame(sensor_id):
        sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, f"s_{sensor_id}")
        pos = env.data.site_xpos[sid].copy()
        fwd = env.data.site_xmat[sid].reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
        return pos, fwd / (np.linalg.norm(fwd) + 1e-9)
    return frame


def run_patrol(policy, route, on_step=None, max_s=120.0):
    """Drive the route with the policy + navigator; return a patrol report."""
    env = BraveBotLocomotionEnv(episode_s=1e9, randomize=False)
    obs, _ = env.reset(seed=0)
    nav = WaypointNavigator(route)
    frame = _scan_frame_fn(env)
    detected, fell, max_x = {}, False, -9.0
    steps = int(max_s * env.control_hz)
    for k in range(steps):
        x, y, yaw = _pose(env)
        max_x = max(max_x, x)
        env._cmd[:] = nav.command(x, y, yaw)
        obs, _, term, _, _ = env.step(policy(obs))
        for r in scan_anomalies(frame, facility.ANOMALIES):
            if r.confidence > 0.5 and r.target not in detected:
                detected[r.target] = (r.modality, round(r.confidence, 2))
        if on_step:
            on_step(env, nav)
        if term:
            fell = True
            break
        if nav.done:
            break
    return dict(reached=nav.reached, total=len(route), detected=detected,
                fell=fell, steps=k + 1, end_xy=_pose(env)[:2], max_x=max_x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", action="store_true", help="use policy_champion")
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--view", action="store_true", help="live MuJoCo viewer (mjpython)")
    ap.add_argument("--check", action="store_true", help="headless self-test")
    args = ap.parse_args()
    tag = "policy_champion" if args.champion else "policy"
    policy = load_policy(tag, args.onnx)
    route = [*facility.WAYPOINTS, facility.DOCK]   # start forward into the aisle, end at dock

    if args.view:
        import time
        import mujoco.viewer
        env = BraveBotLocomotionEnv(episode_s=1e9, randomize=False)
        obs, _ = env.reset(seed=0)
        nav = WaypointNavigator(route)
        print("RL inspection patrol — watch the legged robot walk the aisle.")
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            while v.is_running():
                t0 = time.time()
                x, y, yaw = _pose(env)
                env._cmd[:] = nav.command(x, y, yaw)
                obs, _, term, _, _ = env.step(policy(obs))
                if term or nav.done:
                    obs, _ = env.reset(); nav = WaypointNavigator(route)
                v.sync()
                time.sleep(max(0.0, 0.02 - (time.time() - t0)))
        return

    rep = run_patrol(policy, route)
    print(f"\nRL inspection patrol ({tag}):")
    print(f"  upright           : {not rep['fell']}  ({rep['steps']} steps)")
    print(f"  aisle traversal   : reached x={rep['max_x']:.1f} m (far racks ~6.5), "
          f"{rep['reached']}/{rep['total']} waypoints passed")
    print(f"  anomalies detected: {len(rep['detected'])}/{len(facility.ANOMALIES)}")
    for name, (mod, conf) in rep["detected"].items():
        print(f"      [{mod:>8}] {name}  (conf {conf})")

    if args.check:
        # inspection coverage is the goal (the policy tracks yaw rate not heading,
        # so it threads the aisle with some lateral drift — see module docstring).
        assert not rep["fell"], "patrol fell"
        assert rep["max_x"] >= 3.5, f"did not patrol into the aisle (max_x={rep['max_x']:.1f})"
        assert len(rep["detected"]) >= 4, f"only detected {len(rep['detected'])}/5 anomalies"
        print("  CHECK OK")


if __name__ == "__main__":
    main()
