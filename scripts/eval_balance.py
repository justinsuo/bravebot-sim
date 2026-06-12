#!/usr/bin/env python3
"""
Headless evaluation + auto-tuning of the BraveBot balance controller.

`evaluate(gains)` runs the physics model through a stand -> drive -> turn ->
drive command profile and returns metrics (upright time, max pitch, velocity
tracking RMSE, distance) plus a scalar cost. `tune()` does a random/coordinate
search around the seed gains and writes the best to bravebot_sim/gains.json.

    python scripts/eval_balance.py --eval          # score the current/seed gains
    python scripts/eval_balance.py --tune --iters 120
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from bravebot_sim.balance import BalanceController, Gains, settle_upright  # noqa: E402

PHYS_MODEL = os.path.join(ROOT, "description", "mjcf", "bravebot_physics.xml")
GAINS_PATH = os.path.join(ROOT, "bravebot_sim", "gains.json")

# (t_start, v_cmd, omega_cmd) command schedule — agile: fast steps, high speeds
SCHEDULE = [(0.0, 0.0, 0.0), (1.5, 1.5, 0.0), (4.0, 0.0, 0.0), (5.0, 1.5, 0.0),
            (7.5, -0.8, 0.0), (9.0, 0.0, 0.6), (11.0, 1.2, 0.0), (13.5, 0.0, 0.0)]
DURATION = 15.0
SETTLE_IGNORE = 0.35   # s after a command change to ignore (rewards fast rise)


def _cmd(t):
    v, w, ts0 = 0.0, 0.0, 0.0
    for ts, vv, ww in SCHEDULE:
        if t >= ts:
            v, w, ts0 = vv, ww, ts
    return v, w, t - ts0   # also return time-into-segment


def evaluate(gains: Gains, duration: float = DURATION, return_trace: bool = False):
    m = mujoco.MjModel.from_xml_path(PHYS_MODEL)
    d = mujoco.MjData(m)
    ctrl = BalanceController(m, d, gains)
    settle_upright(m, d, ctrl)

    dt = m.opt.timestep
    n = int(duration / dt)
    upright_until = duration
    pitch_sq = 0.0
    max_pitch = 0.0
    verr_sq = verr_n = 0
    werr_sq = werr_n = 0
    fell = False
    trace = []
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    for k in range(n):
        t = k * dt
        v_cmd, w_cmd, t_seg = _cmd(t)
        s = ctrl.control(v_cmd, w_cmd)
        # disturbance test: shove forward at 6.0 s and laterally at 12.0 s, so the
        # tuner must keep enough margin to reject pushes (not just go fast).
        d.xfrc_applied[:] = 0
        if 6.0 <= t < 6.12:
            d.xfrc_applied[torso][0] = 120.0
        elif 12.0 <= t < 12.12:
            d.xfrc_applied[torso][1] = 110.0
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            fell = True
            upright_until = t
            break
        pitch_sq += s.pitch ** 2
        max_pitch = max(max_pitch, abs(s.pitch))
        settled = t_seg > SETTLE_IGNORE   # only ignore a short rise window
        if settled and abs(v_cmd) > 0.1:
            verr_sq += (s.v - v_cmd) ** 2
            verr_n += 1
        if settled and abs(w_cmd) > 0.1:
            werr_sq += (s.yaw_rate - w_cmd) ** 2
            werr_n += 1
        if return_trace and k % 10 == 0:
            trace.append((t, s.pitch, s.v, s.yaw_rate))
        if s.fell:
            fell = True
            upright_until = t
            break

    pitch_rms = math.sqrt(pitch_sq / max(1, k))
    vel_rmse = math.sqrt(verr_sq / max(1, verr_n))
    yaw_rmse = math.sqrt(werr_sq / max(1, werr_n))
    dist = math.hypot(d.qpos[0], d.qpos[1])
    # responsive AND robust: reward velocity tracking, penalize falling and any
    # large pitch excursion (margin) beyond ~17 deg.
    pitch_excess = max(0.0, max_pitch - 0.30)
    cost = ((duration - upright_until) * 8.0 + vel_rmse * 5.0 + yaw_rmse * 2.0
            + pitch_rms * 1.0 + pitch_excess * 12.0)
    metrics = dict(cost=cost, upright_until=round(upright_until, 2), fell=fell,
                   pitch_rms=round(pitch_rms, 4), max_pitch=round(math.degrees(max_pitch), 1),
                   vel_rmse=round(vel_rmse, 4), yaw_rmse=round(yaw_rmse, 4),
                   distance=round(dist, 3))
    return (metrics, trace) if return_trace else metrics


def tune(iters: int = 120, seed: Gains | None = None):
    rng = np.random.default_rng(0)
    best = seed or _load_gains()
    best_v = best.as_vector()
    best_m = evaluate(best)
    print(f"seed gains {np.round(best_v,3)} -> {best_m}")
    lo = np.array([50, 5, 0.05, 0.12, 8.0])
    hi = np.array([1200, 150, 0.9, 0.32, 16.0])   # cap lean_max (margin) and k_yaw (spin)

    radius = 0.6
    for i in range(iters):
        # multiplicative jitter on a copy, occasionally re-seed widely
        cand = best_v * np.exp(rng.normal(0, radius, size=5))
        cand = np.clip(cand, lo, hi)
        m = evaluate(Gains.from_vector(cand))
        if m["cost"] < best_m["cost"]:
            best_v, best_m = cand, m
            print(f"  iter {i:3d}  NEW BEST cost={m['cost']:.2f}  "
                  f"upright={m['upright_until']}s vel_rmse={m['vel_rmse']} "
                  f"gains={np.round(cand,3)}")
        radius = max(0.12, radius * 0.985)   # anneal

    best = Gains.from_vector(best_v)
    _save_gains(best, best_m)
    print(f"\nBEST: {np.round(best_v,3)} -> {best_m}\nsaved -> {os.path.relpath(GAINS_PATH, ROOT)}")
    return best, best_m


def _load_gains() -> Gains:
    if os.path.exists(GAINS_PATH):
        with open(GAINS_PATH) as f:
            return Gains.from_vector(json.load(f)["gains"])
    return Gains()


def _save_gains(g: Gains, metrics: dict):
    with open(GAINS_PATH, "w") as f:
        json.dump({"gains": g.as_vector().tolist(),
                   "fields": ["k_p", "k_d", "k_pv", "lean_max", "k_yaw"],
                   "metrics": metrics}, f, indent=2)


def regression(min_upright: float = 13.5):
    """Assert the saved/default gains keep the robot upright — guards the sign fix."""
    m = evaluate(_load_gains())
    ok = (not m["fell"]) and m["upright_until"] >= min_upright
    print(("PASS" if ok else "FAIL") + f": upright {m['upright_until']}s "
          f"(>= {min_upright}), pitch_rms {m['pitch_rms']} -> {m}")
    if not ok:
        raise SystemExit("balance regression FAILED — check the tau_bal sign in balance.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--iters", type=int, default=120)
    args = ap.parse_args()
    if args.tune:
        tune(args.iters)
    elif args.regression:
        regression()
    else:
        print(json.dumps(evaluate(_load_gains()), indent=2))
