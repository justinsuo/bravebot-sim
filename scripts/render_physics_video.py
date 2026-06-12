#!/usr/bin/env python3
"""
Render an MP4 of the BraveBot under REAL physics: it balances on its wheels,
drives, turns, and recovers from an external shove — all via the MuJoCo
rigid-body dynamics + the balance controller. Offscreen (no window).

Usage:  python scripts/render_physics_video.py [--out renders/physics.mp4]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import mujoco
import imageio.v3 as iio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from bravebot_sim import PhysicsBraveBot, physics_scene_path  # noqa: E402

# (t_start, v_cmd, omega_cmd, caption)
SCHEDULE = [
    (0.0, 0.0, 0.0, "Balancing on two wheels under gravity"),
    (2.0, 1.0, 0.0, "Driving forward (real torque control)"),
    (4.5, 1.0, 0.0, "External shove -> recover while moving"),  # shove at 4.5
    (7.0, 0.0, 0.0, "Station-keeping (no creep)"),
    (8.5, -1.0, 0.0, "Reversing"),
    (11.5, 0.0, 0.0, "Hold"),
]
SHOVE_T, SHOVE_N = 4.5, 90.0
DURATION = 13.5


def _cmd(t):
    v, w, cap = 0.0, 0.0, ""
    for ts, vv, ww, c in SCHEDULE:
        if t >= ts:
            v, w, cap = vv, ww, c
    return v, w, cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "renders", "physics.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    args = ap.parse_args()

    bot = PhysicsBraveBot(physics_scene_path())
    torso = mujoco.mj_name2id(bot.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    renderer = mujoco.Renderer(bot.model, height=args.h, width=args.w)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(bot.model, cam)
    cam.distance, cam.elevation, cam.azimuth = 4.6, -16, 30
    look = [bot.x, bot.y, 0.55]

    dt = bot.model.opt.timestep
    cap_every = max(1, int(round(1.0 / args.fps / dt)))
    frames = []
    n = int(DURATION / dt)
    max_pitch = 0.0

    for k in range(n):
        t = k * dt
        v, w, _ = _cmd(t)
        bot.ctrl.control(v, w)
        bot.data.xfrc_applied[:] = 0
        if SHOVE_T <= t < SHOVE_T + 0.12:        # external shove for 0.12 s
            bot.data.xfrc_applied[torso][0] = SHOVE_N
        mujoco.mj_step(bot.model, bot.data)
        s = bot.ctrl.read_state()
        max_pitch = max(max_pitch, abs(s.pitch))
        if s.fell:
            print(f"fell at t={t:.2f}")
            break
        if k % cap_every == 0:
            look[0] += 0.12 * (bot.x + 0.3 - look[0])
            look[1] += 0.12 * (bot.y - look[1])
            cam.lookat[:] = look
            renderer.update_scene(bot.data, cam)
            frames.append(renderer.render())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    iio.imwrite(args.out, frames, fps=args.fps, codec="libx264",
                output_params=["-pix_fmt", "yuv420p"])
    print(f"wrote {args.out}  ({len(frames)} frames, {len(frames)/args.fps:.1f}s, "
          f"max|pitch|={math.degrees(max_pitch):.1f}deg)")


if __name__ == "__main__":
    main()
