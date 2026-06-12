#!/usr/bin/env python3
"""
Render an MP4 of the BraveBot's balance-assisted legged stepping gait: marching
in place, stepping forward / backward, and a step-turn — legs visibly striding
while the wheel balance keeps it upright. Offscreen.

Usage:  python scripts/render_gait_video.py
"""
from __future__ import annotations
import argparse, math, os, sys

import mujoco
import imageio.v3 as iio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from bravebot_sim import PhysicsBraveBot, physics_model_path  # noqa: E402

# (t_start, v, omega, caption)
SCHEDULE = [
    (0.0, 0.0, 0.0, "Balance-assisted legged gait"),
    (2.0, 0.0, 0.0, "Marching in place"),
    (3.5, 0.4, 0.0, "Stepping forward"),
    (7.0, 0.0, 0.0, "Hold"),
    (8.0, -0.4, 0.0, "Stepping backward"),
    (11.0, 0.0, 0.35, "Step-turn"),
]
DURATION = 14.5


def cmd(t):
    v = w = 0.0
    for ts, vv, ww, _ in SCHEDULE:
        if t >= ts:
            v, w = vv, ww
    return v, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "renders", "gait.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    bot = PhysicsBraveBot(physics_model_path())
    bot.set_walking(True)
    r = mujoco.Renderer(bot.model, height=720, width=1100)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(bot.model, cam)
    cam.distance, cam.elevation, cam.azimuth = 3.2, -14, 120
    look = [bot.x, bot.y, 0.5]

    dt = bot.model.opt.timestep
    cap_every = max(1, int(round(1.0 / args.fps / dt)))
    frames = []
    for k in range(int(DURATION / dt)):
        t = k * dt
        v, w = cmd(t)
        bot.drive(v, w)
        # step one model substep with the gait overlay
        bot.ctrl.control(bot._v, bot._w)
        moving = abs(bot._v) > 0.05 or abs(bot._w) > 0.05
        bot.gait.advance(dt, moving)
        tg = bot.gait.leg_targets()
        for j, aid in bot.ctrl._leg_act.items():
            bot.data.ctrl[aid] = tg[j]
        mujoco.mj_step(bot.model, bot.data)
        if bot.ctrl.read_state().fell:
            print(f"fell t={t:.1f}"); break
        if k % cap_every == 0:
            look[0] += 0.1 * (bot.x - look[0]); look[1] += 0.1 * (bot.y - look[1])
            cam.lookat[:] = look
            r.update_scene(bot.data, cam)
            frames.append(r.render())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    iio.imwrite(args.out, frames, fps=args.fps, codec="libx264",
                output_params=["-pix_fmt", "yuv420p"])
    print(f"wrote {args.out} ({len(frames)} frames, {len(frames)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
