#!/usr/bin/env python3
"""
Render a smooth MP4 of the BraveBot running its data-center patrol — the camera
tracks the robot down the aisle as it detects anomalies. Offscreen (no window).

Usage:  python scripts/render_patrol_video.py [--out renders/patrol.mp4] [--fps 30]
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import mujoco  # noqa: E402
import imageio.v3 as iio  # noqa: E402
from bravebot_sim import BraveBot, PatrolController, diagnose, scene_path, facility  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "renders", "patrol.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    args = ap.parse_args()

    facility.build_scene_xml()
    bot = BraveBot(scene_path())
    bot.x, bot.y = facility.DOCK
    bot.apply()
    ctrl = PatrolController(bot, facility.WAYPOINTS)

    renderer = mujoco.Renderer(bot.model, height=args.h, width=args.w)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(bot.model, cam)
    cam.distance, cam.elevation, cam.azimuth = 7.0, -34, 10
    look = [bot.x, bot.y, 0.6]

    dt = 0.02
    capture_every = max(1, int(round(1.0 / args.fps / dt)))
    frames = []
    alerted: set[str] = set()
    tail = 60  # extra frames after the patrol completes

    k = 0
    while True:
        ctrl.update()
        bot.step(dt)
        for r in bot.scan(facility.ANOMALIES):
            if r.confidence > 0.45 and r.target not in alerted:
                alerted.add(r.target)
                a = diagnose([r])
                print(f"[t={k*dt:4.1f}s] {a.zone} — {r.modality} {r.confidence:.0%}")
        if k % capture_every == 0:
            # smooth chase camera: lerp the look-at toward the robot
            look[0] += 0.15 * (bot.x + 0.4 - look[0])
            look[1] += 0.15 * (bot.y - look[1])
            cam.lookat[:] = look
            renderer.update_scene(bot.data, cam)
            frames.append(renderer.render())
        if ctrl.done:
            tail -= 1
            if tail <= 0:
                break
        k += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    iio.imwrite(args.out, frames, fps=args.fps, codec="libx264",
                output_params=["-pix_fmt", "yuv420p"])
    print(f"\nwrote {args.out}  ({len(frames)} frames, {len(frames)/args.fps:.1f}s, "
          f"{len(alerted)}/{len(facility.ANOMALIES)} anomalies)")


if __name__ == "__main__":
    main()
