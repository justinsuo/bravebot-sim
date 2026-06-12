#!/usr/bin/env python3
"""
Headless patrol demo: drive the BraveBot down a data-center aisle, scan with the
four-sensor stack, fuse readings into risk-scored alerts, and render the run to
PNG frames + a contact sheet. No window is opened (offscreen rendering).

Usage:
    python scripts/patrol_demo.py [--seconds 26] [--no-render]
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
from bravebot_sim import BraveBot, PatrolController, diagnose, scene_path, facility  # noqa: E402

RENDER_DIR = os.path.join(ROOT, "renders", "patrol")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=26.0)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    facility.build_scene_xml()
    bot = BraveBot(scene_path())
    bot.x, bot.y = facility.DOCK
    bot.apply()
    ctrl = PatrolController(bot, facility.WAYPOINTS)

    renderer = cam = None
    if not args.no_render:
        import imageio.v3 as iio  # noqa: F401
        os.makedirs(RENDER_DIR, exist_ok=True)
        renderer = mujoco.Renderer(bot.model, height=720, width=1100)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(bot.model, cam)

    n = int(args.seconds / args.dt)
    seen: dict[str, float] = {}      # zone -> best confidence over the run
    alerted: set[str] = set()        # zones we've already raised an alert for
    frames = []
    shot_every = max(1, n // 6)

    for k in range(n):
        ctrl.update()
        bot.step(args.dt)
        for r in bot.scan(facility.ANOMALIES):
            seen[r.target] = max(seen.get(r.target, 0.0), r.confidence)
            if r.confidence > 0.45 and r.target not in alerted:
                alerted.add(r.target)
                alert = diagnose([r])
                print(f"\n[t={k*args.dt:5.1f}s]  detected via {r.modality} array")
                print("   " + alert.render().replace("\n", "\n   "))
        if renderer is not None and (k % shot_every == 0 or k == n - 1):
            import imageio.v3 as iio
            cam.lookat[:] = [bot.x, bot.y, 0.55]
            cam.distance, cam.elevation, cam.azimuth = 5.6, -14, 22
            renderer.update_scene(bot.data, cam)
            path = os.path.join(RENDER_DIR, f"frame_{len(frames):02d}.png")
            iio.imwrite(path, renderer.render())
            frames.append(path)
        if ctrl.done:
            print(f"\npatrol complete at t={k*args.dt:.1f}s "
                  f"({len(seen)} anomalies localized)")
            break

    print(f"\nAnomalies detected: {len(seen)}/{len(facility.ANOMALIES)}")
    for zone, conf in sorted(seen.items(), key=lambda x: -x[1]):
        print(f"  {conf:5.0%}  {zone}")

    if frames:
        _contact_sheet(frames, os.path.join(ROOT, "renders", "patrol_contact.png"))
        print(f"\nwrote {len(frames)} frames + renders/patrol_contact.png")


def _contact_sheet(frames, out):
    import imageio.v3 as iio
    imgs = [iio.imread(f) for f in frames]
    h, w = imgs[0].shape[:2]
    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    sheet = np.full((rows * h, cols * w, 3), 16, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet[r*h:(r+1)*h, c*w:(c+1)*w] = im[..., :3]
    iio.imwrite(out, sheet)


if __name__ == "__main__":
    main()
