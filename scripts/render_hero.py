#!/usr/bin/env python3
"""
Render hero stills of the BraveBot:  a studio shot of the robot, and an in-aisle
patrol shot in the data-center scene. Offscreen — opens no window.

Usage:  python scripts/render_hero.py
"""

from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import mujoco  # noqa: E402
import imageio.v3 as iio  # noqa: E402
from bravebot_sim import BraveBot, model_path, scene_path, facility  # noqa: E402

OUT = os.path.join(ROOT, "renders")


def shot(model, data, name, *, dist, elev, azim, look, w=1280, h=960):
    r = mujoco.Renderer(model, height=h, width=w)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance, cam.elevation, cam.azimuth = dist, elev, azim
    cam.lookat[:] = look
    r.update_scene(data, cam)
    path = os.path.join(OUT, name)
    iio.imwrite(path, r.render())
    print("wrote", os.path.relpath(path, ROOT))


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1 — studio shot of the bare robot
    bot = BraveBot(model_path())
    bot.apply()
    shot(bot.model, bot.data, "hero_studio.png",
         dist=2.4, elev=-12, azim=135, look=[0.02, 0, 0.6])

    # 2 — in-aisle patrol shot
    facility.build_scene_xml()
    bot = BraveBot(scene_path())
    bot.x, bot.y, bot.yaw = 3.0, 0.0, 0.0
    bot.set_stance("nominal")
    bot.apply()
    shot(bot.model, bot.data, "hero_patrol.png",
         dist=5.6, elev=-14, azim=22, look=[3.2, 0.0, 0.7])

    # 3 — low stance (wheel-legged variable geometry)
    bot.x, bot.y, bot.yaw = 3.0, 0.0, 0.0
    bot.set_stance("low")
    bot.apply()
    shot(bot.model, bot.data, "hero_low_stance.png",
         dist=5.2, elev=-12, azim=22, look=[3.0, 0.0, 0.5])


if __name__ == "__main__":
    main()
