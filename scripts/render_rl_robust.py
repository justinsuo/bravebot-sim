#!/usr/bin/env python3
"""
Render the RL champion's PUSH RECOVERY (the headline of the robustness work): the
policy drives forward and absorbs strong lateral/longitudinal shoves to the torso,
staying upright — what the domain-randomization + robustness-aware training bought.

    python scripts/render_rl_robust.py [--out renders/rl_robust.mp4] [--shove 120]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import imageio.v3 as iio

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import mujoco  # noqa: E402
from bravebot_sim.rl.env import BraveBotLocomotionEnv  # noqa: E402

RL_DIR = os.path.join(ROOT, "bravebot_sim", "rl")
# (time, Fx, Fy) shoves in Newtons — alternating directions, well above the
# 30-100 N domain-randomization training range to show real margin.
SHOVES = [(2.0, 0, 130), (5.0, 0, -130), (8.0, 150, 0), (11.0, 0, 150), (14.0, -150, 0)]
SHOVE_DUR = 0.12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "renders", "rl_robust.mp4"))
    ap.add_argument("--tag", default="policy_champion")
    ap.add_argument("--shove", type=float, default=1.0, help="scale on the shove forces")
    ap.add_argument("--secs", type=float, default=17.0)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(os.path.join(RL_DIR, args.tag), device="cpu")
    env = BraveBotLocomotionEnv(episode_s=1e9, randomize=False)
    obs, _ = env.reset(seed=0)
    env._cmd[:] = [0.35, 0.0, 0.0]            # gentle forward drive while being shoved

    renderer = mujoco.Renderer(env.model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.distance, cam.elevation, cam.azimuth = 3.8, -14, 50
    look = [0.0, 0.0, 0.6]
    cap_every = max(1, round(env.control_hz / args.fps))

    frames, shoves_survived, fell = [], 0, False
    for k in range(int(args.secs * env.control_hz)):
        t = k / env.control_hz
        # apply a shove window
        env.data.xfrc_applied[env._torso, :3] = 0.0
        for (ts, fx, fy) in SHOVES:
            if ts <= t < ts + SHOVE_DUR:
                env.data.xfrc_applied[env._torso, :2] = [fx * args.shove, fy * args.shove]
        a = model.predict(obs, deterministic=True)[0]
        obs, _, term, _, _ = env.step(a)
        if term:
            fell = True
            print(f"fell at t={t:.1f}s")
            break
        if k % cap_every == 0:
            p = env.data.xpos[env._base]
            look[0] += 0.1 * (float(p[0]) - look[0])
            look[1] += 0.1 * (float(p[1]) - look[1])
            cam.lookat[:] = look
            renderer.update_scene(env.data, cam)
            frames.append(renderer.render())
    shoves_survived = sum(1 for (ts, *_ ) in SHOVES if ts < (len(frames) * cap_every / env.control_hz)) - int(fell)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    iio.imwrite(args.out, frames, fps=args.fps, codec="libx264",
                output_params=["-pix_fmt", "yuv420p"])
    print(f"wrote {args.out}  ({len(frames)} frames, upright={not fell}, "
          f"{shoves_survived}/{len(SHOVES)} shoves survived)")


if __name__ == "__main__":
    main()
