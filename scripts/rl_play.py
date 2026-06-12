#!/usr/bin/env python3
"""
Run a trained BraveBot locomotion policy in the sim and render it.

    python scripts/rl_play.py --cmd 0.5 0 0          # walk forward
    python scripts/rl_play.py --cmd 0 0 0.5 --video renders/rl.mp4

Loads bravebot_sim/rl/policy.zip (SB3) — or pass --onnx to use policy.onnx.
A lightly-trained policy will be wobbly; this is the tool to watch whatever the
training has produced (train longer on a GPU for real walking).
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

from bravebot_sim.rl.env import BraveBotLocomotionEnv  # noqa: E402

RL_DIR = os.path.join(ROOT, "bravebot_sim", "rl")


def load_policy(use_onnx: bool):
    if use_onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(os.path.join(RL_DIR, "policy.onnx"))
        return lambda obs: sess.run(None, {"obs": obs[None].astype(np.float32)})[0][0]
    from stable_baselines3 import PPO
    model = PPO.load(os.path.join(RL_DIR, "policy"), device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0],
                    metavar=("VX", "VY", "YAW"))
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    policy = load_policy(args.onnx)
    env = BraveBotLocomotionEnv(episode_s=args.seconds)
    obs, _ = env.reset(seed=0)
    env._cmd[:] = args.cmd     # fixed command

    renderer = cam = None
    frames = []
    if args.video:
        import mujoco
        renderer = mujoco.Renderer(env.model, height=720, width=1100)
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(env.model, cam)
        cam.distance, cam.elevation, cam.azimuth = 3.2, -14, 130

    vx_track, n, fell = 0.0, 0, False
    steps = int(args.seconds * 50)
    for k in range(steps):
        a = policy(obs)
        obs, r, term, trunc, _ = env.step(a)
        vx_track += abs(env._sensor("base_vel")[0] - args.cmd[0]); n += 1
        if renderer is not None and k % 2 == 0:
            cam.lookat[:] = [env.data.qpos[0], env.data.qpos[1], 0.5]
            renderer.update_scene(env.data, cam)
            frames.append(renderer.render())
        if term:
            fell = True; break

    print(f"survived {k+1}/{steps} control steps ({(k+1)/50:.1f}s), "
          f"mean |vx err|={vx_track/max(1,n):.2f} m/s, fell={fell}")
    if frames:
        import imageio.v3 as iio
        os.makedirs(os.path.dirname(args.video), exist_ok=True)
        iio.imwrite(args.video, frames, fps=25, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
        print("wrote", args.video)


if __name__ == "__main__":
    main()
