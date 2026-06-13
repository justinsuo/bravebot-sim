#!/usr/bin/env python3
"""
Watch the trained RL locomotion policy live (MuJoCo passive viewer).

    mjpython scripts/rl_view.py                 # load latest policy.zip
    mjpython scripts/rl_view.py --onnx          # use policy.onnx

Arrow keys set the command the policy tracks:
    Up / Down    vx  (forward / back)      Left / Right   yaw rate
    Enter        stop (zero command)
Resets automatically if it falls. (macOS needs `mjpython`.)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from bravebot_sim.rl.env import BraveBotLocomotionEnv  # noqa: E402

RL_DIR = os.path.join(ROOT, "bravebot_sim", "rl")
UP, DOWN, LEFT, RIGHT, ENTER = 265, 264, 263, 262, 257


def load_policy(use_onnx):
    if use_onnx:
        import onnxruntime as ort
        s = ort.InferenceSession(os.path.join(RL_DIR, "policy.onnx"))
        return lambda o: s.run(None, {"obs": o[None].astype(np.float32)})[0][0]
    from stable_baselines3 import PPO
    m = PPO.load(os.path.join(RL_DIR, "policy"), device="cpu")
    return lambda o: m.predict(o, deterministic=True)[0]


def main():
    import mujoco.viewer
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", action="store_true")
    args = ap.parse_args()

    policy = load_policy(args.onnx)
    env = BraveBotLocomotionEnv(episode_s=1e9, randomize=False)
    obs, _ = env.reset(seed=0)
    cmd = np.zeros(3, np.float32)
    env._cmd[:] = cmd

    def on_key(k):
        if k == UP:    cmd[0] = min(0.8, cmd[0] + 0.1)
        elif k == DOWN: cmd[0] = max(-0.5, cmd[0] - 0.1)
        elif k == LEFT: cmd[2] = min(0.6, cmd[2] + 0.1)
        elif k == RIGHT: cmd[2] = max(-0.6, cmd[2] - 0.1)
        elif k == ENTER: cmd[:] = 0.0
        env._cmd[:] = cmd
        print(f"cmd vx={cmd[0]:+.1f} vy={cmd[1]:+.1f} yaw={cmd[2]:+.1f}")

    print(__doc__)
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as v:
        while v.is_running():
            t0 = time.time()
            a = policy(obs)
            obs, _, term, trunc, _ = env.step(a)
            if term:
                obs, _ = env.reset()
                env._cmd[:] = cmd
            v.sync()
            time.sleep(max(0.0, 0.02 - (time.time() - t0)))


if __name__ == "__main__":
    main()
