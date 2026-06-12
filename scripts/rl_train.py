#!/usr/bin/env python3
"""
Train a BraveBot locomotion policy with PPO (the real path to robust walking +
turning). Checkpoints continuously: saves policy.zip + policy.onnx and logs
mean episode reward to rl/progress.csv every `--save-every` steps, so you can
watch it improve and load the latest policy at any time.

    python scripts/rl_train.py --steps 20_000_000 --envs 16     # full run (M-series / GPU)
    python scripts/rl_train.py --steps 60_000 --envs 6          # quick sanity run
    python scripts/rl_train.py --resume --steps 10_000_000      # continue from policy.zip

Outputs (bravebot_sim/rl/):  policy.zip (SB3), policy.onnx, progress.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "bravebot_sim", "rl")
ZIP = os.path.join(OUT_DIR, "policy")
ONNX = os.path.join(OUT_DIR, "policy.onnx")
CSV = os.path.join(OUT_DIR, "progress.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=500_000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from bravebot_sim.rl.env import BraveBotLocomotionEnv

    os.makedirs(OUT_DIR, exist_ok=True)

    def thunk():
        return Monitor(BraveBotLocomotionEnv())

    VecEnv = DummyVecEnv if args.envs == 1 else SubprocVecEnv
    venv = VecEnv([thunk for _ in range(args.envs)])

    if args.resume and os.path.exists(ZIP + ".zip"):
        print("resuming from", ZIP + ".zip")
        model = PPO.load(ZIP, env=venv, device="cpu")
    else:
        model = PPO("MlpPolicy", venv, verbose=0, seed=args.seed,
                    n_steps=1024, batch_size=4096, n_epochs=5, gae_lambda=0.95,
                    gamma=0.99, learning_rate=3e-4, ent_coef=0.0, clip_range=0.2,
                    policy_kwargs=dict(net_arch=[256, 256]), device="cpu")

    class Saver(BaseCallback):
        def __init__(self, every):
            super().__init__()
            self.every = every
            self.next = every
            if not args.resume or not os.path.exists(CSV):
                with open(CSV, "w", newline="") as f:
                    csv.writer(f).writerow(["timesteps", "ep_rew_mean", "ep_len_mean"])

        def _on_step(self):
            if self.num_timesteps >= self.next:
                self.next += self.every
                buf = self.model.ep_info_buffer
                rew = float(np.mean([e["r"] for e in buf])) if buf else float("nan")
                ln = float(np.mean([e["l"] for e in buf])) if buf else float("nan")
                model.save(ZIP)
                try:
                    export_onnx(model, ONNX)
                except Exception as e:
                    print("onnx export skipped:", e)
                with open(CSV, "a", newline="") as f:
                    csv.writer(f).writerow([self.num_timesteps, round(rew, 2), round(ln, 1)])
                print(f"[{self.num_timesteps:>10,}]  ep_rew_mean={rew:7.2f}  ep_len={ln:6.1f}  (saved)")
            return True

    print(f"training {args.steps:,} steps, {args.envs} envs, device={model.device}")
    model.learn(total_timesteps=args.steps, callback=Saver(args.save_every),
                reset_num_timesteps=not args.resume, progress_bar=False)
    model.save(ZIP)
    export_onnx(model, ONNX)
    print("done. saved", ZIP + ".zip", "and", ONNX)


def export_onnx(model, path):
    """Export the deterministic actor (obs -> action) to ONNX."""
    import torch

    class Actor(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            features = self.policy.extract_features(obs)
            latent_pi = self.policy.mlp_extractor.forward_actor(features)
            return self.policy.action_net(latent_pi)

    obs_dim = model.observation_space.shape[0]
    actor = Actor(model.policy).to("cpu").eval()
    dummy = torch.zeros(1, obs_dim)
    torch.onnx.export(actor, dummy, path, input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
                      opset_version=17, dynamo=False)


if __name__ == "__main__":
    main()
