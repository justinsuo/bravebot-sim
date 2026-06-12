#!/usr/bin/env python3
"""
Train a BraveBot locomotion policy with PPO (the real path to robust walking +
turning). Exports the trained policy to ONNX so it can be loaded back into the
sim — the same drop-in pattern as the TRON 1 Isaac Gym policies.

    python scripts/rl_train.py --steps 2_000_000 --envs 8     # real run (GPU box)
    python scripts/rl_train.py --steps 50_000 --envs 4        # quick CPU sanity run

Outputs:  bravebot_sim/rl/policy.zip (SB3) and bravebot_sim/rl/policy.onnx
Full convergence needs millions of steps (use a GPU). A short CPU run is enough
to confirm the env + reward are learnable (mean episode reward should rise).
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "bravebot_sim", "rl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from bravebot_sim.rl.env import BraveBotLocomotionEnv

    def thunk():
        return Monitor(BraveBotLocomotionEnv())

    VecEnv = DummyVecEnv if args.envs == 1 else SubprocVecEnv
    venv = VecEnv([thunk for _ in range(args.envs)])

    model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed,
                n_steps=2048, batch_size=512, gae_lambda=0.95, gamma=0.99,
                learning_rate=3e-4, ent_coef=0.0, clip_range=0.2,
                policy_kwargs=dict(net_arch=[256, 256]),
                device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"training {args.steps} steps on {model.device}, {args.envs} envs ...")
    model.learn(total_timesteps=args.steps, progress_bar=False)

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save(os.path.join(OUT_DIR, "policy"))
    print("saved", os.path.join(OUT_DIR, "policy.zip"))
    export_onnx(model, os.path.join(OUT_DIR, "policy.onnx"))


def export_onnx(model, path):
    """Export the deterministic actor (obs -> action) to ONNX."""
    import torch

    class Actor(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            # deterministic mean action from the SB3 policy
            features = self.policy.extract_features(obs)
            latent_pi = self.policy.mlp_extractor.forward_actor(features)
            return self.policy.action_net(latent_pi)

    obs_dim = model.observation_space.shape[0]
    actor = Actor(model.policy).to("cpu").eval()
    dummy = torch.zeros(1, obs_dim)
    # dynamo=False uses the legacy TorchScript exporter (no onnxscript dependency)
    torch.onnx.export(actor, dummy, path, input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
                      opset_version=17, dynamo=False)
    print("exported", path)


if __name__ == "__main__":
    main()
