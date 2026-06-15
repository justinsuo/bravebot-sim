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
BEST = os.path.join(OUT_DIR, "policy_best")
BEST_ONNX = os.path.join(OUT_DIR, "policy_best.onnx")
CSV = os.path.join(OUT_DIR, "progress.csv")
EVAL_CMDS = [[0.5, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.0, 0.0, 0.5],
             [0.0, 0.0, -0.5], [0.4, 0.0, 0.3], [0.0, 0.0, 0.0]]


def eval_score(model, env, steps=300):
    """Deterministic eval over fixed commands -> scalar (upright + tracking).

    Higher is better. Used to keep the BEST checkpoint by evaluation, not the last
    (PPO plateaus and drifts — the final policy is often not the strongest)."""
    import numpy as np
    total = 0.0
    for cmd in EVAL_CMDS:
        obs, _ = env.reset(seed=12345)
        env._cmd[:] = cmd
        up, terr = 0, 0.0
        for _ in range(steps):
            a = model.predict(obs, deterministic=True)[0]
            obs, _, term, _, _ = env.step(a)
            if term:
                break
            up += 1
            terr += abs(float(env._sensor("base_vel")[0]) - cmd[0])
        total += up - 0.6 * terr        # reward staying upright, penalize tracking error
    return total


def combined_eval(model, clean_env, dr_env):
    """Robustness-aware keep-best metric: clean tracking + tracking under FULL domain
    randomization + pushes. The clean-only score can't see robustness (it plateaued
    while DR-hardened policies kept improving disturbance-tracking), so the deployment-
    relevant metric blends both. Higher is better."""
    return eval_score(model, clean_env) + eval_score(model, dr_env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=500_000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heading", action="store_true",
                    help="train the heading-aware policy on its own track (42-d obs)")
    args = ap.parse_args()

    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback

    # heading-aware track ships to separate files so the 40-d champion is untouched.
    if args.heading:
        from bravebot_sim.rl.heading_env import HeadingAwareEnv as EnvCls
        ZIP = os.path.join(OUT_DIR, "policy_heading")
        ONNX, BEST = ZIP + ".onnx", os.path.join(OUT_DIR, "policy_heading_best")
        BEST_ONNX, CSV = BEST + ".onnx", os.path.join(OUT_DIR, "progress_heading.csv")
    else:
        from bravebot_sim.rl.env import BraveBotLocomotionEnv as EnvCls
        ZIP = os.path.join(OUT_DIR, "policy")
        ONNX, BEST = ZIP + ".onnx", os.path.join(OUT_DIR, "policy_best")
        BEST_ONNX, CSV = BEST + ".onnx", os.path.join(OUT_DIR, "progress.csv")

    os.makedirs(OUT_DIR, exist_ok=True)

    resume = args.resume and os.path.exists(ZIP + ".zip")

    # CONTINUE the curriculum across --resume: each env's per-env step counter must
    # pick up where the prior run left off, else DR / pushes / command range restart
    # at 0 for ~4.5M steps and de-randomize a hardened policy. We bake the offset
    # into the env constructor (inside thunk) so it reaches the real env in every
    # SubprocVecEnv worker — set_attr on the venv only hits the Monitor wrapper,
    # whose attribute the inner BraveBotLocomotionEnv never reads.
    global_offset = 0
    if resume:
        print("resuming from", ZIP + ".zip")
        model = PPO.load(ZIP, device="cpu")            # load weights first (no env)
        global_offset = model.num_timesteps // args.envs
        print(f"  curriculum resumed at _global={global_offset:,}/env "
              f"({model.num_timesteps:,} total) — DR/curriculum continue")

    def thunk():
        return Monitor(EnvCls(global_offset=global_offset))

    VecEnv = DummyVecEnv if args.envs == 1 else SubprocVecEnv
    venv = VecEnv([thunk for _ in range(args.envs)])

    if resume:
        # reload WITH the venv so the rollout buffer resizes to args.envs — lets a
        # fine-tune resume with a different --envs than the saved run (set_env alone
        # asserts equal n_envs and refuses).
        model = PPO.load(ZIP, env=venv, device="cpu")
    else:
        model = PPO("MlpPolicy", venv, verbose=0, seed=args.seed,
                    n_steps=1024, batch_size=8192, n_epochs=5, gae_lambda=0.95,
                    gamma=0.99, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2,
                    policy_kwargs=dict(net_arch=[512, 256, 128]), device="cpu")

    eval_env = EnvCls(randomize=False)   # clean deterministic eval
    dr_eval_env = EnvCls(randomize=True)  # robustness eval (DR + pushes)
    dr_eval_env._global = 10_000_000                     # force DR ramp fully ON
    keepbest = lambda m: combined_eval(m, eval_env, dr_eval_env)

    class Saver(BaseCallback):
        def __init__(self, every):
            super().__init__()
            self.every = every
            self.next = every
            self.best = -1e18
            if not args.resume or not os.path.exists(CSV):
                with open(CSV, "w", newline="") as f:
                    csv.writer(f).writerow(["timesteps", "ep_rew_mean", "ep_len_mean", "eval", "best"])

        def _on_training_start(self):
            # On --resume num_timesteps starts at e.g. 25M; anchor the next save to
            # the upcoming cadence boundary so we don't fire ~50 back-to-back saves
            # (each a full eval + ONNX export) while `next` crawls up from `every`.
            self.next = ((self.model.num_timesteps // self.every) + 1) * self.every

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
                # keep-best by robustness-aware evaluation (not by last checkpoint)
                score = keepbest(model)
                if score > self.best:
                    self.best = score
                    model.save(BEST)
                    try:
                        export_onnx(model, BEST_ONNX)
                    except Exception:
                        pass
                star = " *BEST*" if score >= self.best else ""
                with open(CSV, "a", newline="") as f:
                    csv.writer(f).writerow([self.num_timesteps, round(rew, 2), round(ln, 1),
                                            round(score, 1), round(self.best, 1)])
                print(f"[{self.num_timesteps:>10,}]  ep_rew={rew:7.1f}  ep_len={ln:6.1f}  "
                      f"eval={score:7.1f}  best={self.best:7.1f}{star}")
            return True

    print(f"training {args.steps:,} steps, {args.envs} envs, device={model.device}")
    model.learn(total_timesteps=args.steps, callback=Saver(args.save_every),
                reset_num_timesteps=not args.resume, progress_bar=False)
    # ship the BEST-by-eval policy (final is often not the strongest)
    final_score = keepbest(model)
    if os.path.exists(BEST + ".zip"):
        from stable_baselines3 import PPO as _PPO
        best_model = _PPO.load(BEST, device="cpu")
        best_score = keepbest(best_model)
        winner, label = (model, "final") if final_score >= best_score else (best_model, "best-checkpoint")
        print(f"shipping {label}: final={final_score:.1f} vs best={best_score:.1f}")
        winner.save(ZIP)
        export_onnx(winner, ONNX)
    else:
        model.save(ZIP)
        export_onnx(model, ONNX)
    print("done. shipped", ZIP + ".zip", "and", ONNX)


def export_onnx(model, path):
    """Export the deterministic actor (obs -> action) to ONNX.

    The action is clamped to [-1, 1] so the ONNX matches SB3's predict() (which
    clips to the action space) and is a SELF-CONTAINED deployable artifact — without
    this the raw policy mean can be out-of-range (±4+) on out-of-distribution states,
    and the consumer on the real robot would have to know to clip. (The sim env
    clips in step(), so in-sim behavior is unchanged either way.)"""
    import torch

    class Actor(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            features = self.policy.extract_features(obs)
            latent_pi = self.policy.mlp_extractor.forward_actor(features)
            return torch.clamp(self.policy.action_net(latent_pi), -1.0, 1.0)

    obs_dim = model.observation_space.shape[0]
    actor = Actor(model.policy).to("cpu").eval()
    dummy = torch.zeros(1, obs_dim)
    torch.onnx.export(actor, dummy, path, input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
                      opset_version=17, dynamo=False)


if __name__ == "__main__":
    main()
