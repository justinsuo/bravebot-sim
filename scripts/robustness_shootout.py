#!/usr/bin/env python3
"""
Robustness shootout: compare two PPO policies under FULL domain randomization +
pushes (the conditions a sim-to-real policy must survive). The keep-best eval
runs on a clean (no-DR) env, so it cannot see robustness — this can. Reports, per
policy, fall rate / mean survival steps / tracking error over many randomized
episodes with fixed seeds (same seeds for both, for a paired comparison).

    python scripts/robustness_shootout.py policy_champion policy_drhardened
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "bravebot_sim", "rl")

import numpy as np
from stable_baselines3 import PPO
from bravebot_sim.rl.env import BraveBotLocomotionEnv

EVAL_CMDS = [[0.5, 0, 0], [-0.4, 0, 0], [0, 0, 0.5], [0, 0, -0.5], [0.4, 0, 0.3], [0, 0, 0]]


def evaluate(tag, episodes=60, seed0=7000):
    model = PPO.load(os.path.join(OUT, tag), device="cpu")
    env = BraveBotLocomotionEnv(randomize=True)         # full DR + pushes + latency + noise
    env._global = 10_000_000                            # force DR ramp fully ON
    falls = 0; steps_sum = 0; terr_sum = 0.0; n_track = 0
    for i in range(episodes):
        cmd = EVAL_CMDS[i % len(EVAL_CMDS)]
        obs, _ = env.reset(seed=seed0 + i)
        env._cmd[:] = cmd
        steps = 0
        for _ in range(env.max_steps):
            a = model.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, _ = env.step(a)
            steps += 1
            terr_sum += abs(float(env._sensor("base_vel")[0]) - cmd[0]); n_track += 1
            if term:
                falls += 1; break
            if trunc:
                break
        steps_sum += steps
    return dict(tag=tag, episodes=episodes, fall_rate=falls / episodes,
                mean_steps=steps_sum / episodes, track_err=terr_sum / max(1, n_track))


if __name__ == "__main__":
    tags = sys.argv[1:] or ["policy_champion", "policy_drhardened"]
    rows = [evaluate(t) for t in tags]
    print(f"\n{'policy':<22}{'falls%':>8}{'mean_steps':>12}{'track_err':>11}  (DR+push, 60 eps, paired seeds)")
    for r in rows:
        print(f"{r['tag']:<22}{r['fall_rate']*100:>7.1f}%{r['mean_steps']:>12.1f}{r['track_err']:>11.3f}")
    if len(rows) == 2:
        a, b = rows
        more_robust = b['mean_steps'] > a['mean_steps'] and b['fall_rate'] <= a['fall_rate']
        print(f"\n{b['tag']} more robust than {a['tag']}? {more_robust} "
              f"(survival {b['mean_steps']:.0f} vs {a['mean_steps']:.0f}, "
              f"falls {b['fall_rate']*100:.0f}% vs {a['fall_rate']*100:.0f}%)")
