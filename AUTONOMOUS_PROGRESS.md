# Autonomous improvement log

Self-directed work session: "keep making it better" for ~10 hours. Each wake-up:
read this file, check training, do work, update this file, commit, schedule next.

## Current state
- **Model**: faithful LimX WF_TRON1A physics (real inertials/limits/efforts) + tire
  grip + an **actuated waist-roll joint** (BraveBot addition) so the RL policy
  balances roll by leaning the upper body instead of splaying the legs.
- **RL env** (`bravebot_sim/rl/env.py`): 40-d obs, 9-d action (6 leg + 2 wheel +
  1 waist), domain randomization + pushes + curriculum, reward shaping incl. ab/ad
  splay penalty.
- **Training**: waist-roll policy retraining (22M). Early posture = tucked legs,
  bent knees (natural). Need it to learn to balance with the waist.

## Backlog (work through, roughly priority order)
1. [active] Waist-roll policy: train to convergence; verify natural stance + robust.
2. Trainer: **keep-best-by-eval** checkpointing (last checkpoint != best; bit me at 75M).
3. Per-cycle: evaluate latest policy (walk/turn/strafe/push), showcase if improved.
4. Code review workflow → fix confirmed bugs.
5. RL: LR decay schedule, reward/curriculum tuning, longer/again if still climbing.
6. Robustness: rougher terrain, steps, stronger pushes in DR.
7. Feature: RL-policy-driven inspection patrol (legged inspection rounds).
8. Sim-to-real: more DR (latency, sensor bias), observation realism.
9. Code quality: tests, refactors, docstrings; regression coverage.
10. Visuals/docs: refresh gallery GIFs as policy improves; keep README current.

## END-OF-SESSION DELIVERABLE (user request)
After ~10h of training: update GitHub, render NEW demo videos + GIFs of the
improved policy, refresh the README/gallery with what it looks like now, and post
a written summary update of everything that improved.

## Log
- cycle 0: added waist-roll joint + retrain launched. Early posture natural (tucked).
- cycle 1: waist-roll run learned to balance (ep_len->677 @5M) BUT reverted to
  SPLAYING (abad ±30) under DR — the splay penalty is too weak vs splay's roll
  benefit, and it ignored the weaker waist. Fix: CAP abad range to 0.16 (force
  narrow) + keep waist for roll + soften pushes 130->100N. Relaunched 22M.
  Also shipped keep-best-by-eval trainer. NEXT: at ~next wake, check if capped+
  waist gives a natural tucked stance that still balances + is robust; if the
  waist authority is too weak (falls under push), consider lowering the waist
  pivot / widening range, or accept a modest stance. Watch knees (were over-bent).
