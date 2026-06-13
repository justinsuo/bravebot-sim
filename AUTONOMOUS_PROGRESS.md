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
- cycle 6: *** FIXED ALL 13 CODE-REVIEW BUGS (adversarially verified) ***. The
  review workflow (w1mmjo6ot) confirmed 13 real bugs; I fixed every one, then ran an
  adversarial verification workflow (w96q5hy7z) that re-ran each repro on the patched
  code — it CAUGHT 2 of my fixes as incomplete (the _global resume set_attr only hit
  the Monitor wrapper; the turn-lock re-arm gate was still unsatisfiable for opposite
  turns). Re-fixed both and a 2nd verification (wt67t271g) returned all_clear.
  HIGH: (1) curriculum _global now baked into the env constructor via thunk() so DR
  continues across --resume (verified end-to-end: resumed dr_ramp=1.0, fresh=0.0) —
  this was silently de-randomizing every resumed run for ~4.5M steps. (2) turn governor
  = proactive budget + DIRECTIONAL lock (block only the lean-worsening turn, always
  allow the releveling one): upright in all 6 scenarios 60s+ AND opposite turns
  execute (no permanent freeze, no ratchet-to-tip). (3) kinematic stance now per-joint
  mirrored (both wheels seat; mass 59.5->34kg). MED: Saver no longer bursts ~50 saves
  on resume. LOW: NaN guard before reward, kinematic inertials, 40/9 docstrings,
  removed dead base_z/u, --check --walk actually walks. Physics XML byte-identical.
  tests 7/7, eval_balance PASS. Killed the de-randomized refine run; restored
  policy{,_best}<-champion; RELAUNCHED a refine run from the champion WITH the fix
  (first run with TRUE DR continuation — eval starts ~1720 under full DR, ep_rew
  3600->3300 as expected since DR is harder). Also fixed a .gitignore slip (untracked
  ~21MB of stray .zip snapshots + junk log; amended+force-pushed clean). NEXT: let the
  DR-hardened run harden; PROMOTE only if eval beats champion 1757 AND robust; then
  diversify (RL-driven inspection patrol / terrain). Watch the --envs-16 resume rule
  (SB3 asserts num_envs matches the checkpoint).
- cycle 5: PROMOTED refined policy (eval 1684->1757). Diversified: (a) added
  tests/test_sim.py (7 tests, all pass: meshes, both MJCF compile + sane mass,
  URDF connectivity, kinematic drive+scan, physics balance, RL env shapes+
  determinism, gait) — regression coverage. (b) launched a code-review workflow
  (w1mmjo6ot) to find bugs -> INTEGRATE confirmed fixes next cycle. (c) relaunched
  a refine run (pid 14892, 12M). NEXT: read code-review results, fix confirmed
  bugs; then RL-driven inspection patrol or terrain/DR.
- cycle 4: *** NATURAL STANCE ACHIEVED ***. Strong waist-upright penalty (-0.8)
  straightened the torso: policy now stands UPRIGHT (waist ~0) with a moderate ±24°
  stance — not the splits, not tilted. Robust: 0 falls through walk/turn/arc/back +
  120N & 90N shoves; tracking turn 0.06 / fwd 0.14; eval 1683 (best yet). Refreshed
  gallery GIF + mp4 to the new look. Saved policy_champion (the shipped deliverable).
  Launched a low-risk RESUME-refine run (12M, pid 94994) to polish further —
  champion is protected, so PROMOTE the refined policy next cycle ONLY if its eval
  beats the champion's. POSTURE IS SOLVED — next cycles DIVERSIFY: code-review
  workflow (find/fix bugs), unit tests, RL-driven inspection patrol, terrain/DR.
- cycle 3: moderate-splay(17°cap)+waist run FINISHED clean — robust, good tracking
  (fwd 0.11, turn 0.20, 0 falls), sane velocities, moderate ±22° splay (NOT splits),
  eval 1630 (keep-best worked, dodged a 20.5M PPO hiccup). Snapshot = policy_v3.
  Remaining cosmetic issue: constant -35° TORSO TILT (waist leans). Diagnosed: CoM
  is centered (+0.5mm) and tilt is identical across seeds -> spurious learned offset,
  NOT needed. Relaunched (pid 64073, 20M) with strong waist-upright penalty
  (-0.25 -> -0.8) to straighten the torso while keeping the moderate splay for roll.
  NEXT: check torso upright + still robust; if good -> SHOWCASE (render GIF, relaunch
  viewer, maybe refresh gallery). Then diversify backlog (don't only do posture):
  code-review workflow, RL-driven patrol, tests, terrain/DR.
- cycle 2: capped-0.16+waist run went DEGENERATE (waist saturated +52°, base_vel
  23.9 m/s exploit, ep_len stuck ~430, eval crashed to -8000). Key insight: the
  waist only shifts CoM within a NARROW base; splay actually WIDENS the support —
  so a too-narrow stance can't be roll-robust regardless of waist. Pivoted to a
  MODERATE splay (cap 0.30 ≈17°, natural-looking but roll-stable) as the main
  mechanism + waist as a GENTLE helper (penalize waist deviation -0.25*w^2 to keep
  torso upright + prevent exploit) + waist damping 3->8. Relaunched 22M (pid 43244).
  The new keep-best EVAL column is a great health signal — watch it stay positive.
  NEXT: check posture (want abad ~15°, torso upright, knees bent) + ep_len->800 +
  eval climbing. If good, showcase. If still struggling, try cap 0.35 or accept.
- cycle 0: added waist-roll joint + retrain launched. Early posture natural (tucked).
- cycle 1: waist-roll run learned to balance (ep_len->677 @5M) BUT reverted to
  SPLAYING (abad ±30) under DR — the splay penalty is too weak vs splay's roll
  benefit, and it ignored the weaker waist. Fix: CAP abad range to 0.16 (force
  narrow) + keep waist for roll + soften pushes 130->100N. Relaunched 22M.
  Also shipped keep-best-by-eval trainer. NEXT: at ~next wake, check if capped+
  waist gives a natural tucked stance that still balances + is robust; if the
  waist authority is too weak (falls under push), consider lowering the waist
  pivot / widening range, or accept a modest stance. Watch knees (were over-bent).
