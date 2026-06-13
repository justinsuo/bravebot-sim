# BraveBot Sim — Improvement Session Update (2026-06-13)

A multi-hour autonomous "keep making it better" session. The robot is the
wheel-legged BraveBot (modified LimX TRON 1) on full MuJoCo rigid-body physics.

## Headlines

- **Code review found 13 real bugs — all fixed and adversarially re-verified.**
  A multi-agent review surfaced 13 confirmed issues; a verification workflow re-ran
  every repro on the patched code and *caught 2 of the first fixes as incomplete*,
  which were then re-fixed until a second pass returned all-clear. The most important:
  - **Curriculum-resume bug (high):** every `--resume` reset the per-env curriculum
    counter to 0, silently disabling domain randomization, pushes, and the command
    curriculum for ~4.5M steps of each resumed run — quietly eroding sim-to-real
    robustness. Fixed by baking the step offset into the env constructor.
  - **Permanent turn-lock (high):** the turn governor could lock turning forever
    (the roll re-arm gate was unsatisfiable). Replaced with a proactive budget + a
    *directional* lock: it blocks only the lean-worsening turn and always allows the
    releveling one — upright in every scenario, never frozen, never ratchets over.
  - **Kinematic stance ignored the mirrored LimX joint axes** (one wheel floated
    14.6 cm; bogus 59.5 kg model) — fixed to a symmetric, valid 34 kg rigid body.

- **Robustness-aware training → a measurably more robust shipped policy.**
  The keep-best eval was clean-only and structurally blind to robustness, so it had
  plateaued. Made it **robustness-aware** (clean tracking *plus* tracking under full
  domain randomization + pushes). Under the corrected metric, training produced and
  promoted strictly-better champions: the shipped policy now **survives 130–150 N
  shoves (5/5)**, 0 falls, while *recovering* clean performance (~1757) — robustness
  gained at no nominal cost. ([renders/rl_robust.mp4](renders/rl_robust.mp4))

- **RL-driven inspection patrol.** The learned policy now balances and drives a full
  legged inspection round of the data-center cold aisle while the onboard sensors
  scan for anomalies ([scripts/rl_patrol.py](scripts/rl_patrol.py),
  [renders/rl_patrol.mp4](renders/rl_patrol.mp4)).

- **Regression test suite** ([tests/test_sim.py](tests/test_sim.py), 9/9): meshes,
  both MJCF models compile with sane mass/CoM, URDF connectivity, kinematic
  drive+scan, physics balance, RL env shapes + determinism, scripted gait, and
  dedicated regressions for the two high-severity fixes above.

## New / refreshed demos
- `renders/rl_patrol.mp4` — RL-policy inspection round (walks the aisle, scans 4/5+).
- `renders/rl_robust.mp4` — push recovery, survives 5/5 shoves at 130–150 N.

## Known limitation / next big lever
The locomotion policy tracks commanded yaw *rate*, not absolute heading (it has no
heading observation), so it threads the aisle with some lateral drift — area
coverage works, survey-grade path-tracking does not. The clean fix is a
**heading-aware policy** (add a heading-error observation + retrain), the one
remaining large quality lever, left as a deliberate future change.
