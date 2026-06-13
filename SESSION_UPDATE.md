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

- **RL-driven inspection patrol.** A waypoint navigator on top of the learned policy
  drives the robot through a sensor sweep of the cold aisle
  ([scripts/rl_patrol.py](scripts/rl_patrol.py)). It runs stably and detects 4/5
  anomalies; clean full-route traversal is limited by the policy's heading drift (see
  below) and the deterministic kinematic `PatrolController` remains the survey-grade
  coverage tool.

- **Regression test suite** ([tests/test_sim.py](tests/test_sim.py), 9/9): meshes,
  both MJCF models compile with sane mass/CoM, URDF connectivity, kinematic
  drive+scan, physics balance, RL env shapes + determinism, scripted gait, and
  dedicated regressions for the two high-severity fixes above.

## New / refreshed demos
- `renders/rl_robust.mp4` — RL push recovery: the champion drives while absorbing
  5/5 lateral/longitudinal shoves at 130–150 N (above its 30–100 N training range).

- **Heading-aware policy → the rotation drift is fixed.** The 40-d champion tracks
  commanded yaw *rate*, not absolute heading, so it drifts (~53° off a straight
  command) and can't cleanly traverse the patrol. The new
  [`HeadingAwareEnv`](bravebot_sim/rl/heading_env.py) adds a heading-error
  observation (42-d obs) + a heading-hold reward; trained on its own track
  (`rl_train.py --heading`, champion untouched), the resulting policy holds heading
  (~5° drift), tracks turns to ~1°, **walks the full out-and-back inspection round
  (9/9 waypoints, 5/5 anomalies, 0 falls), and still survives 130–150 N shoves**
  ([renders/rl_patrol_heading.mp4](renders/rl_patrol_heading.mp4)). Training was
  still converging at the time of writing — the champion remains the protected
  shipped policy; this is a strictly-better-navigating sibling on its own track.

## Next lever
Promote the heading-aware policy to flagship once its training converges and its
40-d tooling (`rl_view`, `rl_play`) is ported to the 42-d observation.
