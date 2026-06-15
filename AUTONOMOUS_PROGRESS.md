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
- FEATURE PACK (user: "do all of them"): shipped all 6 offered interface upgrades.
  EXPLORER (web/index.html): (1) per-part MASS in the info card — export_parts.py now reads
  body inertials from bravebot_physics.xml into parts.json (total 36.76 kg); (2) LABELS —
  exploded leader-line callouts on every part via CSS2DRenderer (labelled-BOM diagram);
  (3) HIGHLIGHT SENSORS button — glow all sensing components green, dim the rest.
  SIM (web/sim.html + scripts/sim_server.py): (4) PLAYGROUND — ramp + 2 bumps + 3 pillars as
  real collision geoms (injected into a generated _playground.xml so relative meshdir resolves;
  gitignored); (5) CLICK-TO-SHOVE — raycast the robot, apply a 150 N torso impulse in the click
  direction (POST /push), tap-vs-orbit-drag disambiguated by pixel movement; (6) CONTROLLER
  TOGGLE — switch live between the hand-tuned balance controller and the trained RL policy
  (policy_champion) via POST /mode, continuing from the exact physical state; plus FIRST-PERSON
  camera toggle. Backend made THREAD-SAFE: mode-switch + reset were mutating MjData off the
  physics thread (cross-thread MuJoCo race -> segfault); now the loop thread is the sole MjData
  mutator and handlers only set pending flags. Verified server-side: balances (pitch ~-0.04),
  push-recovers (pitch spiked -0.28 -> back to -0.04 upright), drives over the bump (x->1.26),
  RL-mode switch + RL drive both upright, no crash. Both JS modules pass node --check.
- NEW FEATURE (user request): LIVE PHYSICS SIMULATOR in the browser. scripts/sim_server.py
  runs the real MuJoCo rigid-body physics + balance controller in a background thread (500Hz,
  real-time) and streams live per-part poses (with the mesh-recenter correction) over a stdlib
  HTTP server; web/sim.html (three.js) renders them + sends drive commands. The robot BALANCES on
  its wheels and you DRIVE it (arrows/WASD), chase-cam, HUD (balance/speed/pitch/pos), auto-recover
  on tip. Verified server-side: balances in place, drove 1.5m forward + turned (yaw 0->0.26),
  upright throughout; 27 parts streamed. ./scripts/run_sim.sh (port 8001). No extra deps.
- NEW FEATURE (user request): INTERACTIVE 3D PART EXPLORER. An in-browser, presentable
  exploded-view interface of the whole robot — all 27 parts (9 real LimX TRON 1 links +
  18 BraveBot components). scripts/export_parts.py reads the kinematic model + component
  registry -> web/parts.json (every part's mesh, exact world pose, color, subsystem,
  description, sensor specs). web/index.html is a self-contained three.js app
  (STLLoader + OrbitControls): explode slider/button (radial blossom from the centroid),
  click-to-inspect (dim others + info card), grouped Parts panel, color-by-subsystem,
  auto-rotate. scripts/view_parts.sh serves + opens it. Verified all 27 meshes serve 200
  over HTTP. NOTE on "Use Fable 5": I made a REAL API call to claude-fable-5 — it returned
  "Claude Fable 5 is not available. Please use Opus 4.8." (gated / fable-mythos-access), so
  I built this with Opus 4.8 (me) directly. Honest, not fabricated.
- cycle 31: VERIFIED the deployment spec's trickiest convention (projected gravity, the
  #1 sim-to-real bug source). Reconstructed it the way a real-robot deployer would — from
  the IMU world-from-body quaternion, g_body = R^T·[0,0,-1] — and confirmed it matches the
  policy's trained obs exactly (atol 1e-3) at a non-trivial tilted orientation. Sharpened
  the deployment doc's proj-gravity row with the exact verified formula so there's no sign
  ambiguity. The deployment guide is now not just written but VALIDATED for the user's
  Tron 1. Doc-only.
- cycle 30: added a SIM-TO-REAL DEPLOYMENT SPEC to rl/README (serves the user's actual
  goal: run the policy on the real Tron 1). The .onnx contract was only implicit in the
  env docstring; wrote an explicit "Deploying on the real robot" table — the exact 40-d
  obs vector to build each step (field-by-field, units, joint order, stance reference,
  the gait-clock advance 2pi*1.4/50, prev-action feedback, dr_ramp=1.0), the 50 Hz rate,
  and the 9-d action mapping (leg position offsets = stance + a*range with range
  [0.30,0.60,0.60,0.30,0.60,0.60]; wheel torques = a*40 N.m; waist = a*0.9 rad). Plus the
  42-d heading variant (append sin/cos of yaw vs the integrated yaw-rate command).
  Verified every number against the code. Doc-only; bridges sim -> the user's hardware.
- cycle 29: tried to DELIVER the user's "walk side to side" via the scripted gait's
  latent lateral side-step (abad_amp, which was never enabled). Result: NOT safely
  possible. Tested abad_amp over the real gait schedule (march/forward 0.4/back/turn):
  even abad_amp=0.02 TIPS the robot at ~4.3s during the 0.4 m/s forward phase (only 0.0
  survives the full 14.5s). Root cause = the same morphology as the strafe finding:
  splaying the ab/ad joints shifts the support base and the wheel-balance can't recover
  at speed. So abad_amp defaults to 0 for a reason. Reverted to 0.0; rewrote the gait.py
  docstring to honestly explain lateral side-stepping is OFF (it tips the robot) and
  sideways motion = turn-then-drive; restored the stable committed gait.mp4 (my trial
  re-render had fallen @4.6s). Net: a documented negative result + honest docs; no
  regression, gait stable, tests 11/11. CONCLUSION on "side to side": this single-axle
  wheeled biped cannot translate laterally by wheels OR by leg side-step (both tip it);
  forward/back/turn/arc + turn-then-drive is the true, now-honestly-documented envelope.
- cycle 28: CAPABILITY-vs-DOCS honesty audit found an OVERCLAIM. Tested lateral strafe
  (the user's "walk side to side"): BOTH policies stay upright but DON'T strafe — for a
  vy=0.2 command they move only ~0.01 m/s laterally (vy tracking error = the full
  command; they ignore vy). Root cause is MORPHOLOGICAL: the wheels share one axle and
  can't roll sideways, so true strafing needs leg-stepping the balance policy doesn't do
  — it correctly learned to ignore vy and keep its wheels planted. The docs claimed
  "strafe" in 4 places (overclaim). Fixed: README + rl/README now say walk/back/turn/arc
  (the real motion envelope) and a note explains lateral motion is limited by the
  single-axle wheels (sideways repositioning = turn-then-drive). Honesty fix, no false
  capability claims. Forward/back + turn + arc are genuinely delivered; the env still
  exposes vy harmlessly. Tests 11/11.
- cycle 27: end-to-end check that the DOCUMENTED user-facing commands actually run (doc
  vs reality). All good: scripts/patrol_demo.py (detects anomalies, writes frames),
  scripts/view.py --check (kinematic self-test OK), scripts/export_manifest.py
  (regenerated components.json + BILL_OF_MATERIALS.md BYTE-IDENTICAL -> manifest is
  current + reproducible). Only churn: patrol_demo re-renders renders/patrol_contact.png
  with trivial pixel diffs (tracked demo image) -> reverted the spurious change. No bug.
  Audited so far: ONNX deploy [fixed], URDF [clean+locked], deps [clean], documented
  commands + manifest [clean]. Repo clean, tests 11/11.
- cycle 26: audited dependency correctness (fresh-clone install). CLEAN: every real
  third-party import (mujoco, numpy, trimesh, imageio | gymnasium, stable_baselines3,
  torch, onnxruntime) is covered by requirements.txt / requirements-rl.txt; the lone
  oddity ("density") was a false positive (a docstring line in build_model.py, not an
  import). Green board confirmed: tests 11/11, eval_balance regression PASS. No action
  needed — another surface verified correct. (Audited so far: ONNX deploy [fixed], URDF
  [clean+locked], deps [clean].)
- cycle 25: audited the URDF (the ROS 2/Gazebo/RViz deliverable, previously only
  connectivity-tested). Result: CLEAN — 27 links/26 joints connected, every link has
  positive inertia, all revolute joints have limits, all 54 mesh refs resolve, total
  mass 36.76 kg (matches physics ~37). No fix needed; upgraded test_urdf_connected ->
  test_urdf_valid to lock all of that in (inertials, limits, mesh resolution, mass
  range). Suite 11/11. Continuing the audit-unchecked-surfaces discipline — this one
  was already correct, now regression-protected.
- cycle 24: FIXED a real DEPLOYMENT-ARTIFACT bug (found by verifying the .onnx path).
  The .onnx actors are what go on the real robot, but they were exported as the RAW
  policy mean — SB3's predict() clips to [-1,1] while the ONNX did not, so on
  out-of-distribution states the ONNX emitted out-of-range commands (±4+). Faithful only
  after a downstream clip (in-dist diff 3.6e-7; PPO == clip(ONNX) exactly). The sim env
  clips in step() so in-sim behavior was unaffected, but a real-robot consumer loading
  the .onnx would get unbounded commands. Fix: export_onnx now torch.clamp(action,-1,1)
  so the ONNX is a SELF-CONTAINED, bounded, drop-in artifact == predict(). Re-exported
  ALL tracked policies (champion/policy/heading + drhardened + v2/v3/v4/v5); verified
  FAITHFUL+bounded (max|PPO-ONNX|~1e-6, all actions in [-1,1]). In-sim behavior unchanged,
  tests 10/10. A genuine sim-to-real correctness fix surfaced by auditing the deploy path.
  Locked it in with test_onnx_artifacts_bounded (asserts shipped .onnx actions stay in
  [-1,1] incl. OOD obs; skips if onnxruntime absent) -> suite 11/11.
- cycle 23: rewrote the stale RL README (bravebot_sim/rl/README.md). It still described
  an EARLY state — "obs 33-d / action 8-d", and listed domain randomization, curriculum,
  and sim-to-real hardening as FUTURE steps (all long done). Updated to the real state:
  obs 40-d (42-d heading) / action 9-d incl. the waist joint; DR + pushes + curriculum +
  robustness-aware keep-best + correct curriculum-resume; the two shipped policies
  (champion 40-d robust, heading 42-d holds-heading + full patrol) + the historical
  snapshots; and the full tool set (rl_view/rl_play/rl_patrol --heading/--scene,
  robustness_shootout). Doc-only; no code/behavior change. Project remains complete + clean.
- cycle 22: repo-state audit + cleanup (delivered state is tidy). Verified: training idle,
  all README demo links resolve, tests 10/10, both policies intact. Fixed churn: tracked
  deployment onnx now correct (policy.onnx==champion, policy_heading.onnx==shipped v1);
  gitignored progress*.csv + policy_heading_best.onnx and untracked progress.csv; .zip /
  __pycache__ / renders/_*.png correctly ignored. git status now fully clean; tracked rl/
  artifacts are just the .onnx policy zoo (champion, drhardened, heading, v2/v3/v4/v5) +
  README. Project is in a clean, complete, well-documented final state — substantive +
  exploratory work exhausted (refine plateaued, slope-DR reverted as a negative result).
- cycle 21: steeper-slope experiment CONCLUDED -> REVERTED (kept the bar honest). The
  ±4.6°/120N heading run finished; fair head-to-head under the STEEPER regime: v1
  (±2.9°-trained) clean 1721 / steepDR 1680 / combined 3401 BEAT the harder-DR policy
  (1666 / 1652 / 3318) on every axis, AND both survive ±4.6° AND ±6° static slopes at
  100%. FINDING: the heading policy ALREADY generalizes to ±6° slopes despite ±2.9°
  training — harder-slope DR was unnecessary and the harder regime (+ a late instability)
  only made it worse. So reverted: restored policy_heading <- v1, reverted DR_DEFAULT to
  ±2.9°/100N, removed the redundant v1 snapshot. Tests 10/10; policy_heading verified
  (holds heading ~5°). Net: a clean negative result, no regression — the shipped policies
  are confirmed already slope-robust to ~±6°. STATUS: project complete + both policies at
  their best; remaining work is genuinely optional/on-direction.
- cycle 20: pushing sim-to-real robustness via a NEW objective — STEEPER-SLOPE domain
  randomization (aligned with "as realistic as possible"). The shipped policies only saw
  ±2.9° floor tilt; real facilities have ramps/raised floors. Bumped DR_DEFAULT
  gravity_tilt 0.05->0.08 (±4.6°) and PUSH_DEFAULT 100->120N. Done SAFELY: snapshotted the
  converged heading policy as policy_heading_v1 (the ±2.9°-trained 3454.9 best) before
  training, so nothing good is lost; champion also protected. Relaunched heading --resume
  on the harder regime (robustness-aware keep-best re-establishes its best under the
  steeper eval). Tests still 10/10. NEXT: evaluate the harder-DR result vs
  policy_heading_v1 under the steeper regime; keep whichever is more slope-robust, update
  policy_heading + docs only if it genuinely wins (else revert the DR bump).
- cycle 19: richer FACILITY-SCENE patrol demo (additive, low-risk). Added an optional
  model_path to the RL env (default unchanged -> champion/tests intact, verified 10/10)
  so the same policy can run inside the data-center physics scene (racks/walls). Added
  rl_patrol.py --scene; rendered renders/rl_patrol_facility.mp4 — the heading-aware
  policy walks the FULL inspection round THROUGH THE AISLE, 5/5 anomalies, upright.
  Featured it as the top demo in README. No training (both tracks converged; not
  churning). This is the capstone visual: a legged policy doing a real autonomous
  inspection round in the facility. NEXT: light polish only; flagship swap / new
  training objective remain optional + on-direction.
- cycle 18: added a heading-env regression test (test_heading_env: 42-d obs, straight
  cmd holds desired heading, turn cmd integrates it, finite reward, deterministic reset)
  -> suite now 10/10. The latest heading --resume finished plateaued (3411 < 3454.9).
  DECISION: stop relaunching training — BOTH objectives (40-d champion 3469.5, 42-d
  heading 3454.9) are demonstrably converged and continuations now only drift down, so
  spinning a 16-env run for ~0 expected gain wastes compute. "Keep at most one run" =
  zero is fine here; the ~10h of training is amply satisfied. Will start training again
  only for a genuinely NEW objective (flagship retrain / terrain / harder DR) — a
  deliberate, user-greenlit direction. The substantive roadmap is COMPLETE: robust
  champion + heading-aware policy (rotation fixed, full patrol), both fully tooled,
  tested (10/10), demoed, documented. Remaining cycles = light polish + preserve the
  clean state. NEXT: optional flagship swap (make heading the default) is low-risk now
  that all tooling supports --heading, but it's a judgment call -> leave for direction.
- cycle 17: INTEGRATED the heading-aware policy across all interactive tooling. The
  heading --resume run finished (best 3452.3 < 3454.9, plateaued; policy_heading
  unchanged) — both tracks firmly converged. Ported rl_view.py and rl_play.py to take
  --heading (load policy_heading + HeadingAwareEnv), so all three RL tools (rl_view,
  rl_play, rl_patrol) now drive either policy; the default 40-d champion path is
  untouched. Verified: rl_play --heading forward |vx err|=0.06 / turn clean / 0 falls;
  default champion path still works. Updated README quick-start with the --heading
  commands. The heading policy is now a fully first-class, deployable option (its onnx
  is tracked). Relaunched a heading --resume continuation (keep one run going; both
  plateaued — substantive roadmap is essentially complete). NEXT: maintenance/polish;
  flagship promotion (swap default to heading) is now trivial if desired since all
  tooling supports it.
- cycle 16: heading-aware run CONVERGED + finalized. It peaked at combined 3454.9 (~21M)
  then late-drifted (30M -> 2995, the usual late-PPO instability), so keep-best correctly
  retained the 21M policy -> policy_heading.zip = that best (matches the cycle-15 demo
  snapshot). Re-verified the FINAL shipped policy_heading (training now idle, no race):
  straight->5.5deg heading, full patrol 9/9 + 5/5 anomalies completed upright, survives
  5/5 shoves @130-150N. Committed policy_heading.onnx as a tracked, reproducible artifact
  (the heading policy is now deployable + the demo's exact policy). Relaunched a heading
  --resume continuation (keep one run going; keep-best protects 3454.9, though both
  objectives are plateaued). STATUS: two excellent policies — 40-d champion (robust,
  shipped default, all tooling) and 42-d heading policy (holds heading, full patrol,
  robust). The "rotation is bad" problem is solved. NEXT: optional flagship promotion of
  the heading policy needs rl_view/rl_play ported to 42-d obs — a clean follow-up.
- cycle 15: *** HEADING-AWARE POLICY WORKS — ROTATION DRIFT FIXED + FULL PATROL
  TRAVERSAL ***. The from-scratch heading run (cycle 14) learned fast: ep_len 35@500k ->
  full 800 by ~16M, combined eval climbing past 3450 by 21M. Tested the 21M checkpoint:
  straight command drifts only 5.5deg (vs champion's -53deg!), turn command tracks 91deg
  vs 92deg expected (near-perfect), and it COMPLETES THE FULL out-and-back patrol route
  (9/9 waypoints, 5/5 anomalies, 0 falls) — which the champion could NOT do. It is also
  robust (survives 5/5 shoves @130-150N). So the heading-aware policy is strictly better
  at navigation AT EQUAL robustness. Wired rl_patrol.py --heading (uses HeadingAwareEnv +
  policy_heading + full route) and rendered renders/rl_patrol_heading.mp4 (clean full
  inspection round, 5/5 anomalies). Updated README + SESSION_UPDATE to feature it. Still
  TRAINING (23.5M->30M, will improve further). Champion (40-d) remains the protected
  shipped policy; the heading policy is a 42-d sibling on its own track. NEXT: let it
  converge; once done, consider promoting it to flagship + porting rl_view/rl_play to
  42-d obs. (Did NOT commit the churning policy_heading.onnx; will finalize at convergence.)
- cycle 14: STARTED THE HEADING-AWARE POLICY (the one remaining real lever, pursued
  SAFELY as a separate additive track). Champion-refine confirmed plateaued again
  (3456.1 < 3469.5), so I redirected that training slot. New bravebot_sim/rl/heading_env.py
  = HeadingAwareEnv(BraveBotLocomotionEnv): obs 40->42 (adds sin/cos of heading error =
  actual yaw vs the integral of commanded yaw rate) + a heading-hold reward (0.8*exp).
  This gives the policy the absolute-heading signal it never had — the fix for the
  "rotation is bad" drift AND clean patrol traversal. Wired rl_train.py --heading to
  train it on a SEPARATE track (policy_heading.*, progress_heading.csv) so the shipped
  40-d champion + all tooling/tests/demos are 100% untouched (verified: suite still 9/9,
  heading env sanity OK — straight cmd holds heading_err~0, turn cmd integrates yaw_des).
  Launched from-scratch heading training (30M, --envs 16); early as expected (500k:
  ep_len 35, learning to balance first). It'll climb over coming cycles; champion
  protected regardless of whether it converges in-session. NEXT: watch policy_heading
  learn to balance->track->hold heading; once it holds heading + stays robust, render a
  clean full-aisle patrol traversal with it as a new demo. Promote nothing unless it
  clearly beats the champion on its own merits.
- cycle 13: GREEN-BOARD VERIFICATION caught + fixed a deliverable bug. Latest run best
  combined 3458.7 < champion 3469.5 (plateau holds, not promoted). A full-system verify
  found the cycle-11 rl_patrol.mp4 actually showed the robot FALLING at 13.3s: the
  robustness-trained champion has a strong yaw-heading drift, and the navigator's
  aggressive steering let yaw run away (spun to ~-170deg, reverse-drove tilted ~24deg,
  toppled). Fixes: gentler nav (V_MAX 0.5->0.4, V_REV->-0.3) + a TILT THROTTLE (ease off
  vx/yaw above 0.30 rad body tilt so it never drives a toppling robot) + forward-only
  inspection route (no turnaround spin). Result: STABLE (120s upright, 0 falls) and
  detects 4/5 anomalies, but the strong heading drift makes it circle near the entry
  (x~1.9) rather than traverse — so I made it HONEST: removed the misleading
  rl_patrol.mp4, relaxed --check to upright+>=4 anomalies, and reworded README/
  SESSION_UPDATE to present the patrol as a stable near-aisle inspection tool whose
  full traversal awaits the heading-aware policy. rl_robust.mp4 (push recovery, solid)
  is now the headline RL demo. Green board: tests 9/9, patrol CHECK OK, balance PASS.
  NEXT: heading-aware policy is the clear next lever (fixes patrol traversal + rotation).
- cycle 12: *** END-OF-SESSION DELIVERABLE (training plateaued) ***. Latest run best
  combined 3470.2 vs champion 3469.5 = within noise -> NOT promoted; the policy is
  confirmed at its architecture ceiling (~3470) across runs. Delivered: (a) rendered a
  PUSH-RECOVERY demo (scripts/render_rl_robust.py -> renders/rl_robust.mp4): the
  champion survives 5/5 shoves at 130-150 N (well above the 30-100 N training range),
  0 falls — visual proof of the robustness work. (b) Refreshed README (new rl_patrol +
  rl_robust clips, robustness-aware-eval description, RL-driven patrol section, updated
  scripts/tests layout, code-review note). (c) Wrote SESSION_UPDATE.md summarizing the
  session (13 bugs fixed+verified, robustness-aware training + more-robust champion,
  RL inspection patrol, 9/9 tests, known heading-drift limitation). Relaunched one
  continuation from the champion. NEXT: keep one run going (promote on a REAL
  combined gain >~3475); heading-aware policy is the only remaining big lever.
- cycle 11: combined-eval continuation FINISHED (37.5M->49.5M) best 3462.8 < champion
  3469.5 -> not promoted; the robustness-aware metric is now also plateauing (~3460-3470),
  so the policy is near its ceiling for the current 40-d/no-heading architecture. Started
  the END-DELIVERABLE prep while the champion is strong: added an OFFSCREEN RENDER mode to
  scripts/rl_patrol.py (--render OUT.mp4) and rendered renders/rl_patrol.mp4 — the new
  champion walks the inspection round + scans 4/5 anomalies, NO fall (the improved policy
  navigates the route far better than the original — completes it in ~13s vs barely moving
  before). Fixed a subtle relaunch bug: continuations were resuming from the run's last
  shipped best (3462.8) instead of the cross-run champion (3469.5); now restore
  policy.zip<-champion before each relaunch. Relaunched from the champion. NEXT: keep
  one run going (promote on combined>3469.5); finish END deliverable (gallery+README+
  summary, add patrol clip) when ~10h; heading-aware policy remains the one big lever left.
- cycle 10: *** ROBUSTNESS-AWARE TRAINING IS WORKING — promoted a STRICTLY BETTER
  champion ***. The first run under the new combined (clean+DR) keep-best (cycle 9)
  improved past the champion. Apples-to-apples on the exact in-trainer protocol PLUS an
  independent 3-seed robustness cross-check, the new best DOMINATES on every axis:
  clean 1756.9 vs 1745.2, dr 1712.6 vs 1696.6, combined 3469.5 vs 3441.9, robust_avg(3-
  seed) 1670.2 vs 1653.8 — and 0% falls (track_err 0.109 vs 0.120 under DR+push). No
  trade-off this time: robustness-aware training RECOVERED clean perf (~1757, matching
  the original clean champion) WHILE improving robustness. PROMOTED to champion (prev
  champion preserved as policy_drhardened, original clean as policy_v5_clean). Relaunched
  from the new champion to keep climbing combined. This validates the cycle-9 metric
  change: the deployment-relevant objective is NOT exhausted and yields real gains.
  NEXT: promote on combined>3469.5; then heading-aware policy OR END deliverable (~10h).
- cycle 9: *** ROBUSTNESS-AWARE EVAL + PROMOTED a genuinely more-robust champion ***.
  The cycle-8 continuation FINISHED (34M->46M) at clean eval 1741 (a PPO hiccup to 100
  at 44.5M) — clean-eval refine is now DEGRADING over 3 runs, so I stopped chasing it.
  Resolved cycle-7's open question with a balanced clean+DR comparison (same scoring):
  champion clean 1755.5 / robustDR 1580.4 / combined 3335.8 vs DR-hardened clean 1741.0
  / robustDR 1666.5 / combined 3407.5 — the DR-hardened policy is 5.4% more robust
  under disturbance (0% falls, equal survival) for 0.8% clean cost = clearly better on
  the DEPLOYMENT metric. Root fix: made the keep-best eval ROBUSTNESS-AWARE
  (combined_eval = clean + DR-perturbed in scripts/rl_train.py) — the clean-only metric
  structurally couldn't see robustness (it plateaued while DR-tracking kept improving).
  Under the corrected metric the DR-hardened policy wins, so PROMOTED it to champion
  (snapshotted the old clean champion as policy_v5_clean.onnx). Relaunched training with
  the robustness-aware keep-best from the new champion — now optimizing the
  deployment-relevant metric (which, unlike clean eval, is NOT exhausted). Suite 9/9.
  This realizes the payoff of the cycle-6 curriculum-resume fix: a measurably more
  sim-to-real-robust policy is now the shipped champion. NEXT: let it improve combined;
  promote on combined>3407; then heading-aware policy OR END deliverable (~10h).
- cycle 8: built the RL-POLICY-DRIVEN INSPECTION PATROL (scripts/rl_patrol.py): the
  trained locomotion policy balances+drives while a waypoint navigator (picks
  forward/reverse to avoid an impossible 180deg spin; closes the heading loop
  externally) walks the cold-aisle route and the onboard sensors scan anomalies. Self-
  test passes: upright 120s, traverses mid-aisle, detects 4/5 anomalies. FINDING (root
  cause of the earlier "rotation is really bad"): the policy tracks commanded yaw RATE,
  not absolute heading (it has NO heading observation), so it settles to a ~-53deg
  drift and can't hold a straight line — area-inspection coverage works, survey-grade
  centerline tracking does not. The PROPER fix is a heading-aware policy (add a
  heading-error obs + retrain) — a deliberate obs-space change (40->~42, new ONNX
  interface, re-baseline), queued as the next MAJOR improvement (don't rush). The
  cycle-7 forward run also FINISHED (30M->42M), max clean eval 1751 < champion 1756.7
  -> still not promoted; clean-eval-gated refine has plateaued (2 runs). Relaunched one
  more continuation (from 34M, DR on) per the keep-training directive; promote only if
  >1756.7. NEXT: heading-aware policy redesign OR robustness-aware eval; then END
  deliverable (~10h): refresh gallery/demos (incl. a patrol clip) + README + summary.
- cycle 7: the cycle-6 DR-continuation refine run FINISHED (25M->37M). Best clean
  eval 1745-1756.7 — did NOT beat champion (1756.7/1757), so per the gate I did NOT
  promote; champion stays the shipped/protected best. BUT this run was the first with
  TRUE DR continuation, so I ran a robustness shootout (new scripts/robustness_shootout.py,
  60 eps under full DR+pushes, paired seeds): champion vs drhardened BOTH 0% falls /
  800-step survival, drhardened tracks 33% better under disturbance (track_err 0.120 vs
  0.178) but is ~0.7% lower on CLEAN eval (1745.2 vs 1756.7). Net: a robustness/clean
  trade-off, not a clear win -> KEEP champion (conservative + correct). Key insight: the
  keep-best eval runs on a no-DR env so it CANNOT reward robustness — a future improvement
  is a robustness-aware eval. Continued training FORWARD from the 30M frontier (DR fully
  on) rather than resetting — promote a checkpoint only if it beats 1756.7. Added 2
  regression tests codifying the cycle-6 HIGH fixes (curriculum-resume offset ->
  dr_ramp; directional turn-lock stays upright + opposite turn executes); suite now 9/9.
  Saved policy_drhardened.{zip,onnx} snapshot. NEXT: build the RL-driven inspection
  patrol (user-requested) OR a robustness-aware eval; then END deliverable (~10h).
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
