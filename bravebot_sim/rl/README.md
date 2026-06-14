# BraveBot RL — learned locomotion (walking + turning)

This is the **proper** path to robust walking and clean rotation. Hand-tuned
control can't do it on this robot: it's a high-CoM body on a single wheel axle
with no real roll actuator, so a *sustained* turn winds up an uncontrolled roll
mode and a 2-legged robot is statically unstable the moment it lifts a foot. Both
are classic **learned-policy** problems — the same reason the real LimX TRON 1
ships an Isaac-trained policy. Two policies are trained and shipped here (below).

## What's here

| File | Purpose |
|------|---------|
| `env.py` | Gymnasium env, command-conditioned `(vx, vy, yaw-rate)`. **Obs 40-d**, **action 9-d** = 6 leg position offsets + 2 wheel torques + 1 **torso-roll (waist)** target. The waist joint lets the policy lean the payload to control roll instead of splaying the legs. Reward = velocity/yaw tracking + upright (projected gravity) + height + alive + lean shaping − smoothness/effort/limit/slip penalties; terminates on a fall. Domain randomization + pushes + a command/DR curriculum are built in. Optional `model_path=` runs the policy inside the facility physics scene. |
| `heading_env.py` | `HeadingAwareEnv(BraveBotLocomotionEnv)`: **obs 42-d** — adds sin/cos of the heading error (actual yaw vs the integral of commanded yaw-rate) + a heading-hold reward, so the policy holds an absolute commanded heading instead of drifting. |
| `randomize.py` | `DomainRandomizer` (mass ±20%, friction ±40%, COM jitter, actuator gain ±20%, ±2.9° floor tilt), `ActionLatency` (~20 ms), `PushPerturber` (30–100 N shoves). All ramped by a curriculum. |
| `../../scripts/rl_train.py` | PPO (Stable-Baselines3) trainer. `--heading` trains the heading-aware track. **Keep-best is robustness-aware** (`combined_eval` = clean tracking + tracking under full DR + pushes), so the shipped checkpoint is chosen for sim-to-real robustness, not just nominal score. Curriculum resume is correct: the per-env step offset is baked into the env so `--resume` *continues* DR instead of restarting it. |
| `../../scripts/rl_view.py` · `rl_play.py` | Drive / watch a policy live or render it (`--heading` for the 42-d policy). |
| `../../scripts/rl_patrol.py` | RL-policy-driven inspection patrol (`--heading` for the full route, `--scene` to run in the facility). |
| `../../scripts/robustness_shootout.py` | Compare policies under full DR + pushes. |

## Shipped policies (`.onnx` = deployable actor)

| Policy | Obs | What it does |
|--------|-----|--------------|
| `policy_champion.onnx` (also `policy.onnx`) | 40 | Robust default. Walk / back / turn / arc, 0 falls; **survives 130–150 N shoves and ~±6° slopes**. Tracks yaw *rate* (drifts off absolute heading). |
| `policy_heading.onnx` | 42 | **Holds heading** (~5° drift on a straight command vs ~53° for the champion), tracks turns to ~1°, walks the **full out-and-back inspection round** (9/9 waypoints, 5/5 anomalies), equally shove/slope-robust. The fix for the rotation drift. |
| `policy_drhardened`, `policy_v2_25M`, `policy_v3`, `policy_v4`, `policy_v5_clean` | 40 | Historical snapshots kept for reference. |

## Workflow

```bash
pip install -r requirements-rl.txt        # torch + sb3 + gymnasium + onnx

python scripts/rl_train.py --steps 60000 --envs 6          # CPU sanity (reward climbs)
python scripts/rl_train.py --steps 25_000_000 --envs 16    # full champion run
python scripts/rl_train.py --heading --steps 30_000_000 --envs 16   # heading-aware track
python scripts/rl_train.py --resume --envs 16              # continue (same --envs as the checkpoint)

mjpython scripts/rl_view.py [--heading]                    # drive live (arrows = vx / yaw)
python   scripts/rl_play.py --cmd 0 0 0.6 [--heading]      # e.g. turn in place
python   scripts/rl_patrol.py --heading [--scene]          # full inspection round (in the aisle)
python   scripts/robustness_shootout.py                    # DR + push comparison
```

`policy_heading.onnx` / `policy_champion.onnx` are the deployable artifacts — load
them the way the TRON 1 `isaacgym/policy.onnx` is loaded, or drop into
`PhysicsBraveBot` to replace the scripted gait + balance controller.

## Notes

- **Compute:** these policies were trained for tens of millions of env steps; a
  60k-step CPU run only validates that the env + reward are learnable.
- **Lateral motion (vy):** the env exposes a `vy` command, but the robot rolls on a
  single wheel axle (the wheels can't roll sideways), so true sideways *strafing*
  isn't achievable without leg-stepping — the balance policy learns to ignore `vy`
  and keep its wheels planted. Forward/back, turning, and arcs are the real motion
  envelope; sideways repositioning is done by turn-then-drive.
- **Sim-to-real:** observation noise, action latency, and dynamics randomization
  are already in the training loop; the keep-best metric explicitly rewards
  robustness under that randomization.
- **Known limitation / next lever:** see the top-level `SESSION_UPDATE.md`. The
  champion's yaw-rate-only control is what motivated `heading_env.py`; a possible
  follow-up is making the heading policy the default everywhere (port the few
  remaining 40-d defaults to the 42-d observation).

## Deploying on the real robot (Tron 1)

`policy_*.onnx` maps a 40-d observation → a 9-d action in `[-1, 1]` (the export is
**clamped**, so the output is already bounded — no extra clip needed). Run the loop
at **50 Hz** and rebuild the observation each step **in this exact order**:

| # | field (size) | how to build it |
|---|---|---|
| 1 | projected gravity (3) | gravity-down in the **body frame**: `R_wb·ᵀ[0,0,−1]` where `R_wb` is the IMU world-from-body rotation (verified to match the trained obs) |
| 2 | base gyro (3) | body angular velocity rad/s (IMU) |
| 3 | base lin vel (3) | body linear velocity m/s (IMU) |
| 4 | leg qpos − stance (6) | the 6 leg-joint angles **minus** the stance `[0, 0.30, 0.60, 0, −0.30, −0.60]`, joint order `[abad_L, hip_L, knee_L, abad_R, hip_R, knee_R]` (mirrored LimX axes) |
| 5 | leg qvel (6) | the 6 leg-joint velocities rad/s, same order |
| 6 | wheel speeds (2) | wheel L, R angular velocity rad/s |
| 7 | waist roll q, qd (2) | torso-roll joint angle (rad) + velocity |
| 8 | gait clock (2) | `[sin φ, cos φ]`, advance `φ += 2π·1.4/50` each step (any start phase) |
| 9 | prev action (9) | the 9-vector you commanded last step |
| 10 | command (3) | target `(vx, vy, yaw_rate)` |
| 11 | dr ramp (1) | `1.0` at deployment |

Apply the 9-d action (`a`):
- `a[0:6]` → leg **position-servo** targets: `target_j = stance_j + a_j · range_j`, `range = [0.30, 0.60, 0.60, 0.30, 0.60, 0.60]`
- `a[6:8]` → wheel **motor torques**: `τ = a · 40 N·m` (L, R)
- `a[8]` → torso-roll **position-servo** target: `a · 0.9 rad`

For the **heading-aware** policy (`policy_heading.onnx`, 42-d) append 2 more obs:
`[sin(ψ_err), cos(ψ_err)]` where `ψ_err = yaw − ψ_des` and `ψ_des += yaw_rate_cmd / 50`
each step (the integral of the commanded yaw rate).
