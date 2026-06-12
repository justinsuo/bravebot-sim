# BraveBot RL — learned locomotion (walking + turning)

This is the **proper** path to robust walking and clean rotation. Hand-tuned
control can't do it on this robot: it's a high-CoM body on a single wheel axle
with no real roll actuator (measured: the ab/ad joints move roll only ~0.27° per
0.1 rad command — far too weak to control a turn). So a *sustained* turn winds up
an uncontrolled roll mode, and a 2-legged robot is statically unstable the moment
it lifts a foot. Both are classic **learned-policy** problems — the same reason
the real LimX TRON 1 ships an Isaac Gym / Isaac Lab trained policy.

## What's here

| File | Purpose |
|------|---------|
| `env.py` | Gymnasium env: command-conditioned (vx, vy, yaw-rate) locomotion. Obs 33-d, action 8-d (6 leg position offsets + 2 wheel torques), reward = command-tracking + upright + alive − effort, terminates on a fall. |
| `../../scripts/rl_train.py` | PPO (Stable-Baselines3) trainer → `policy.zip` + `policy.onnx`. |
| `../../scripts/rl_play.py` | Run a trained policy in the sim and render it. |

## Workflow

```bash
pip install -r requirements-rl.txt        # torch + sb3 + gymnasium + onnx

# quick CPU sanity run (proves the env learns — ep_rew_mean rises)
python scripts/rl_train.py --steps 60000 --envs 6

# real run — do this on a GPU box (millions of steps)
python scripts/rl_train.py --steps 5_000_000 --envs 16

# watch the trained policy
python scripts/rl_play.py --cmd 0.6 0 0 --video renders/rl_walk.mp4   # walk forward
python scripts/rl_play.py --cmd 0 0 0.6                               # turn in place
```

`policy.onnx` is the deployable artifact — load it the same way the TRON 1
`isaacgym/policy.onnx` is loaded, and drop it into `PhysicsBraveBot` to replace
the scripted gait + balance controller with the learned policy.

## Notes / next steps

- **Compute:** a biped locomotion policy needs ~10⁷ environment steps; that's a
  GPU job (your TRON 1 cloud-GPU / Isaac workflow is ideal). The included 60k-step
  CPU run only validates that the env + reward are learnable (reward climbs); it
  will not walk yet.
- **Curriculum:** start with vx-only small commands, then add yaw, then vy
  (lateral). Lateral (strafe) is the hardest given the wheel layout.
- **Sim-to-real:** add observation noise, action delay, and dynamics
  randomization (mass, friction, latency) before transferring to hardware.
- **Reward shaping:** `env.py`'s reward is a sensible default; add foot-clearance,
  gait-symmetry, and contact-schedule terms for cleaner gaits.
