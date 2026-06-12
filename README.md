# BraveBot Sim

A 3D model and simulation library for **BraveBot** — the wheel-legged autonomous
inspection robot from [vigiles / bravebot-website](https://github.com/justinsuo/bravebot-website).
BraveBot is built on a modified **LimX Dynamics TRON 1** (WF_TRON1A) base and
carries a four-sensor inspection payload (acoustic · thermal · gas · visual)
with on-edge AI.

This repo turns the marketing-site concept into a runnable robot: a complete
robot description (MJCF + URDF), procedurally generated component meshes, and a
MuJoCo-based Python library to drive it, patrol a facility, and simulate sensor
coverage and anomaly detection.

![BraveBot patrolling a data-center aisle](renders/hero_patrol.png)

---

## What's in the BraveBot

The model is assembled from two sources:

| Part of the robot | Source | Geometry |
|-------------------|--------|----------|
| Base, ab/ad, hip, knee, wheel (×2 legs) | **Real LimX TRON 1 meshes** (Apache-2.0) | `description/meshes/tron1/*.stl` |
| Payload deck, rugged torso, mast, sensor head, edge-AI core, hot-swap battery, displays, rails, antennas, e-stop, and the 4 sensors | **Original BraveBot geometry**, generated from primitives | `description/meshes/bravebot/*.stl` |

Every component, its mount frame, and its engineering metadata live in one place:
[`bravebot_sim/registry.py`](bravebot_sim/registry.py). A full parts list is in
[`docs/BILL_OF_MATERIALS.md`](docs/BILL_OF_MATERIALS.md) and
[`description/config/components.json`](description/config/components.json).

**Four-sensor inspection stack:** phased acoustic imaging array (ultrasonic to
100 kHz), radiometric thermal camera, VOC/CO/H₂ gas + optional OGI/TDLAS, and an
HD visual-AI camera with gauge OCR — plus a LiDAR/depth navigation cluster.

![Studio view](renders/hero_studio.png)

---

## Quick start

```bash
cd ~/bravebot-sim
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/build_model.py        # generate meshes + MJCF + URDF
python scripts/view.py               # open the interactive viewer
```

### Interactive viewer

```bash
python scripts/view.py               # robot in the data-center scene
python scripts/view.py --bare        # robot only, empty floor
```

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `W` / `S` | drive forward / back | `A` / `D` | turn left / right |
| `Q` / `E` | stance lower / taller | `X` | full stop |
| `SPACE` | scan → print risk-scored alerts | `P` | toggle autonomous patrol |
| `R` | reset to charging dock | `O` | print odometry |

### Headless patrol demo (no window)

```bash
python scripts/patrol_demo.py        # drive the route, detect anomalies, render frames
```

```
[t=  0.2s]  detected via acoustic array
   ALERT · Rack 4B coolant micro-leak
     finding:    ultrasonic jet at a liquid-cooling connector
     risk:       moderate  (confidence 70%)
     action:     isolate line, inspect connector/switchgear, dispatch technician
...
Anomalies detected: 5/5
```

---

## Library API

```python
from bravebot_sim import BraveBot, PatrolController, diagnose, scene_path, facility

bot = BraveBot(scene_path())
ctrl = PatrolController(bot, facility.WAYPOINTS)

for _ in range(1500):
    ctrl.update()                      # nav toward next waypoint
    bot.step(0.02)                     # advance the unicycle model + pose legs/wheels
    alert = diagnose(bot.scan(facility.ANOMALIES))   # edge-AI fusion stub
    if alert and alert.confidence > 0.45:
        print(alert.render())

pos, look = bot.sensor_frame("thermal")  # world pose of any sensor
bot.set_stance("low")                    # wheel-legged variable geometry
```

The default control mode is **kinematic patrol**: the base follows a unicycle
(v, ω) model and the wheels/legs are posed for display. It is deterministic and
never tips over — the right abstraction for inspection-coverage and sensor
simulation. Velocity actuators on the wheels are present in the MJCF for
balance/physics research, but keeping a wheeled biped upright is the real-robot
RL problem and is intentionally out of scope here.

---

## Layout

```
bravebot_sim/         the library
  registry.py         every component, mount frame, sensor + spec (source of truth)
  meshgen.py          procedural STL generation for BraveBot parts (trimesh)
  sim.py              BraveBot class: drive, stance, sensor frames, scan
  facility.py         data-center scene + anomalies + patrol waypoints
  patrol.py           waypoint controller + edge-AI diagnosis stub
description/
  meshes/tron1/       real LimX TRON 1 link meshes (Apache-2.0, see NOTICE)
  meshes/bravebot/    generated BraveBot component meshes
  mjcf/               MuJoCo model (bravebot.xml) + scene (bravebot_scene.xml)
  urdf/               bravebot.urdf for ROS 2 / Gazebo / RViz
  config/             components.json manifest
scripts/              build_model · view · patrol_demo · render_hero · export_manifest
renders/              hero stills + patrol frames
```

## Regenerating everything

```bash
python scripts/build_model.py      # meshes + MJCF + URDF
python scripts/export_manifest.py  # components.json + bill of materials
python scripts/render_hero.py      # hero stills (offscreen)
```

---

## How this fits a real robotics stack

The simulation mirrors BraveBot's control hierarchy and is designed to drop into
a ROS 2 / TRON 1 workflow:

- **URDF** (`description/urdf/bravebot.urdf`) loads in RViz / Gazebo and can be
  repackaged as a `*_description` ROS 2 package (swap the relative mesh paths for
  `package://` URIs).
- **MJCF** runs the interactive MuJoCo sim locally on Apple Silicon — the same
  engine used for TRON 1 RL work — with no GPU required.
- The `PatrolController` → `BraveBot.drive()` boundary is the same
  *strategy → navigation → locomotion* split a real deployment uses: a
  high-level planner sets goals, the robot handles motion.

See [`docs/BILL_OF_MATERIALS.md`](docs/BILL_OF_MATERIALS.md) for the full parts
and sensor tables.

## Licensing

This project is Apache-2.0 (see `LICENSE`). The LimX TRON 1 link meshes in
`description/meshes/tron1/` are © LimX Dynamics, redistributed unmodified under
Apache-2.0 — see [`NOTICE`](NOTICE). All BraveBot modification geometry is
original work.
