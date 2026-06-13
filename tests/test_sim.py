"""
Regression / smoke tests for the BraveBot sim.

Runnable two ways:
    python -m pytest tests/                 # if pytest is installed
    python tests/test_sim.py                # plain runner (no deps)

Covers: meshes present, both MJCF models compile with sane mass/CoM, the URDF
parses and is fully connected, the kinematic + physics robots load/drive, the
balance controller stays upright, the RL env has the right shapes + finite
rewards + deterministic resets, and the scripted gait steps without falling.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

import mujoco  # noqa: E402
from bravebot_sim import registry as R  # noqa: E402


def test_meshes_present():
    md = os.path.join(ROOT, "description", "meshes")
    for k, mesh in R.TRON1_MESH.items():
        assert os.path.exists(os.path.join(md, "tron1", f"{mesh}.stl")), mesh
    for c in R.COMPONENTS:
        assert os.path.exists(os.path.join(md, "bravebot", f"{c.id}.stl")), c.id


def test_models_compile():
    for name, nu_min in [("bravebot.xml", 2), ("bravebot_physics.xml", 9)]:
        m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "description", "mjcf", name))
        assert m.nu >= nu_min, (name, m.nu)
        if "physics" in name:
            mass = m.body_subtreemass[1]
            assert 30.0 < mass < 45.0, f"mass {mass}"        # ~37 kg real LimX + payload
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            assert np.isfinite(d.qpos).all()


def test_urdf_connected():
    import xml.etree.ElementTree as ET
    r = ET.parse(os.path.join(ROOT, "description", "urdf", "bravebot.urdf")).getroot()
    links = {l.get("name") for l in r.findall("link")}
    joints = r.findall("joint")
    assert len(links) == len(joints) + 1, "tree must be links = joints + 1"
    for j in joints:                                          # no dangling refs
        assert j.find("parent").get("link") in links
        assert j.find("child").get("link") in links


def test_kinematic_drive_and_scan():
    from bravebot_sim import BraveBot, scene_path, facility
    bot = BraveBot(scene_path())
    bot.x, bot.y = facility.DOCK
    bot.drive(0.8, 0.0)
    for _ in range(100):
        bot.step(0.02)
    assert bot.x > facility.DOCK[0] + 0.3, "should have driven forward"
    # somewhere on a full pass it should see an anomaly
    seen = 0
    for _ in range(300):
        bot.step(0.02)
        seen += len(bot.scan(facility.ANOMALIES))
    assert seen > 0, "scan never detected an anomaly"


def test_physics_balance():
    from bravebot_sim.balance import BalanceController, Gains, settle_upright
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "description", "mjcf", "bravebot_physics.xml"))
    d = mujoco.MjData(m)
    c = BalanceController(m, d, Gains())
    settle_upright(m, d, c)
    for k in range(int(8 / m.opt.timestep)):
        s = c.control(0.0, 0.0)
        mujoco.mj_step(m, d)
        assert not s.fell, f"balance controller fell at {k * m.opt.timestep:.1f}s"


def test_rl_env_shapes_and_determinism():
    from bravebot_sim.rl.env import BraveBotLocomotionEnv
    e = BraveBotLocomotionEnv(randomize=False)
    o1, _ = e.reset(seed=0)
    assert o1.shape == e.observation_space.shape == (40,)
    assert e.action_space.shape == (9,)
    r_sum = 0.0
    for _ in range(30):
        o, r, term, trunc, _ = e.step(np.zeros(9, np.float32))
        assert np.isfinite(o).all() and math.isfinite(r)
        r_sum += r
        if term or trunc:
            break
    # determinism: same seed -> same first obs
    o2, _ = e.reset(seed=0)
    assert np.allclose(o1, o2), "reset(seed) not deterministic"


def test_gait_walks():
    from bravebot_sim import PhysicsBraveBot, physics_model_path
    bot = PhysicsBraveBot(physics_model_path())
    bot.set_walking(True)
    bot.drive(0.3, 0.0)
    for _ in range(int(4 / 0.02)):
        bot.step(0.02)
        assert not bot.state().fell, "gait fell"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            fails += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    return fails


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
