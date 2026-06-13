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


def test_curriculum_resume_offset():
    # Regression for the high-sev resume bug: a fresh env warms up (DR off), but an
    # env built with a large global_offset (what --resume bakes in) has DR fully on.
    from bravebot_sim.rl.env import BraveBotLocomotionEnv
    fresh = BraveBotLocomotionEnv(randomize=True)
    assert fresh._global == 0 and fresh._dr_ramp() == 0.0, "fresh run must warm up"
    resumed = BraveBotLocomotionEnv(randomize=True, global_offset=25_000_000 // 16)
    assert resumed._dr_ramp() == 1.0, "resumed run must continue DR (not restart at 0)"
    assert resumed._cmd_ramp() == 1.0


def test_turn_lock_directional():
    # Regression for the turn-governor bug: it must (a) stay upright under a sustained
    # same-direction max arc, and (b) NOT freeze — an opposite-direction turn after a
    # same-direction lock must still change the heading.
    from bravebot_sim import PhysicsBraveBot, physics_model_path
    bot = PhysicsBraveBot(physics_model_path())
    for _ in range(int(20 / 0.002)):                 # 20 s constant max arc
        bot.drive(0.6, 0.45)
        bot.step(0.002)
        assert not bot.state().fell, "fell on a sustained arc (turn governor unsafe)"
    bot = PhysicsBraveBot(physics_model_path())      # fresh: right, then left
    def seg(v, w, T):
        for _ in range(int(T / 0.002)):
            bot.drive(v, w); bot.step(0.002)
    seg(0.5, 0.45, 12); h_right = bot.yaw
    seg(0.5, 0.0, 6)
    seg(0.5, -0.45, 14)
    assert bot.yaw < h_right - 0.3, "opposite turn frozen (lock not directional)"
    assert not bot.state().fell


def test_heading_env():
    # The heading-aware env (the fix for the yaw drift) adds 2 obs (sin/cos of the
    # heading error vs the integrated yaw-rate command) and a heading-hold reward.
    from bravebot_sim.rl.heading_env import HeadingAwareEnv
    e = HeadingAwareEnv(randomize=False)
    o, _ = e.reset(seed=0)
    assert o.shape == e.observation_space.shape == (42,)
    assert e.action_space.shape == (9,)
    # straight command: desired heading stays put, heading error stays small
    e._cmd[:] = [0.4, 0, 0]
    for _ in range(20):
        o, r, term, trunc, _ = e.step(np.zeros(9, np.float32))
        assert np.isfinite(o).all() and math.isfinite(r)
        if term or trunc:
            break
    assert abs(e._yaw_des) < 1e-6, "straight command must not move the desired heading"
    # turn command: desired heading integrates the commanded yaw rate
    e._cmd[:] = [0, 0, 0.5]
    for _ in range(25):
        e.step(np.zeros(9, np.float32))
    assert e._yaw_des > 0.1, "turn command must advance the desired heading"
    # determinism: two fresh envs, same seed -> same initial obs
    a, _ = HeadingAwareEnv(randomize=False).reset(seed=0)
    b, _ = HeadingAwareEnv(randomize=False).reset(seed=0)
    assert np.allclose(a, b), "heading env reset(seed) not deterministic"


def test_onnx_artifacts_bounded():
    # Regression for the deployment fix: the shipped .onnx actors must clamp actions to
    # [-1,1] so they're safe drop-in artifacts on the real robot (the raw policy mean can
    # be out-of-range on OOD states). Skips gracefully if onnxruntime / the artifacts are
    # absent (e.g. a fresh clone without the RL extras).
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (skip: onnxruntime not installed)")
        return
    rl = os.path.join(ROOT, "bravebot_sim", "rl")
    checked = 0
    for name in ("policy.onnx", "policy_champion.onnx", "policy_heading.onnx"):
        path = os.path.join(rl, name)
        if not os.path.exists(path):
            continue
        sess = ort.InferenceSession(path)
        dim = sess.get_inputs()[0].shape[-1]
        dim = int(dim) if isinstance(dim, int) else (42 if "heading" in name else 40)
        rng = np.random.default_rng(0)
        for _ in range(80):
            o = rng.normal(0, 2.0, (1, dim)).astype(np.float32)   # incl. OOD states
            a = sess.run(None, {"obs": o})[0][0]
            assert a.min() >= -1.0001 and a.max() <= 1.0001, f"{name} emits out-of-range action {a}"
        checked += 1
    assert checked >= 1, "no .onnx artifacts found to check"


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
