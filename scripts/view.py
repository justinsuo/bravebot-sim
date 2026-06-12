#!/usr/bin/env python3
"""
Interactive BraveBot viewer (MuJoCo passive viewer).

By default the robot runs an autonomous balancing patrol — just watch (and orbit
the camera with the mouse). The ARROW KEYS take manual control.

    python scripts/view.py                 # kinematic robot in the facility
    python scripts/view.py --physics       # REAL physics: it balances on its wheels
    python scripts/view.py --bare          # robot only, no facility
    python scripts/view.py --check         # headless self-test (no window)

Controls (focus the MuJoCo window):
    Up / Down     drive forward / back        Left / Right   turn
    Enter         resume autonomous patrol (and stop)

Anomalies are scanned continuously and printed as the robot sees them.

NOTE: the MuJoCo viewer binds every LETTER key to a visualization toggle
(W=wireframe, S=shadow, ...), so robot control uses the arrow keys, which the
viewer leaves free. macOS: launch with `mjpython scripts/view.py ...`.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from bravebot_sim import (BraveBot, PatrolController, diagnose,  # noqa: E402
                          model_path, scene_path, facility)

# GLFW key codes — arrows + enter are the only keys the MuJoCo viewer leaves free
RIGHT, LEFT, DOWN, UP = 262, 263, 264, 265
ENTER = 257
V_STEP, W_STEP = 0.3, 0.2     # per arrow press (gentle; commands slew to target)


class Session:
    def __init__(self, bare: bool, physics: bool = False, walk: bool = False):
        self.bare = bare
        self.physics = physics or walk
        if self.physics:
            from bravebot_sim import PhysicsBraveBot, physics_model_path, physics_scene_path
            self.bot = PhysicsBraveBot(physics_model_path() if bare else physics_scene_path())
            if walk:
                self.bot.set_walking(True)
        else:
            self.bot = BraveBot(model_path() if bare else scene_path())
            self.bot.x, self.bot.y = (0.0, 0.0) if bare else facility.DOCK
            self.bot.apply()
        self.ctrl = PatrolController(self.bot, facility.WAYPOINTS)
        self.autopilot = not bare        # patrol by default in the facility
        self.v_cmd = 0.0
        self.omega_cmd = 0.0
        self._alerted: set[str] = set()

    # ---- input ----
    def on_key(self, keycode: int):
        if keycode == UP:
            self.v_cmd = min(self.bot.MAX_V, self.v_cmd + V_STEP); self.autopilot = False
        elif keycode == DOWN:
            self.v_cmd = max(-self.bot.MAX_V, self.v_cmd - V_STEP); self.autopilot = False
        elif keycode == LEFT:
            self.omega_cmd = min(self.bot.MAX_OMEGA, self.omega_cmd + W_STEP); self.autopilot = False
        elif keycode == RIGHT:
            self.omega_cmd = max(-self.bot.MAX_OMEGA, self.omega_cmd - W_STEP); self.autopilot = False
        elif keycode == ENTER:
            self.v_cmd = self.omega_cmd = 0.0
            self.autopilot = not self.bare
            self.ctrl = PatrolController(self.bot, facility.WAYPOINTS)
            print("resumed autonomous patrol" if self.autopilot else "manual hold")

    # ---- loop ----
    def tick(self, dt: float):
        if self.autopilot:
            self.ctrl.update()
            if self.ctrl.done:
                self.ctrl = PatrolController(self.bot, facility.WAYPOINTS)   # loop the route
        else:
            self.bot.drive(self.v_cmd, self.omega_cmd)
        self.bot.step(dt)
        self._auto_scan()

    def _auto_scan(self):
        if self.bare:
            return
        for r in self.bot.scan(facility.ANOMALIES):
            if r.confidence > 0.5 and r.target not in self._alerted:
                self._alerted.add(r.target)
                alert = diagnose([r])
                print(f"\n[scan] {r.modality} array detected:")
                print("  " + alert.render().replace("\n", "\n  "))


def run(bare: bool, physics: bool, walk: bool = False):
    import mujoco.viewer
    sess = Session(bare, physics, walk)
    dt = 0.02
    banner = ("WALK mode — balance-assisted legged stepping gait.\n" if walk else
              "PHYSICS mode — the robot balances on its wheels.\n" if physics else "")
    print(banner + (__doc__.split("Controls")[1] if "Controls" in __doc__ else ""))
    with mujoco.viewer.launch_passive(sess.bot.model, sess.bot.data,
                                      key_callback=sess.on_key) as viewer:
        while viewer.is_running():
            t0 = time.time()
            sess.tick(dt)
            viewer.sync()
            time.sleep(max(0.0, dt - (time.time() - t0)))


def check(physics: bool = False):
    """Headless validation of the control loop — opens no window."""
    sess = Session(bare=False, physics=physics)
    for _ in range(600):           # ~12 s of autonomous patrol
        sess.tick(0.02)
    for code in (UP, UP, LEFT, ENTER):   # exercise manual + resume
        sess.on_key(code)
    for _ in range(150):
        sess.tick(0.02)
    upright = (not sess.bot.fell) if physics else True
    print(f"check OK ({'physics' if physics else 'kinematic'}): "
          f"x={sess.bot.x:.2f} y={sess.bot.y:.2f} upright={upright} "
          f"alerts={len(sess._alerted)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bare", action="store_true", help="robot only, no facility")
    ap.add_argument("--physics", action="store_true", help="real rigid-body dynamics + balance")
    ap.add_argument("--walk", action="store_true", help="balance-assisted legged stepping gait")
    ap.add_argument("--check", action="store_true", help="headless self-test")
    args = ap.parse_args()
    if args.check:
        check(args.physics or args.walk)
    else:
        run(args.bare, args.physics, args.walk)
