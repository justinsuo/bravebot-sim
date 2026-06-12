#!/usr/bin/env python3
"""
Interactive BraveBot viewer (MuJoCo passive viewer).

Drive the robot around the data-center scene, change stance, run an autonomous
patrol, and scan for anomalies with the four-sensor stack.

    python scripts/view.py                 # open the interactive window
    python scripts/view.py --bare          # robot only, no facility
    python scripts/view.py --check         # headless self-test (no window)

Controls (focus the MuJoCo window):
    W / S    drive forward / back            A / D   turn left / right
    X        full stop                       Q / E   stance lower / taller
    SPACE    scan now -> print alerts        P       toggle autonomous patrol
    R        reset to the charging dock       O       print robot odometry
Mouse orbits/zooms the camera as usual.
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
from bravebot_sim.sim import STANCES  # noqa: E402

# GLFW key codes
W, A, S, D, X, Q, E, P, R, O = 87, 65, 83, 68, 88, 81, 69, 80, 82, 79
SPACE = 32
STANCE_ORDER = ["low", "nominal", "tall"]


class Session:
    def __init__(self, bare: bool):
        self.bare = bare
        self.bot = BraveBot(model_path() if bare else scene_path())
        self.bot.x, self.bot.y = (0.0, 0.0) if bare else facility.DOCK
        self.bot.apply()
        self.ctrl = PatrolController(self.bot, facility.WAYPOINTS)
        self.autopilot = False
        self.v_cmd = 0.0
        self.omega_cmd = 0.0
        self.stance_i = STANCE_ORDER.index("nominal")

    def on_key(self, keycode: int):
        b = self.bot
        if keycode == W:
            self.v_cmd = min(b.MAX_V, self.v_cmd + 0.3); self.autopilot = False
        elif keycode == S:
            self.v_cmd = max(-b.MAX_V, self.v_cmd - 0.3); self.autopilot = False
        elif keycode == A:
            self.omega_cmd = min(b.MAX_OMEGA, self.omega_cmd + 0.25); self.autopilot = False
        elif keycode == D:
            self.omega_cmd = max(-b.MAX_OMEGA, self.omega_cmd - 0.25); self.autopilot = False
        elif keycode == X:
            self.v_cmd = self.omega_cmd = 0.0; self.autopilot = False
        elif keycode in (Q, E):
            self.stance_i = max(0, min(len(STANCE_ORDER) - 1,
                                       self.stance_i + (1 if keycode == E else -1)))
            b.set_stance(STANCE_ORDER[self.stance_i])
            print(f"stance -> {STANCE_ORDER[self.stance_i]}")
        elif keycode == SPACE:
            self._scan()
        elif keycode == P:
            self.autopilot = not self.autopilot
            print(f"autopilot {'ON — patrolling route' if self.autopilot else 'OFF'}")
            if self.autopilot:
                self.ctrl = PatrolController(self.bot, facility.WAYPOINTS)
        elif keycode == R:
            b.x, b.y, b.yaw = (*facility.DOCK, 0.0)
            self.v_cmd = self.omega_cmd = 0.0; self.autopilot = False
            b.apply(); print("reset to dock")
        elif keycode == O:
            print(f"odom  x={b.x:+.2f} y={b.y:+.2f} yaw={math.degrees(b.yaw):+.0f}deg "
                  f"v={b.v:+.2f} stance={b.stance}")

    def _scan(self):
        readings = self.bot.scan(facility.ANOMALIES)
        if not readings:
            print("scan: nothing in sensor range / FOV")
            return
        print(f"\nscan @ x={self.bot.x:.1f} y={self.bot.y:.1f}: "
              f"{len(readings)} reading(s)")
        for r in readings:
            print(f"  {r.modality:8s} {r.confidence:4.0%}  {r.target}  ({r.distance:.1f} m)")
        alert = diagnose(readings)
        if alert and alert.confidence > 0.45:
            print("  --\n  " + alert.render().replace("\n", "\n  "))

    def tick(self, dt: float):
        if self.autopilot:
            self.ctrl.update()
            if self.ctrl.done:
                self.autopilot = False
                print("patrol complete")
        else:
            self.bot.drive(self.v_cmd, self.omega_cmd)
        self.bot.step(dt)


def run(bare: bool):
    import mujoco.viewer
    sess = Session(bare)
    dt = 0.02
    print(__doc__.split("Controls")[1] if "Controls" in __doc__ else "")
    with mujoco.viewer.launch_passive(sess.bot.model, sess.bot.data,
                                      key_callback=sess.on_key) as viewer:
        while viewer.is_running():
            t0 = time.time()
            sess.tick(dt)
            viewer.sync()
            time.sleep(max(0.0, dt - (time.time() - t0)))


def check():
    """Headless validation of the control loop — opens no window."""
    sess = Session(bare=False)
    for code in (W, W, A, SPACE, E, P):
        sess.on_key(code)
    for _ in range(400):
        sess.tick(0.02)
    print(f"check OK: drove to x={sess.bot.x:.2f} y={sess.bot.y:.2f}, "
          f"patrol idx={sess.ctrl.idx}, stance={sess.bot.stance}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bare", action="store_true", help="robot only, no facility")
    ap.add_argument("--check", action="store_true", help="headless self-test")
    args = ap.parse_args()
    if args.check:
        check()
    else:
        run(args.bare)
