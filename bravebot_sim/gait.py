"""
Scripted, balance-assisted stepping gait for the BraveBot.

Honest scope: a 2-legged robot on a single wheel axle cannot statically support
itself the instant a foot leaves the ground, and lifting a wheel triggers the
uncontrolled roll mode. So this is NOT a true leg-driven dynamic walk — that
needs a learned policy (see `bravebot_sim/rl/`). Instead, the wheel balance
controller keeps the robot upright and provides translation, while the legs run
a visible alternating stepping cycle on top. The amplitude is kept inside the
range that does not destabilize balance (found empirically).

It gives the robot legged-looking locomotion now — march in place, step forward
/ backward, side-to-side (lateral lean + heading), and a step-turn — runnable on
a Mac with no training.
"""

from __future__ import annotations

import math

from .balance import STANCE


class Gait:
    """Generates alternating leg-stepping targets, phase-advanced over time."""

    def __init__(self, freq: float = 1.1, hip_amp: float = 0.05,
                 knee_amp: float = 0.08, abad_amp: float = 0.0):
        self.freq = freq
        self.hip_amp = hip_amp
        self.knee_amp = knee_amp      # bends the knee in swing -> foot lifts a touch
        self.abad_amp = abad_amp      # lateral sway for side-stepping
        self.phase = 0.0
        self.active = False

    def advance(self, dt: float, moving: bool):
        # only cycle the legs while actually walking; ease to a clean stance at rest
        if self.active or moving:
            self.phase += 2 * math.pi * self.freq * dt

    def leg_targets(self) -> dict:
        """Stepping targets for the 6 leg joints around the nominal stance."""
        ph = self.phase
        sL, sR = math.sin(ph), math.sin(ph + math.pi)
        return {
            "hip_L": STANCE["hip"] + self.hip_amp * sL,
            "knee_L": STANCE["knee"] - self.knee_amp * max(0.0, sL),
            "abad_L": self.abad_amp * sL,
            "hip_R": STANCE["hip"] + self.hip_amp * sR,
            "knee_R": STANCE["knee"] - self.knee_amp * max(0.0, sR),
            "abad_R": self.abad_amp * sR,
        }
