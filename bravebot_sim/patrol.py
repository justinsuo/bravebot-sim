"""
Waypoint patrol controller + a tiny edge-AI fusion stub.

`PatrolController` turns the BraveBot toward each waypoint and drives to it — a
stand-in for the Nav2 / locomotion layers on the real robot. `diagnose()`
mimics the on-edge mixture-of-experts: it groups sensor readings, picks the
highest-confidence finding and emits a human-readable, risk-scored alert in the
same shape as the product site's sample alert.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .sim import BraveBot, SensorReading


@dataclass
class Alert:
    zone: str
    finding: str
    first_seen: str
    risk: str
    confidence: float
    action: str

    def render(self) -> str:
        return (f"ALERT · {self.zone}\n"
                f"  finding:    {self.finding}\n"
                f"  first seen: {self.first_seen}\n"
                f"  risk:       {self.risk}  (confidence {self.confidence:.0%})\n"
                f"  action:     {self.action}")


_ACTIONS = {
    "acoustic": "isolate line, inspect connector/switchgear, dispatch technician",
    "thermal": "verify airflow/termination, rebalance setpoints, schedule survey",
    "gas": "isolate affected string, increase ventilation, dispatch battery tech",
    "visual": "close/latch door or seal, log offender for facilities review",
}


def diagnose(readings: list[SensorReading]) -> Alert | None:
    """Edge-AI fusion stub: highest-confidence reading -> risk-scored alert."""
    if not readings:
        return None
    top = max(readings, key=lambda r: r.confidence)
    risk = "high" if top.confidence > 0.7 else "moderate" if top.confidence > 0.45 else "low"
    return Alert(
        zone=top.target or "unknown",
        finding=top.note,
        first_seen=f"{top.modality} · {top.confidence:.0%} conf, {top.distance:.1f} m",
        risk=risk,
        confidence=top.confidence,
        action=_ACTIONS.get(top.modality, "review and log"),
    )


class PatrolController:
    def __init__(self, bot: BraveBot, waypoints, reach=0.4,
                 cruise=0.85, turn_gain=1.2, turn_rate=0.35, align_tol=0.35):
        self.bot = bot
        self.waypoints = list(waypoints)
        self.reach = reach
        self.cruise = cruise
        self.turn_gain = turn_gain
        self.turn_rate = turn_rate     # in-place turn speed (rad/s)
        self.align_tol = align_tol     # heading error (rad) to turn in place vs drive
        self.idx = 0
        self.done = False

    @property
    def target(self):
        return self.waypoints[self.idx]

    def update(self):
        """Compute drive command toward the current waypoint. Call before step()."""
        if self.done:
            self.bot.drive(0.0, 0.0)
            return
        tx, ty = self.target
        dx, dy = tx - self.bot.x, ty - self.bot.y
        dist = math.hypot(dx, dy)
        if dist < self.reach:
            self.idx += 1
            if self.idx >= len(self.waypoints):
                self.done = True
                self.bot.drive(0.0, 0.0)
                return
            tx, ty = self.target
            dx, dy = tx - self.bot.x, ty - self.bot.y
            dist = math.hypot(dx, dy)
        heading_err = _wrap(math.atan2(dy, dx) - self.bot.yaw)
        rev_err = _wrap(heading_err - math.pi)   # heading error if we drive backward
        # A single-axle balancer can't safely make large turns (no roll actuation),
        # so never U-turn: if the waypoint is behind, REVERSE toward it. Only the
        # nearer of forward/backward is used, so the robot turns at most ~90°, and
        # on a straight aisle it just drives forward then back — no turning at all.
        if abs(heading_err) <= abs(rev_err):
            err, vdir = heading_err, 1.0
        else:
            err, vdir = rev_err, -1.0
        if abs(err) > self.align_tol:
            v, omega = 0.0, math.copysign(self.turn_rate, err)   # turn in place, gently
        else:
            v, omega = vdir * self.cruise * math.cos(err), self.turn_gain * err
        self.bot.drive(v, omega)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
