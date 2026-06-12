"""
bravebot_sim — simulation library for the BraveBot wheel-legged inspection robot.

Quick start::

    from bravebot_sim import BraveBot, scene_path
    bot = BraveBot(scene_path())
    bot.drive(1.0, 0.2)
    bot.step(0.02)

The robot is a modified LimX Dynamics TRON 1 (real link meshes) carrying the
BraveBot four-sensor inspection payload (original geometry).
"""

from __future__ import annotations

import os

from .registry import COMPONENTS, SENSORS, SPECS, COMPONENT_BY_ID, SENSOR_COMPONENTS
from .sim import BraveBot, Anomaly, SensorReading
from .patrol import PatrolController, diagnose, Alert
from . import facility

__all__ = [
    "BraveBot", "Anomaly", "SensorReading", "PatrolController", "diagnose",
    "Alert", "facility", "COMPONENTS", "SENSORS", "SPECS", "COMPONENT_BY_ID",
    "SENSOR_COMPONENTS", "model_path", "scene_path",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))


def model_path() -> str:
    """Path to the bare robot MJCF."""
    return os.path.join(_ROOT, "description", "mjcf", "bravebot.xml")


def scene_path() -> str:
    """Path to the facility scene MJCF (builds it if missing)."""
    p = facility.SCENE_PATH
    if not os.path.exists(p):
        facility.build_scene_xml(p)
    return p
