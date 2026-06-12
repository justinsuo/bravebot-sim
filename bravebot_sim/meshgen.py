"""
Procedural mesh generation for the BraveBot modification components.

Every non-LimX part is an original multi-primitive assembly. The primitive
specs are transcribed 1:1 from the product site's Three.js part library
(`bravebot-site/src/components/landing/RobotParts.tsx`) so the simulation model
matches the marketing renders exactly.

Conventions (mirroring the Three.js source):
  * coordinates are URDF-local: +x forward, +y left, +z up
  * a cylinder's default axis is +y (Three.js convention); RZ -> +x, RX -> +z
  * `rbox` (rounded box) is approximated as a plain box for the sim mesh

Run `python -m bravebot_sim.meshgen` (or scripts/build_meshes.py) to export one
STL per component into description/meshes/bravebot/.
"""

from __future__ import annotations

import math
import os
from typing import Sequence

import numpy as np
import trimesh

HALF_PI = math.pi / 2
RZ = (0.0, 0.0, HALF_PI)     # cylinder axis -> +x
RX = (HALF_PI, 0.0, 0.0)     # cylinder axis -> +z
RY = (0.0, HALF_PI, 0.0)     # used for torus bezels

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "description", "meshes", "bravebot"))


# --------------------------------------------------------------------------- #
#  Primitive builder                                                          #
# --------------------------------------------------------------------------- #

def _euler_matrix(r: Sequence[float]) -> np.ndarray:
    """Single-axis Euler rotations (only kind used) -> 4x4 matrix."""
    m = np.eye(4)
    for axis, ang in zip("xyz", r):
        if abs(ang) < 1e-12:
            continue
        m = trimesh.transformations.rotation_matrix(ang, {"x": [1, 0, 0],
                                                           "y": [0, 1, 0],
                                                           "z": [0, 0, 1]}[axis]) @ m
    return m


def _prim(geom: str, a: Sequence[float], p=(0, 0, 0), r=(0, 0, 0)) -> trimesh.Trimesh:
    if geom in ("box", "rbox"):
        mesh = trimesh.creation.box(extents=[a[0], a[1], a[2]])
    elif geom == "cyl":
        # a = [radiusTop, radiusBottom, height, segments]; baseline axis +y.
        radius = (a[0] + a[1]) / 2.0
        height = a[2]
        sections = int(a[3]) if len(a) > 3 else 24
        mesh = trimesh.creation.cylinder(radius=radius, height=height,
                                         sections=max(3, sections))
        # trimesh cylinder is +z; rotate to +y to match the Three.js default.
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-HALF_PI, [1, 0, 0]))
    elif geom == "sph":
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=a[0])
    elif geom == "tor":
        # a = [radius, tube, radialSeg, tubularSeg]; lies in XY plane, axis +z.
        mesh = trimesh.creation.torus(major_radius=a[0], minor_radius=a[1],
                                      major_sections=max(8, int(a[3]) if len(a) > 3 else 28),
                                      minor_sections=max(6, int(a[2]) if len(a) > 2 else 10))
    else:
        raise ValueError(f"unknown geom {geom}")

    if any(abs(v) > 1e-12 for v in r):
        mesh.apply_transform(_euler_matrix(r))
    mesh.apply_translation(p)
    return mesh


def _bolt(p, r=RZ, s=0.009):
    return _prim("cyl", [s, s, s * 1.3, 6], p=p, r=r)


def _camera(lens: float):
    """Camera assembly facing +x (used by thermal & HD cameras)."""
    return [
        _prim("box", [0.056, 0.066, 0.064], p=[-0.014, 0, 0]),
        _prim("cyl", [lens + 0.012, lens + 0.014, 0.018, 28], p=[0.014, 0, 0], r=RZ),
        _prim("cyl", [lens + 0.013, lens + 0.013, 0.01, 28], p=[0.026, 0, 0], r=RZ),
        _prim("cyl", [lens + 0.007, lens + 0.009, 0.024, 28], p=[0.04, 0, 0], r=RZ),
        _prim("cyl", [lens, lens, 0.012, 28], p=[0.052, 0, 0], r=RZ),
        _prim("cyl", [lens * 0.5, lens * 0.5, 0.006, 20], p=[0.057, 0, 0], r=RZ),
        _prim("tor", [lens + 0.006, 0.004, 8, 28], p=[0.05, 0, 0], r=RY),
    ]


# --------------------------------------------------------------------------- #
#  Per-component primitive assemblies                                          #
# --------------------------------------------------------------------------- #

def _payload():
    parts = [
        _prim("box", [0.3, 0.36, 0.034]),
        _prim("box", [0.3, 0.032, 0.05], p=[0, 0.168, 0.02]),
        _prim("box", [0.3, 0.032, 0.05], p=[0, -0.168, 0.02]),
        _prim("box", [0.05, 0.34, 0.026], p=[0.09, 0, 0.014]),
        _prim("box", [0.05, 0.34, 0.026], p=[-0.09, 0, 0.014]),
    ]
    for x, y in [(0.13, 0.16), (-0.13, 0.16), (0.13, -0.16), (-0.13, -0.16)]:
        parts.append(_bolt([x, y, 0.026], r=RX, s=0.012))
    return parts


def _torso():
    parts = [
        _prim("box", [0.27, 0.3, 0.24]),
        _prim("box", [0.23, 0.26, 0.045], p=[0, 0, 0.135]),
        _prim("box", [0.29, 0.32, 0.03], p=[0, 0, -0.12]),
    ]
    for i in range(5):
        parts.append(_prim("box", [0.17, 0.006, 0.011], p=[0, 0.152, 0.058 - i * 0.026]))
    parts += [
        _prim("box", [0.272, 0.006, 0.006], p=[0, 0, 0.03]),
        _prim("box", [0.012, 0.13, 0.13], p=[-0.138, 0.04, -0.02]),
        _prim("cyl", [0.014, 0.014, 0.02, 16], p=[0.138, -0.08, 0.06], r=RZ),
    ]
    for y, z in [(0.11, 0.085), (-0.11, 0.085)]:
        parts.append(_bolt([0.137, y, z], r=RZ))
    return parts


def _battery():
    parts = [
        _prim("box", [0.16, 0.24, 0.1]),
        _prim("box", [0.166, 0.034, 0.066], p=[0, 0, -0.014]),
    ]
    for i in range(3):
        parts.append(_prim("box", [0.006, 0.2, 0.07], p=[0.083, 0, -0.022 + i * 0.024]))
    parts += [
        _prim("box", [0.022, 0.022, 0.042], p=[0.052, 0, 0.07]),
        _prim("box", [0.022, 0.022, 0.042], p=[-0.052, 0, 0.07]),
        _prim("cyl", [0.012, 0.012, 0.13, 16], p=[0, 0, 0.092], r=RZ),
        _prim("cyl", [0.013, 0.013, 0.02, 18], p=[0.05, 0.072, 0.05], r=RX),
        _prim("cyl", [0.013, 0.013, 0.02, 18], p=[-0.05, 0.072, 0.05], r=RX),
    ]
    for i in range(3):
        parts.append(_prim("cyl", [0.005, 0.005, 0.012, 12], p=[0.082, -0.04 + i * 0.03, 0.03], r=RZ))
    return parts


def _edgeai():
    parts = [_prim("box", [0.13, 0.2, 0.08])]
    for i in range(9):
        parts.append(_prim("box", [0.108, 0.008, 0.052], p=[0, -0.076 + i * 0.019, 0.064]))
    parts += [
        _prim("box", [0.03, 0.07, 0.03], p=[0.082, 0, -0.018]),
        _prim("box", [0.018, 0.03, 0.018], p=[0.082, 0.05, 0.02]),
        _prim("cyl", [0.005, 0.005, 0.012, 12], p=[0.068, 0.07, 0.024], r=RZ),
    ]
    return parts


def _display():
    parts = [
        _prim("box", [0.03, 0.21, 0.15]),
        _prim("box", [0.012, 0.17, 0.108], p=[0.014, 0, 0.012]),
        _prim("box", [0.006, 0.165, 0.103], p=[0.02, 0, 0.012]),
        _prim("cyl", [0.01, 0.01, 0.016, 18], p=[0.018, 0.07, -0.056], r=RZ),
        _prim("cyl", [0.01, 0.01, 0.016, 18], p=[0.018, -0.07, -0.056], r=RZ),
        _prim("cyl", [0.005, 0.005, 0.012, 12], p=[0.018, 0, -0.058], r=RZ),
    ]
    for x, y in [(0.095, 0.06), (-0.095, 0.06), (0.095, -0.06), (-0.095, -0.06)]:
        parts.append(_bolt([0.015, x, y], s=0.007))
    return parts


def _rear():
    parts = [
        _prim("box", [0.024, 0.22, 0.2]),
        _prim("box", [0.014, 0.09, 0.03], p=[0.02, 0, 0.04]),
        _prim("cyl", [0.011, 0.011, 0.07, 16], p=[0.03, 0, 0.04], r=RZ),
    ]
    for i in range(3):
        parts.append(_prim("box", [0.012, 0.13, 0.006], p=[0.014, 0, -0.05 + i * 0.022]))
    parts += [
        _prim("box", [0.016, 0.026, 0.022], p=[0.016, 0.092, 0.07]),
        _prim("box", [0.016, 0.026, 0.022], p=[0.016, -0.092, 0.07]),
    ]
    for x, y in [(0.09, 0.085), (-0.09, 0.085), (0.09, -0.085), (-0.09, -0.085)]:
        parts.append(_bolt([0.015, x, y], s=0.008))
    return parts


def _estop():
    return [
        _prim("cyl", [0.032, 0.036, 0.016, 28], p=[0, 0, -0.014], r=RX),
        _prim("cyl", [0.03, 0.03, 0.008, 28], p=[0, 0, -0.002], r=RX),
        _prim("cyl", [0.022, 0.026, 0.014, 24], p=[0, 0, 0.008], r=RX),
        _prim("cyl", [0.03, 0.024, 0.012, 24], p=[0, 0, 0.02], r=RX),
        _prim("cyl", [0.03, 0.03, 0.006, 24], p=[0, 0, 0.028], r=RX),
    ]


def _mast():
    parts = [
        _prim("box", [0.05, 0.05, 0.17]),
        _prim("cyl", [0.054, 0.054, 0.022, 24], p=[0, 0, -0.086], r=RX),
        _prim("cyl", [0.037, 0.037, 0.013, 24], p=[0, 0, 0.044], r=RX),
        _prim("cyl", [0.037, 0.037, 0.013, 24], p=[0, 0, -0.03], r=RX),
        _prim("box", [0.012, 0.014, 0.15], p=[0.03, 0, 0.01]),
    ]
    for x, y in [(0.018, 0.018), (-0.018, 0.018), (0.018, -0.018), (-0.018, -0.018)]:
        parts.append(_bolt([x, y, -0.092], r=RX, s=0.006))
    return parts


def _head():
    parts = [
        _prim("box", [0.21, 0.25, 0.13]),
        _prim("box", [0.03, 0.215, 0.105], p=[0.1, 0, 0]),
        _prim("box", [0.17, 0.215, 0.026], p=[0, 0, 0.082]),
        _prim("box", [0.06, 0.006, 0.04], p=[0, 0.128, 0.02]),
    ]
    for x, y in [(0.092, 0.04), (-0.092, 0.04), (0.092, -0.04), (-0.092, -0.04)]:
        parts.append(_bolt([0.114, x, y], s=0.007))
    return parts


def _acoustic():
    parts = [
        _prim("cyl", [0.054, 0.054, 0.024, 32], r=RZ),
        _prim("tor", [0.05, 0.006, 10, 32], p=[0.012, 0, 0], r=RY),
        _prim("cyl", [0.008, 0.008, 0.012, 14], p=[0.015, 0, 0], r=RZ),
    ]
    for rad, cnt in ((0.022, 6), (0.04, 12)):
        for k in range(cnt):
            ang = (k / cnt) * math.tau
            parts.append(_prim("cyl", [0.005, 0.005, 0.012, 10],
                               p=[0.015, math.cos(ang) * rad, math.sin(ang) * rad], r=RZ))
    return parts


def _gas():
    parts = [
        _prim("box", [0.05, 0.07, 0.05]),
        _prim("cyl", [0.026, 0.028, 0.016, 24], p=[0.03, 0, 0], r=RZ),
        _prim("cyl", [0.024, 0.024, 0.012, 24], p=[0.04, 0, 0], r=RZ),
    ]
    for i in (-2, -1, 0, 1, 2):
        parts.append(_prim("box", [0.008, 0.04, 0.004], p=[0.047, 0, i * 0.009]))
    for i in (-1, 0, 1):
        parts.append(_prim("box", [0.008, 0.004, 0.04], p=[0.047, i * 0.012, 0]))
    parts += [
        _prim("cyl", [0.005, 0.005, 0.01, 10], p=[0.026, 0.03, 0.02], r=RZ),
        _prim("cyl", [0.007, 0.007, 0.016, 14], p=[0.01, -0.038, 0], r=RX),
    ]
    return parts


def _nav():
    parts = [
        _prim("box", [0.06, 0.16, 0.04]),
        _prim("cyl", [0.032, 0.034, 0.012, 28], p=[0, 0, 0.024], r=RX),
        _prim("cyl", [0.03, 0.03, 0.03, 28], p=[0, 0, 0.046], r=RX),
        _prim("cyl", [0.034, 0.026, 0.012, 28], p=[0, 0, 0.064], r=RX),
        _prim("cyl", [0.008, 0.008, 0.014, 12], p=[0, 0, 0.074], r=RX),
    ]
    for y in (0.05, -0.05):
        parts.append(_prim("cyl", [0.015, 0.016, 0.012, 18], p=[0.03, y, -0.004], r=RZ))
        parts.append(_prim("cyl", [0.011, 0.011, 0.012, 18], p=[0.038, y, -0.004], r=RZ))
    return parts


def _antenna():
    return [
        _prim("cyl", [0.015, 0.018, 0.014, 18], p=[0, 0, -0.052], r=RX),
        _prim("cyl", [0.011, 0.011, 0.02, 18], p=[0, 0, -0.036], r=RX),
        _prim("cyl", [0.0055, 0.0055, 0.12, 14], p=[0, 0, 0.012], r=RX),
        _prim("sph", [0.011, 16, 14], p=[0, 0, 0.074]),
    ]


def _rail():
    parts = [
        _prim("cyl", [0.019, 0.019, 0.34, 20], r=RX),
        _prim("sph", [0.021, 16, 14], p=[0, 0, 0.17]),
        _prim("sph", [0.021, 16, 14], p=[0, 0, -0.17]),
    ]
    for z in (0.1, -0.1):
        parts.append(_prim("box", [0.06, 0.04, 0.034], p=[-0.03, 0, z]))
        parts.append(_bolt([-0.058, 0, z], r=RZ, s=0.007))
    return parts


PART_PRIMITIVES = {
    "payload": _payload, "torso": _torso, "battery": _battery, "edgeai": _edgeai,
    "display": _display, "rear": _rear, "estop": _estop, "mast": _mast, "head": _head,
    "acoustic": _acoustic, "thermal": lambda: _camera(0.022), "hdcam": lambda: _camera(0.031),
    "gas": _gas, "nav": _nav, "antennaL": _antenna, "antennaR": _antenna,
    "railL": _rail, "railR": _rail,
}


def build_part(part_id: str) -> trimesh.Trimesh:
    """Concatenate a component's primitives into a single watertight-ish mesh."""
    parts = PART_PRIMITIVES[part_id]()
    mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices()
    return mesh


def build_all(out_dir: str = OUT_DIR, verbose: bool = True) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for part_id in PART_PRIMITIVES:
        mesh = build_part(part_id)
        path = os.path.join(out_dir, f"{part_id}.stl")
        mesh.export(path)
        written[part_id] = path
        if verbose:
            ext = np.round(mesh.extents, 3)
            print(f"  {part_id:10s} -> {os.path.basename(path):14s} "
                  f"{len(mesh.faces):5d} tris  bbox={ext}")
    return written


if __name__ == "__main__":
    print(f"Generating BraveBot component meshes -> {OUT_DIR}")
    build_all()
    print("done.")
