#!/usr/bin/env python3
"""
Real-time BraveBot PHYSICS simulator, streamed to the browser.

Runs the actual MuJoCo rigid-body sim + the balance controller in a background
thread (500 Hz, real-time) — the robot balances on its wheels like the inverted
pendulum it is — and serves the live part poses to web/sim.html, which renders
them and sends back drive commands. So you drive a real-physics robot in 3D.

    python scripts/sim_server.py            # serve on :8001, open web/sim.html
    python scripts/sim_server.py 8002       # custom port

Endpoints (same origin, also serves the static page + meshes):
    GET  /state   -> {parts:[{id,pos,quat}], base, speed, upright, fell}
    POST /cmd     <- {v, w}     set drive (m/s, rad/s)
    POST /reset   <- {}         re-settle upright
"""
from __future__ import annotations

import json, os, sys, threading, time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from bravebot_sim import PhysicsBraveBot, physics_model_path  # noqa: E402

SIM_DT = 0.02            # 20 ms real-time chunk (= 10 physics substeps @ 500 Hz)


class Sim:
    def __init__(self):
        self.bot = PhysicsBraveBot(physics_model_path())
        self.bot.reset_upright()
        m = self.bot.model
        # precompute per visual-mesh geom: (geom id, part id, mesh_pos, R(mesh_quat)ᵀ)
        # so we can map raw-STL space -> world each frame (MuJoCo recenters meshes).
        self.geoms = []
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid)
            if not nm or not nm.startswith("m_"):
                continue
            Rmq = np.zeros(9); mujoco.mju_quat2Mat(Rmq, m.mesh_quat[mid])
            self.geoms.append((g, nm[2:], m.mesh_pos[mid].copy(), Rmq.reshape(3, 3).T.copy()))
        self.lock = threading.Lock()
        self.v = self.w = 0.0
        self.state = {}
        self._compute()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _compute(self):
        d = self.bot.data
        parts = []
        for g, pid, mp, RmqT in self.geoms:
            Rc = d.geom_xmat[g].reshape(3, 3) @ RmqT
            pos = d.geom_xpos[g] - Rc @ mp
            q = np.zeros(4); mujoco.mju_mat2Quat(q, Rc.flatten())
            parts.append({"id": pid, "pos": [round(float(x), 5) for x in pos],
                          "quat": [round(float(x), 6) for x in q]})
        s = self.bot.state()
        self.state = {"parts": parts, "base": [round(self.bot.x, 4), round(self.bot.y, 4)],
                      "yaw": round(self.bot.yaw, 4), "speed": round(float(s.v), 3),
                      "pitch": round(float(s.pitch), 3), "upright": not s.fell, "fell": s.fell}

    def _loop(self):
        nxt = time.time()
        while self.running:
            with self.lock:
                v, w = self.v, self.w
            self.bot.drive(v, w)
            self.bot.step(SIM_DT)
            if self.bot.fell:                 # auto-recover so the sim never gets stuck
                self.bot.reset_upright()
                with self.lock:
                    self.v = self.w = 0.0
            with self.lock:
                self._compute()
            nxt += SIM_DT
            time.sleep(max(0.0, nxt - time.time()))

    def set_cmd(self, v, w):
        with self.lock:
            self.v = float(np.clip(v, -1.2, 1.2)); self.w = float(np.clip(w, -1.0, 1.0))

    def reset(self):
        with self.lock:
            self.bot.reset_upright(); self.v = self.w = 0.0

    def snapshot(self):
        with self.lock:
            return dict(self.state)


SIM = None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/state"):
            self._json(SIM.snapshot())
        else:
            super().do_GET()

    def do_POST(self):
        n = int(self.headers.get("content-length", 0) or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except Exception:
            data = {}
        if self.path.startswith("/cmd"):
            SIM.set_cmd(data.get("v", 0.0), data.get("w", 0.0)); self._json({"ok": True})
        elif self.path.startswith("/reset"):
            SIM.reset(); self._json({"ok": True})
        else:
            self.send_error(404)


def main():
    global SIM
    SIM = Sim()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"BraveBot physics sim  ->  http://127.0.0.1:{port}/web/sim.html   (Ctrl-C to stop)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
