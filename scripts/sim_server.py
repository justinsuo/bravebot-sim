#!/usr/bin/env python3
"""
Real-time BraveBot PHYSICS simulator, streamed to the browser (web/sim.html).

Runs the actual MuJoCo rigid-body sim in a background thread (real-time) and serves
live poses + a playground (ramps/bumps/pillars). Two controllers you can switch
between live: the hand-tuned BALANCE controller, or the trained RL POLICY. You can
shove the robot with the mouse to test balance recovery.

    python scripts/sim_server.py            # serve on :8001, open web/sim.html

Endpoints (same origin; also serves the static page + meshes):
    GET  /state   -> {parts, base, yaw, speed, pitch, upright, fell, mode}
    GET  /scene   -> {obstacles:[{type,size,pos,quat,color}]}   (static, fetched once)
    POST /cmd     <- {v, w}            drive (m/s, rad/s)
    POST /push    <- {fx, fy, mag}     external shove on the torso (N)
    POST /mode    <- {mode:'balance'|'rl'}
    POST /reset   <- {}
"""
from __future__ import annotations

import json, os, sys, tempfile, threading, time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from bravebot_sim import PhysicsBraveBot, physics_model_path           # noqa: E402
from bravebot_sim.rl.env import BraveBotLocomotionEnv                  # noqa: E402

SIM_DT = 0.02            # 20 ms real-time chunk (= one 50 Hz control step / 10 substeps)
PUSH_S = 0.15            # how long a mouse-shove force is applied

# A small playground the robot can drive around: gentle bumps, a shallow ramp, pillars.
OBSTACLES = """
    <geom name="obs_bump1"  type="box" pos="1.1 0 0.025"  size="0.12 0.7 0.025" rgba="0.85 0.55 0.25 1"/>
    <geom name="obs_bump2"  type="box" pos="-0.7 0.7 0.03" size="0.6 0.12 0.03"  rgba="0.85 0.55 0.25 1"/>
    <geom name="obs_ramp"   type="box" pos="-2.3 0 0.105" quat="0.99813 0 0.06105 0" size="0.75 0.6 0.02" rgba="0.85 0.55 0.25 1"/>
    <geom name="obs_pillarA" type="cylinder" pos="0.8 1.2 0.45"  size="0.09 0.45" rgba="0.46 0.5 0.58 1"/>
    <geom name="obs_pillarB" type="cylinder" pos="-1.3 -1.1 0.45" size="0.09 0.45" rgba="0.46 0.5 0.58 1"/>
    <geom name="obs_pillarC" type="cylinder" pos="1.8 -1.4 0.45"  size="0.09 0.45" rgba="0.46 0.5 0.58 1"/>
"""


def playground_xml():
    """bravebot_physics.xml + static obstacle geoms (collide with the robot + floor).

    Written into the mjcf/ dir (next to the original) so its relative meshdir resolves.
    """
    src = open(physics_model_path()).read()
    inject = "".join(  # give each obstacle collision bits that hit the robot + floor
        line.replace("/>", ' contype="1" conaffinity="1"/>')
        for line in OBSTACLES.strip().splitlines())
    path = os.path.join(ROOT, "description", "mjcf", "_playground.xml")
    with open(path, "w") as f:
        f.write(src.replace("</worldbody>", inject + "\n  </worldbody>"))
    return path


class Sim:
    def __init__(self):
        pg = playground_xml()
        self.bot = PhysicsBraveBot(pg)                     # balance-controller robot
        self.bot.reset_upright()
        self.env = BraveBotLocomotionEnv(randomize=False, episode_s=1e9, model_path=pg)  # RL robot
        self.obs, _ = self.env.reset(seed=0)
        from stable_baselines3 import PPO
        self.policy = PPO.load(os.path.join(ROOT, "bravebot_sim", "rl", "policy_champion"), device="cpu")
        self.mode = "balance"
        m = self.bot.model
        self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        # per visual-mesh geom: (geom id, part id, mesh_pos, R(mesh_quat)ᵀ) for raw-STL -> world
        self.geoms = []
        for g in range(m.ngeom):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]; nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid)
            if not nm or not nm.startswith("m_"):
                continue
            Rmq = np.zeros(9); mujoco.mju_quat2Mat(Rmq, m.mesh_quat[mid])
            self.geoms.append((g, nm[2:], m.mesh_pos[mid].copy(), Rmq.reshape(3, 3).T.copy()))
        self.scene = self._read_obstacles(m, self.bot.data)
        self.lock = threading.Lock()
        self.v = self.w = 0.0
        self.push = np.zeros(3); self.push_until = 0.0
        self._pending_mode = None      # mode switch + reset are applied BY the loop
        self._pending_reset = False    # thread only — never mutate MjData off-thread
        self.state = {}
        self._compute()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _read_obstacles(self, m, d):
        obs = []
        for g in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if not nm or not nm.startswith("obs_"):
                continue
            t = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder"}.get(
                m.geom_type[g], "box")
            q = np.zeros(4); mujoco.mju_mat2Quat(q, d.geom_xmat[g])    # world pose: data
            obs.append({"type": t, "size": [float(x) for x in m.geom_size[g]],
                        "pos": [float(x) for x in d.geom_xpos[g]],
                        "quat": [float(x) for x in q], "color": [float(x) for x in m.geom_rgba[g][:3]]})
        return obs

    def _active(self):
        if self.mode == "balance":
            return self.bot.model, self.bot.data, self.bot.ctrl
        return self.env.model, self.env.data, self.env._ctrl

    def _compute(self):
        import math
        m, d, ctrl = self._active()
        parts = []
        for g, pid, mp, RmqT in self.geoms:
            Rc = d.geom_xmat[g].reshape(3, 3) @ RmqT
            pos = d.geom_xpos[g] - Rc @ mp
            q = np.zeros(4); mujoco.mju_mat2Quat(q, Rc.flatten())
            parts.append({"id": pid, "pos": [round(float(x), 5) for x in pos],
                          "quat": [round(float(x), 6) for x in q]})
        st = ctrl.read_state()
        adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root")]
        w, x, y, z = d.qpos[adr + 3:adr + 7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self.state = {"parts": parts,
                      "base": [round(float(d.qpos[adr]), 4), round(float(d.qpos[adr + 1]), 4)],
                      "yaw": round(yaw, 4), "speed": round(float(st.v), 3),
                      "pitch": round(float(st.pitch), 3), "upright": not st.fell,
                      "fell": st.fell, "mode": self.mode}

    def _apply_push(self, data):
        if time.time() < self.push_until:
            data.xfrc_applied[self.torso, :3] = self.push
        else:
            data.xfrc_applied[self.torso, :3] = 0.0

    def _loop(self):
        nxt = time.time()
        while self.running:
            with self.lock:
                v, w = self.v, self.w
                pend_mode, self._pending_mode = self._pending_mode, None
                pend_reset, self._pending_reset = self._pending_reset, False
            if pend_mode:                 # all MjData mutation happens on this thread
                self._do_set_mode(pend_mode)
            if pend_reset:
                self._recover()
            mode = self.mode
            if mode == "balance":
                self._apply_push(self.bot.data)
                self.bot.drive(v, w); self.bot.step(SIM_DT)
                fell = self.bot.fell
            else:
                self._apply_push(self.env.data)
                self.env._cmd[:] = [v, 0.0, w]
                a = self.policy.predict(self.obs, deterministic=True)[0]
                self.obs, _, term, _, _ = self.env.step(a)
                fell = term
            if fell:
                self._recover()
            with self.lock:
                self._compute()
            nxt += SIM_DT
            time.sleep(max(0.0, nxt - time.time()))

    def _recover(self):
        if self.mode == "balance":
            self.bot.reset_upright()
        else:
            self.obs, _ = self.env.reset(seed=0)
        with self.lock:
            self.v = self.w = 0.0

    # ---- commands ----
    def set_cmd(self, v, w):
        with self.lock:
            self.v = float(np.clip(v, -1.2, 1.2)); self.w = float(np.clip(w, -1.0, 1.0))

    def do_push(self, fx, fy, mag):
        d = np.array([fx, fy, 0.0]); n = np.linalg.norm(d[:2])
        if n < 1e-6:
            ang = float(np.random.uniform(0, 2 * np.pi)); d = np.array([np.cos(ang), np.sin(ang), 0.0])
        else:
            d /= n
        with self.lock:
            self.push = d * float(np.clip(mag, 0, 250)); self.push_until = time.time() + PUSH_S

    def set_mode(self, mode):
        if mode not in ("balance", "rl"):
            return
        with self.lock:
            self._pending_mode = mode          # applied by the loop thread

    def _do_set_mode(self, mode):
        """Switch controller in-place, continuing from the current physical state.
        Loop-thread only — mutates MjData."""
        if mode == self.mode:
            return
        src_m, src_d, _ = self._active()
        sq, sv = src_d.qpos.copy(), src_d.qvel.copy()
        self.mode = mode
        dst_m, dst_d, _ = self._active()
        dst_d.qpos[:] = sq; dst_d.qvel[:] = sv                # continue from same state
        dst_d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(dst_m, dst_d)
        if mode == "rl":
            self.obs = self.env._obs()
            self.env._last_action[:] = 0; self.env._prev_action[:] = 0
        else:
            self.bot.ctrl.reset()
        with self.lock:
            self.v = self.w = 0.0

    def reset(self):
        with self.lock:
            self._pending_reset = True         # applied by the loop thread

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
        elif self.path.startswith("/scene"):
            self._json({"obstacles": SIM.scene})
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
        elif self.path.startswith("/push"):
            SIM.do_push(data.get("fx", 0.0), data.get("fy", 0.0), data.get("mag", 130.0)); self._json({"ok": True})
        elif self.path.startswith("/mode"):
            SIM.set_mode(data.get("mode", "balance")); self._json({"ok": True, "mode": SIM.mode})
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
