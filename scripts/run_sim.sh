#!/usr/bin/env bash
# Launch the BraveBot LIVE PHYSICS simulator (real MuJoCo + balance controller),
# streamed to the browser — drive the robot around with the arrow keys.
#   ./scripts/run_sim.sh            # serve on :8001 and open it
set -e
cd "$(dirname "$0")/.."
PORT="${1:-8001}"
( sleep 1.5 && open "http://127.0.0.1:${PORT}/web/sim.html" 2>/dev/null ) &
exec ./.venv/bin/python scripts/sim_server.py "${PORT}"
