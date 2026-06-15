#!/usr/bin/env bash
# Launch the BraveBot interactive part explorer (web/index.html) in your browser.
#   ./scripts/view_parts.sh           # serve on :8000 and open it
set -e
cd "$(dirname "$0")/.."
PORT="${1:-8000}"
# regenerate the parts manifest if the sim env is available (otherwise use the committed one)
[ -x .venv/bin/python ] && .venv/bin/python scripts/export_parts.py 2>/dev/null || true
( sleep 1 && open "http://127.0.0.1:${PORT}/web/" 2>/dev/null ) &
echo "BraveBot part explorer → http://127.0.0.1:${PORT}/web/   (Ctrl-C to stop the server)"
exec python3 -m http.server "${PORT}" --bind 127.0.0.1
