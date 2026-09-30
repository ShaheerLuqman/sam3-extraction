#!/usr/bin/env bash
# Dev: uvicorn (GPU 1 only) + Vite dev server with HMR.
set -euo pipefail
cd "$(dirname "$0")/.."                       # repo root

# run.sh stop | start | restart | status | logs | install | uninstall: the services
case "${1:-}" in
  stop|start|restart|status|logs|install|uninstall) exec bash webapp/service.sh "$1" ;;
  "") ;;
  *) echo "usage: $0 [stop|start|restart|status|logs|install|uninstall]" >&2; exit 2 ;;
esac

# Installed as services (webapp/service.sh), the app is already running and
# restarts itself; starting it here too would fight them for the ports.
if systemctl --user is-active --quiet sam3-webapp-api 2>/dev/null; then
  echo "webapp: running as a service already — http://127.0.0.1:${SAM3_VITE_PORT:-5173}"
  echo "webapp: webapp/service.sh restart | logs | stop   (or uninstall, to go back to run.sh)"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"   # physical GPU 1 for SAM 3; Qwen uses GPU 0
# Vite serves the frontend on :5173; the backend on :8000 is the API only, so it
# never shows a stale build of the frontend (make serve still serves dist/)
export SAM3_SERVE_FRONTEND=0
export SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-$PWD/checkpoints/sam3.pt}"

API_PORT="${SAM3_PORT:-8000}"
WEB_PORT="${SAM3_VITE_PORT:-5173}"
REPO="$PWD"
UVICORN_MATCH="uvicorn webapp.backend.main:app"
# precise enough that another project's Vite on the same port is left alone
VITE_MATCH="$REPO/webapp/frontend/node_modules"

# --------------------------------------------------------------------------- #
# Reclaim anything a previous start left behind.
#
# The EXIT trap below only fires on a graceful exit. When run.sh is SIGKILLed,
# or its terminal simply disappears, the trap never runs and its uvicorn
# outlives it — still holding :8000, still holding ~8 GB of GPU. Every later
# start then fails to bind, and you get Vite with a dead backend behind it,
# which looks like the app working until nothing responds.
#
# Killing our own leftovers is safe. Killing whatever else happens to be on the
# port is not, so that case stops the script instead.
# --------------------------------------------------------------------------- #
stop_pid() {                                   # stop_pid <pid> <label>
  local pid="$1" label="$2" i
  kill -TERM "$pid" 2>/dev/null || true
  for i in $(seq 1 60); do
    kill -0 "$pid" 2>/dev/null || { echo "webapp: stopped $label (pid $pid)"; return; }
    sleep 0.1
  done
  echo "webapp: $label (pid $pid) ignored TERM, forcing"
  kill -9 "$pid" 2>/dev/null || true
}

reclaim_port() {                               # reclaim_port <port> <cmdline match>
  local port="$1" want="$2" pid cmd
  for pid in $(ss -ltnpH "sport = :$port" 2>/dev/null \
               | grep -oP 'pid=\K[0-9]+' | sort -u); do
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$cmd" == *"$want"* ]]; then
      stop_pid "$pid" "leftover on :$port"
    else
      echo "webapp: port $port is held by something that is not ours:" >&2
      echo "webapp:   pid $pid — $cmd" >&2
      echo "webapp: refusing to kill it. Free the port, or set SAM3_PORT /" >&2
      echo "webapp: SAM3_VITE_PORT to run somewhere else." >&2
      exit 1
    fi
  done
}

# Other run.sh instances first — their own traps tidy up their children.
# The pattern is anchored so it matches a shell *running* the script and not a
# command that merely mentions it, and the cwd check keeps it to this repo
# (run.sh cd's to the repo root, however it was invoked).
for pid in $(pgrep -f 'run\.sh$' 2>/dev/null || true); do
  [ "$pid" = "$$" ] && continue
  [ "$pid" = "${PPID:-0}" ] && continue
  [ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" = "$REPO" ] || continue
  stop_pid "$pid" "earlier run.sh"
done

reclaim_port "$API_PORT" "$UVICORN_MATCH"
reclaim_port "$WEB_PORT" "$VITE_MATCH"

# a backend that failed to bind can still be alive and holding the GPU
for pid in $(pgrep -f "$UVICORN_MATCH" 2>/dev/null || true); do
  [ "$pid" = "$$" ] && continue
  stop_pid "$pid" "orphaned backend"
done

if [ -n "${SAM3_QUIET_START:-}" ]; then :; else
  echo "webapp: starting on http://127.0.0.1:$WEB_PORT (api :$API_PORT)"
fi

# .venv's interpreter lives inside the VS Code snap and vanishes when VS Code
# self-updates — re-point it at the stable copy (see webapp/fix-venv.sh).
if ! .venv/bin/python -c "" 2>/dev/null; then
  echo "webapp: .venv interpreter is missing — running webapp/fix-venv.sh"
  bash webapp/fix-venv.sh
fi

# HUP matters: closing the terminal is the usual way this gets orphaned
trap 'kill 0' EXIT INT TERM HUP

.venv/bin/uvicorn webapp.backend.main:app --host 127.0.0.1 --port "$API_PORT" --workers 1 &

if [ -d webapp/frontend/node_modules ]; then
  (cd webapp/frontend && npm run dev -- --host 127.0.0.1 --port "$WEB_PORT") &
else
  echo "webapp/frontend not set up yet — backend only on http://127.0.0.1:$API_PORT"
fi

wait
