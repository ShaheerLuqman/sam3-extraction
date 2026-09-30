#!/usr/bin/env bash
# The webapp as two systemd user services, so it keeps running when the terminal
# (or VS Code's remote connection) that started it goes away, and comes back by
# itself if it crashes. run.sh ties both to its terminal: when VS Code reconnects
# or reloads, the terminal gets SIGHUP and takes the backend and Vite down with it.
#
#   webapp/service.sh install   write the units, enable them, start them
#   webapp/service.sh restart   restart both (e.g. after a backend change)
#   webapp/service.sh stop | start | status
#   webapp/service.sh logs      follow both logs (journald)
#   webapp/service.sh uninstall stop, disable and remove the units
#
# Same ports and GPUs as run.sh: Vite on 127.0.0.1:5173, the API on 127.0.0.1:8000,
# SAM 3 on GPU 1, Qwen on GPU 0.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
API=sam3-webapp-api
WEB=sam3-webapp-web
API_PORT="${SAM3_PORT:-8000}"
WEB_PORT="${SAM3_VITE_PORT:-5173}"
NODE_BIN="$(dirname "$(command -v node)")"

write_units() {
  mkdir -p "$UNIT_DIR"
  cat > "$UNIT_DIR/$API.service" <<EOF
[Unit]
Description=SAM 3 webapp — API (uvicorn, SAM 3 on GPU 1, Qwen on GPU 0)
After=network.target

[Service]
WorkingDirectory=$REPO
Environment=CUDA_VISIBLE_DEVICES=1
Environment=SAM3_SERVE_FRONTEND=0
Environment=SAM3_CHECKPOINT=$REPO/checkpoints/sam3.pt
Environment=PYTHONUNBUFFERED=1
ExecStart=$REPO/.venv/bin/uvicorn webapp.backend.main:app --host 127.0.0.1 --port $API_PORT --workers 1
# the whole cgroup, so the Qwen workers (own sessions) go too
KillMode=control-group
TimeoutStopSec=40
Restart=always
RestartSec=5
# 143 = stopped by SIGTERM, i.e. a normal stop
SuccessExitStatus=143

[Install]
WantedBy=default.target
EOF
  cat > "$UNIT_DIR/$WEB.service" <<EOF
[Unit]
Description=SAM 3 webapp — frontend (Vite dev server, proxies /api to :$API_PORT)
After=network.target

[Service]
WorkingDirectory=$REPO/webapp/frontend
Environment=PATH=$NODE_BIN:/usr/local/bin:/usr/bin:/bin
ExecStart=$NODE_BIN/node $REPO/webapp/frontend/node_modules/vite/bin/vite.js --host 127.0.0.1 --port $WEB_PORT --strictPort
Restart=always
RestartSec=3
SuccessExitStatus=143

[Install]
WantedBy=default.target
EOF
}

case "${1:-status}" in
  install)
    # anything run.sh started still holds the ports and the GPU
    for pid in $(pgrep -f 'run\.sh$' || true); do
      [ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" = "$REPO" ] && kill -TERM "$pid" || true
    done
    write_units
    systemctl --user daemon-reload
    systemctl --user enable --now "$API" "$WEB"
    if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
      loginctl enable-linger "$USER" 2>/dev/null \
        && echo "enabled lingering: the services keep running with nobody logged in" \
        || echo "note: run 'sudo loginctl enable-linger $USER' so the services survive the last logout"
    fi
    echo "installed. app: http://127.0.0.1:$WEB_PORT   logs: webapp/service.sh logs"
    ;;
  uninstall)
    systemctl --user disable --now "$API" "$WEB" || true
    rm -f "$UNIT_DIR/$API.service" "$UNIT_DIR/$WEB.service"
    systemctl --user daemon-reload
    ;;
  start|stop|restart|status)
    systemctl --user "$1" "$API" "$WEB" --no-pager || true
    ;;
  logs)
    journalctl --user -u "$API" -u "$WEB" -f -n 100
    ;;
  *)
    echo "usage: $0 install|uninstall|start|stop|restart|status|logs" >&2
    exit 2
    ;;
esac
