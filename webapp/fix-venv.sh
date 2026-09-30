#!/usr/bin/env bash
# Re-point ./.venv at a stable Python 3.12 when its interpreter goes missing.
#
# The venv was created by uv using the Python that ships *inside* the VS Code
# snap (~/snap/code/<rev>/.local/share/uv/python/...). VS Code auto-updates and
# deletes the old <rev>, which orphans .venv/bin/python and every console script
# ("bad interpreter: .../.venv/bin/python3: No such file or directory").
#
# Fix: copy a python-build-standalone 3.12 out of the snap into a stable location
# once, then point the venv's pyvenv.cfg + bin/python* symlinks at it. The 7 GB of
# site-packages (torch, sam3, ...) are untouched.
set -euo pipefail
cd "$(dirname "$0")/.."                                   # repo root

STABLE_DIR="$HOME/.local/share/python-standalone"
DST="$STABLE_DIR/cpython-3.12"

if [ ! -x "$DST/bin/python3.12" ]; then
  SRC="$(ls -d "$HOME"/snap/code/*/.local/share/uv/python/cpython-3.12*-linux-x86_64-gnu 2>/dev/null | sort -V | tail -1 || true)"
  if [ -z "${SRC:-}" ] || [ ! -x "$SRC/bin/python3.12" ]; then
    echo "No cpython-3.12 found under ~/snap/code/*/.local/share/uv/python/." >&2
    echo "Open VS Code once (it re-provisions one), or install any Python 3.12 and" >&2
    echo "point STABLE_DIR/cpython-3.12 at it, then re-run this script." >&2
    exit 1
  fi
  echo "copying $SRC -> $DST"
  mkdir -p "$STABLE_DIR"
  rm -rf "$DST"
  cp -a "$SRC" "$DST"
fi

"$DST/bin/python3.12" -c "import ssl, sqlite3, ensurepip" \
  || { echo "copied interpreter is broken: $DST" >&2; exit 1; }

echo "re-pointing ./.venv at $DST"
[ -f .venv/pyvenv.cfg ] && cp -f .venv/pyvenv.cfg .venv/pyvenv.cfg.bak
sed -i "s|^home = .*|home = $DST/bin|" .venv/pyvenv.cfg
ln -sfn "$DST/bin/python3.12" .venv/bin/python
ln -sfn python .venv/bin/python3
ln -sfn python .venv/bin/python3.12

.venv/bin/python -c "import sys; print('venv python:', sys.executable, sys.version.split()[0])"
.venv/bin/python -c "import torch, fastapi, uvicorn, sam3; print('core imports OK; cuda:', torch.cuda.is_available())"
.venv/bin/python -c "import ultralytics; print('ultralytics', ultralytics.__version__)" \
  || echo "(ultralytics missing — 'find similar → YOLOE' needs:  .venv/bin/python -m pip install ultralytics)"
echo "done — ./webapp/run.sh should work now"
