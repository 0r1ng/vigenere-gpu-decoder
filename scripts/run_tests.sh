#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python2.7}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: $PYTHON_BIN was not found. Set PYTHON to your Python 2.7 executable." >&2
    exit 1
fi

"$PYTHON_BIN" --version
exec "$PYTHON_BIN" -m unittest discover -s tests -v
