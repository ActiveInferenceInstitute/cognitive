#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec uv run --locked --extra dev --python 3.12 python code/scripts/gate.py "$@"
