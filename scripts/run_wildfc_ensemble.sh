#!/usr/bin/env bash
# Compatibility wrapper. Prefer: python scripts/run_wildfc.py
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python scripts/run_wildfc.py "$@"
