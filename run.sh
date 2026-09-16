#!/usr/bin/env bash
# Run from any working directory; override PYTHON to select an environment.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$REPO_DIR/main.py" "$@"
