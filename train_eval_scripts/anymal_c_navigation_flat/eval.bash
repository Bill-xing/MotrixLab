#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_ENVS="${1:-64}"
POLICY="${2:-}"

args=(
  uv run scripts/play.py
  --env anymal_c_navigation_flat
  --num-envs "${NUM_ENVS}"
)

if [[ -n "${POLICY}" ]]; then
  args+=(--policy "${POLICY}")
fi

"${args[@]}"
