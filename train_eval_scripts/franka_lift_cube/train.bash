#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_ENVS="${1:-4096}"
TRAIN_BACKEND="${2:-jax}"
PRETRAINED="${3:-}"

args=(
  uv run scripts/train.py
  --env franka_lift_cube
  --num-envs "${NUM_ENVS}"
  --train-backend "${TRAIN_BACKEND}"
)

if [[ -n "${PRETRAINED}" ]]; then
  args+=(--pretrained "${PRETRAINED}")
fi

"${args[@]}"
