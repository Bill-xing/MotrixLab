#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_motrixlab_go1.sh train 1024 true jax
#   ./scripts/run_motrixlab_go1.sh play 128 false jax runs/go1-flat-terrain-walk/latest/checkpoints/best_agent.pickle

MODE="${1:-train}"
NUM_ENVS="${2:-1024}"
HEADLESS="${3:-true}"
TRAIN_BACKEND="${4:-jax}"
POLICY="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${MOTRIXLAB_ENV:-go1-flat-terrain-walk}"
TRAIN_ENTRY="${MOTRIXLAB_TRAIN_ENTRY:-${REPO_ROOT}/scripts/train.py}"
PLAY_ENTRY="${MOTRIXLAB_PLAY_ENTRY:-${REPO_ROOT}/scripts/play.py}"

render_args=()
case "${HEADLESS}" in
  true | 1 | yes) ;;
  false | 0 | no) render_args=(--render) ;;
  *)
    echo "HEADLESS must be true or false, got: ${HEADLESS}" >&2
    exit 2
    ;;
esac

case "${MODE}" in
  train)
    python "${TRAIN_ENTRY}" \
      --env "${ENV_NAME}" \
      --num-envs "${NUM_ENVS}" \
      --train-backend "${TRAIN_BACKEND}" \
      "${render_args[@]}"
    ;;
  eval | play)
    if [[ -z "${POLICY}" ]]; then
      echo "play mode requires a policy path" >&2
      exit 1
    fi
    python "${PLAY_ENTRY}" \
      --env "${ENV_NAME}" \
      --num-envs "${NUM_ENVS}" \
      --policy "${POLICY}"
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use train or play." >&2
    exit 1
    ;;
esac
