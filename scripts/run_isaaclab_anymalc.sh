#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ISAACLAB_ROOT=../IsaacLab ./scripts/run_isaaclab_anymalc.sh train 1024 true 1000000
#   ISAACLAB_ROOT=../IsaacLab ./scripts/run_isaaclab_anymalc.sh play 256 false 0 logs/skrl/nav/checkpoints/best_agent.pt

MODE="${1:-train}"
NUM_ENVS="${2:-1024}"
HEADLESS="${3:-true}"
MAX_ITERS="${4:-1000000}"
CKPT="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${REPO_ROOT}/../IsaacLab}"

TASK="${ISAACLAB_TASK:-Isaac-Navigation-Flat-Anymal-C-v0}"
PLAY_TASK="${ISAACLAB_PLAY_TASK:-Isaac-Navigation-Flat-Anymal-C-Play-v0}"
TRAIN_ENTRY="${ISAACLAB_TRAIN_ENTRY:-${ISAACLAB_ROOT}/scripts/reinforcement_learning/skrl/train.py}"
PLAY_ENTRY="${ISAACLAB_PLAY_ENTRY:-${ISAACLAB_ROOT}/scripts/reinforcement_learning/skrl/play.py}"

headless_args=()
case "${HEADLESS}" in
  true | 1 | yes) headless_args=(--headless) ;;
  false | 0 | no) ;;
  *)
    echo "HEADLESS must be true or false, got: ${HEADLESS}" >&2
    exit 2
    ;;
esac

case "${MODE}" in
  train)
    python "${TRAIN_ENTRY}" \
      --task "${TASK}" \
      --num_envs "${NUM_ENVS}" \
      "${headless_args[@]}" \
      --max_iterations "${MAX_ITERS}"
    ;;
  eval | play)
    if [[ -z "${CKPT}" ]]; then
      echo "play mode requires a checkpoint path" >&2
      exit 1
    fi
    python "${PLAY_ENTRY}" \
      --task "${PLAY_TASK}" \
      --num_envs "${NUM_ENVS}" \
      "${headless_args[@]}" \
      --checkpoint "${CKPT}"
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use train or play." >&2
    exit 1
    ;;
esac
