#!/usr/bin/env bash
# Sample code: an illustrative benchmark for this example, not a maintained product. Adapt as needed.
#
# Minimal ExTS smoke benchmark for K-MSE.
#
# Runs upstream's own entry point (code/run.py) with --mcts_variant exts on a
# small number of rollouts. Model + API keys are read from the environment; no
# secrets or absolute paths are baked in.
#
# Prerequisites (see README.md):
#   * ./setup.sh has been run (upstream cloned + patch applied under ./upstream)
#   * Heavy assets present under upstream/k-mse/: code/models/ckpt.pth,
#     data/data.json, data/kb.json, data/Spectrum_imgs/
#   * A CUDA GPU (the scorer runs on .cuda())
#   * code/utils/chatbot.py wired to read OPENAI_API_KEY (and optionally
#     OPENAI_API_BASE) from the environment
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KMSE_DIR="${KMSE_DIR:-${SCRIPT_DIR}/upstream/k-mse}"

# --- config from environment (with smoke-run defaults) ----------------------
MODEL="${KMSE_MODEL:-gpt-4o-mini}"           # --model_name
MAX_ROLLOUTS="${KMSE_MAX_ROLLOUTS:-4}"       # rollouts per molecule
RETRIEVE_TOPK="${KMSE_RETRIEVE_TOPK:-1}"
EXPLORATION_TYPE="${KMSE_EXPLORATION_TYPE:-puct}"
SEED="${KMSE_SEED:-0}"
# By default we run --debug (a single molecule) for a fast smoke test.
# Set KMSE_FULL=1 to run the full split-0 slice (up to 20 molecules) instead.
DEBUG_FLAG="--debug"
[ "${KMSE_FULL:-0}" = "1" ] && DEBUG_FLAG=""

if [ ! -d "${KMSE_DIR}" ]; then
    echo "ERROR: ${KMSE_DIR} not found. Run ./setup.sh first." >&2
    exit 1
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "WARNING: OPENAI_API_KEY is not set. Make sure code/utils/chatbot.py" >&2
    echo "         reads your key from the environment before running." >&2
fi

# run.py uses paths relative to the k-mse/ dir (code/prompts/..., data/...).
cd "${KMSE_DIR}"

echo "Running ExTS smoke benchmark:"
echo "  model=${MODEL}  rollouts=${MAX_ROLLOUTS}  exploration=${EXPLORATION_TYPE}  seed=${SEED}  ${DEBUG_FLAG:-(full split)}"

python code/run.py \
    --exp_name exts_smoke \
    --exp_id exts \
    --mcts_variant exts \
    --model_name "${MODEL}" \
    --max_rollouts "${MAX_ROLLOUTS}" \
    --retrieve_topk "${RETRIEVE_TOPK}" \
    --exts_exploration_type "${EXPLORATION_TYPE}" \
    --random_seed "${SEED}" \
    --split 0 \
    ${DEBUG_FLAG}

echo "Done. Results and per-molecule trees are under the run's dump/ directory."
echo "For the original selector baseline, rerun with --mcts_variant original."
