#!/usr/bin/env bash
# Sample code: an illustrative benchmark for this example, not a maintained product. Adapt as needed.
#
# run_matched_multihop.sh -- FULL matched HotpotQA fullwiki MULTI-HOP run.
#
# ORIGINAL (Pareto) vs ExTS candidate selection, seeds {0,42,1024}, budget 6871.
# The 6 arms (3 seeds x 2 selectors) are sharded ONE-PER-SERVER across 6 vLLM
# servers on GPU0-5 (ports 8001-8006) and run in parallel. ColBERT runs once,
# CPU-only, behind a shared local HTTP service (retrieval is diskcached & shared).
#
# Two venvs (built by this benchmark's setup):
#   COLBERT_VENV : .colbert_probe_venv   (colbert-ai + flask, tokenizers<0.20)
#   RUNNER_VENV  : .runner_venv          (dspy 3.x + editable gepa + litellm)
#
# Usage:
#   bash run_matched_multihop.sh            # full run + aggregate
#   GEPA_MAX_METRIC_CALLS=200 GEPA_TRAIN_SIZE=30 GEPA_VAL_SIZE=40 GEPA_TEST_SIZE=40 \
#       GEPA_SEEDS=0 GEPA_SELECTORS="original exts" bash run_matched_multihop.sh   # smoke
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${SCRIPT_DIR}/benchmark_hotpotqa_multihop.py"

COLBERT_VENV="${COLBERT_VENV:-${SCRIPT_DIR}/.colbert_probe_venv}"
RUNNER_VENV="${RUNNER_VENV:-${SCRIPT_DIR}/.runner_venv}"

SELECTORS="${GEPA_SELECTORS:-original exts}"
SEEDS="${GEPA_SEEDS:-0 42 1024}"
BUDGET="${GEPA_MAX_METRIC_CALLS:-6871}"
RESULTS_DIR="${GEPA_RESULTS_DIR:-${SCRIPT_DIR}/results/matched_multihop}"
WORKERS="${GEPA_WORKERS:-32}"

# vLLM servers to shard across (one arm per server). GPU i -> port 8001+i.
PORTS=(${GEPA_PORTS:-8001 8002 8003 8004 8005 8006})
HOST="${VLLM_HOST:-localhost}"

# ColBERT retrieval service.
COLBERT_HOST="${COLBERT_SERVICE_HOST:-127.0.0.1}"
COLBERT_PORT="${COLBERT_SERVICE_PORT:-8899}"
export COLBERT_SERVICE_URL="${COLBERT_SERVICE_URL:-http://${COLBERT_HOST}:${COLBERT_PORT}}"

mkdir -p "${RESULTS_DIR}"
echo "==> selectors : ${SELECTORS}"
echo "==> seeds     : ${SEEDS}"
echo "==> budget    : ${BUDGET} metric calls"
echo "==> servers   : ${PORTS[*]} (host=${HOST})"
echo "==> colbert   : ${COLBERT_SERVICE_URL}"
echo "==> results   : ${RESULTS_DIR}"

# 1) Start the ColBERT retrieval service (CPU-only) if not already up.
if ! curl -s "${COLBERT_SERVICE_URL}/health" >/dev/null 2>&1; then
  echo "==> starting ColBERT service..."
  COLBERT_SERVICE_HOST="${COLBERT_HOST}" COLBERT_SERVICE_PORT="${COLBERT_PORT}" \
    "${COLBERT_VENV}/bin/python" -m hotpotqa_multihop.colbert_service \
    >"${RESULTS_DIR}/colbert_service.log" 2>&1 &
  COLBERT_PID=$!
  echo "    pid=${COLBERT_PID}, log=${RESULTS_DIR}/colbert_service.log"
  trap 'kill ${COLBERT_PID} 2>/dev/null || true' EXIT
fi
echo "==> waiting for ColBERT service to be ready (loads ~13GB index + 5.23M corpus)..."
"${RUNNER_VENV}/bin/python" -c "import sys; sys.path.insert(0,'${SCRIPT_DIR}'); from hotpotqa_multihop.colbert_retriever import wait_until_ready; print(wait_until_ready(1800))"

# 2) Launch the arms, sharded one-per-server.
i=0
pids=()
for selector in ${SELECTORS}; do
  for seed in ${SEEDS}; do
    port="${PORTS[$(( i % ${#PORTS[@]} ))]}"
    log="${RESULTS_DIR}/log_${selector}_seed${seed}.txt"
    echo "----> arm[${i}] selector=${selector} seed=${seed} -> server :${port}  (log ${log})"
    OPENAI_API_BASE="http://${HOST}:${port}/v1" OPENAI_API_KEY="dummy" \
      GEPA_WORKERS="${WORKERS}" COLBERT_SERVICE_URL="${COLBERT_SERVICE_URL}" \
      "${RUNNER_VENV}/bin/python" "${BENCH}" \
        --selector "${selector}" --seed "${seed}" \
        --max-metric-calls "${BUDGET}" --results-dir "${RESULTS_DIR}" \
        >"${log}" 2>&1 &
    pids+=($!)
    i=$(( i + 1 ))
  done
done

echo "==> launched ${#pids[@]} arms; waiting..."
rc=0
for pid in "${pids[@]}"; do
  wait "${pid}" || rc=1
done

echo "==> aggregating results"
"${RUNNER_VENV}/bin/python" "${BENCH}" --aggregate --results-dir "${RESULTS_DIR}"
exit "${rc}"
