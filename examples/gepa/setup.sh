#!/usr/bin/env bash
#
# setup.sh — fetch pinned upstream GEPA artifact and apply the ExTS patch.
#
# We do NOT vendor upstream code. This script clones the public GEPA
# paper-artifact repository at a pinned commit and applies exts.patch, which
# adds the ExTS candidate-selection controller on top of stock (Pareto-only)
# GEPA. The script is idempotent: re-running it is safe.
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Public upstream GEPA paper-artifact repository.
#   Verified: this is the parent of the ExTS development fork
#   (`gh repo view <fork> --json parent` => gepa-ai/gepa-artifact), and the
#   pinned commit below exists in this public repo. Override with the
#   GEPA_UPSTREAM_URL env var if upstream ever moves.
UPSTREAM_URL="${GEPA_UPSTREAM_URL:-https://github.com/gepa-ai/gepa-artifact.git}"

# Pinned base commit: stock, Pareto-only GEPA that the patch was generated
# against ("Merge pull request #17 from gepa-ai/fix/aliased-val-subscores").
PIN="cbefbc1aa0f43dd39874ec4bf42211365dbda42e"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLONE_DIR="${GEPA_CLONE_DIR:-${SCRIPT_DIR}/gepa-artifact}"
PATCH="${SCRIPT_DIR}/exts.patch"

# ---------------------------------------------------------------------------
# 1. Clone (idempotent)
# ---------------------------------------------------------------------------
if [ ! -d "${CLONE_DIR}/.git" ]; then
  echo ">> Cloning ${UPSTREAM_URL}"
  echo "   -> ${CLONE_DIR}"
  git clone "${UPSTREAM_URL}" "${CLONE_DIR}"
else
  echo ">> Upstream already cloned at ${CLONE_DIR}; fetching latest refs."
  git -C "${CLONE_DIR}" fetch --all --tags --quiet || true
fi

# ---------------------------------------------------------------------------
# 2. Check out the pinned commit
# ---------------------------------------------------------------------------
echo ">> Checking out pinned commit ${PIN}"
git -C "${CLONE_DIR}" checkout --quiet "${PIN}"

# ---------------------------------------------------------------------------
# 3. Apply the ExTS patch (idempotent)
# ---------------------------------------------------------------------------
if git -C "${CLONE_DIR}" apply --reverse --check "${PATCH}" 2>/dev/null; then
  echo ">> ExTS patch already applied; skipping."
elif git -C "${CLONE_DIR}" apply --check "${PATCH}" 2>/dev/null; then
  echo ">> Applying ExTS patch (exts.patch)"
  git -C "${CLONE_DIR}" apply "${PATCH}"
  echo "   Patch applied: gepa_artifact/gepa/exts.py (new) + gepa_artifact/gepa/gepa.py"
else
  echo "ERROR: exts.patch does not apply cleanly at ${PIN}." >&2
  echo "       Check GEPA_UPSTREAM_URL / PIN, or that the clone is unmodified." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Next steps
# ---------------------------------------------------------------------------
cat <<EOF

============================================================================
ExTS is now integrated into: ${CLONE_DIR}

Next steps
----------
1. Install upstream's dependencies. The artifact pins a custom DSPy fork and
   the Arbor trainer via uv "sources", so use upstream's own helper + uv:

     cd "${CLONE_DIR}"
     bash setup_gepa_repo.sh        # clones the pinned dspy + arbor forks
     uv sync                        # installs everything (recommended)

   (A plain \`pip install -e .\` will NOT pull the custom DSPy fork that GEPA
   depends on; prefer \`uv sync\`.)

2. Provide model credentials via environment variables (nothing is hardcoded):

     export GEPA_MODEL=openai/gpt-4.1-mini      # any LiteLLM/DSPy model id
     export OPENAI_API_KEY=sk-...               # or GEPA_API_KEY
     # For a local OpenAI-compatible server instead:
     #   export GEPA_API_BASE=http://localhost:8001/v1/

3. Run the minimal HotpotQA benchmark with ExTS candidate selection:

     python "${SCRIPT_DIR}/benchmark_hotpotqa.py" --max-metric-calls 200

   The first HotpotQA run auto-downloads the wiki.abstracts.2017 corpus
   (~1.7GB) and builds a local BM25 index (~1.1GB). No GPU / ColBERT index
   required. See README.md for details.
============================================================================
EOF
