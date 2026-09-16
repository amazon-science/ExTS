#!/usr/bin/env bash
#
# setup.sh — fetch the production GEPA library at a pinned commit and apply the
# ExTS integration patch.
#
# We do NOT vendor upstream code. This clones the public gepa-ai/gepa repository
# (the pip-installable `gepa` package) at a fixed release commit and applies
# exts.patch, which adds the ExTS candidate selector as a *purely additive*
# integration (no existing GEPA file is modified). Re-running is safe (idempotent).
#
set -euo pipefail

# --- Configuration (edit here) -------------------------------------------
# Anonymous, tokenless public clone URL. Do not add credentials here.
UPSTREAM_URL="${GEPA_UPSTREAM_URL:-https://github.com/gepa-ai/gepa.git}"
# Pinned to release tag v0.1.4 (latest tagged release at integration time).
PIN="8b0ce6cd99a234f6b74daf37558a2ac0ce18f975"   # tag: v0.1.4
# Clone target. Defaults to ./upstream (git-ignored by the repo's .gitignore).
TARGET_DIR="${GEPA_CLONE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/upstream}"
# -------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="${SCRIPT_DIR}/exts.patch"

if [[ ! -f "${PATCH}" ]]; then
    echo "ERROR: patch not found at ${PATCH}" >&2
    exit 1
fi

echo "==> Upstream : ${UPSTREAM_URL}"
echo "==> Pin      : ${PIN} (v0.1.4)"
echo "==> Target   : ${TARGET_DIR}"

# 1) Clone (only if not already present).
if [[ ! -d "${TARGET_DIR}/.git" ]]; then
    echo "==> Cloning ${UPSTREAM_URL} ..."
    git clone "${UPSTREAM_URL}" "${TARGET_DIR}"
else
    echo "==> ${TARGET_DIR} already a git repo; skipping clone."
fi

# 2) Check out the pinned commit (fetch it first if missing).
pushd "${TARGET_DIR}" >/dev/null
if ! git cat-file -e "${PIN}^{commit}" 2>/dev/null; then
    echo "==> Fetching pinned commit ..."
    git fetch --all --tags
fi
git checkout --quiet "${PIN}"
echo "==> Checked out $(git rev-parse --short HEAD)"

# 3) Apply the patch idempotently.
if git apply --reverse --check "${PATCH}" 2>/dev/null; then
    echo "==> ExTS patch already applied; nothing to do."
elif git apply --check "${PATCH}" 2>/dev/null; then
    git apply "${PATCH}"
    echo "==> Applied ExTS patch."
else
    echo "ERROR: patch does not apply cleanly and is not already applied." >&2
    echo "       Make sure the working tree is clean and at commit ${PIN}." >&2
    popd >/dev/null
    exit 1
fi
popd >/dev/null

cat <<EOF

Done. ExTS is now integrated into: ${TARGET_DIR}

The patch added (additive only — no existing GEPA file changed):
  src/gepa/strategies/exts.py                    (canonical ExTS algorithm)
  src/gepa/strategies/exts_candidate_selector.py (ExTSCandidateSelector)
  tests/test_exts_selector.py                    (dependency-free unit tests)

Next steps:
  cd "${TARGET_DIR}"
  python -m venv .venv && . .venv/bin/activate   # optional but recommended
  pip install -e ".[full]"                       # gepa (MIT) + litellm + datasets

Verify the wiring (no API key needed):
  python -c "import gepa; from gepa.strategies.exts_candidate_selector import ExTSCandidateSelector; print('ok')"
  python -m pytest tests/test_exts_selector.py

The HotpotQA multi-hop benchmark is sample code and needs extra setup: the
HotpotMultiHop program from gepa-artifact (MIT, not shipped here), an LLM
endpoint, a ColBERT wiki17 index, and dspy (see the README for the env vars):
  export GEPA_ARTIFACT_PATH=/path/to/gepa-artifact  # HotpotMultiHop program (MIT)
  export OPENAI_API_BASE=http://localhost:8001/v1  # your LLM endpoint (+ OPENAI_API_KEY)
  bash "${SCRIPT_DIR}/run_matched_multihop.sh"     # original vs exts, seeds 0/42/1024
EOF
