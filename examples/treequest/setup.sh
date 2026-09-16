#!/usr/bin/env bash
#
# Fetch TreeQuest at a pinned commit and apply the ExTS integration patch.
#
# We do NOT vendor third-party code. This clones the public upstream at a fixed
# commit and applies exts.patch on top. Re-running is safe (idempotent).
#
set -euo pipefail

# --- Configuration (edit here) -------------------------------------------
UPSTREAM_URL="https://github.com/SakanaAI/treequest.git"
# "bump version (#24)" — a public release commit; the patch is generated
# against this exact tree, so it applies cleanly.
PIN="96047d712d66bbbf4dcc86dcd3e2eaab98c35f83"
# Where to check the repo out (first CLI arg overrides; default ./treequest).
TARGET_DIR="${1:-treequest}"
# -------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="${SCRIPT_DIR}/exts.patch"

if [[ ! -f "${PATCH}" ]]; then
    echo "ERROR: patch not found at ${PATCH}" >&2
    exit 1
fi

echo "==> Upstream : ${UPSTREAM_URL}"
echo "==> Pin      : ${PIN}"
echo "==> Target   : ${TARGET_DIR}"

# 1) Clone (only if not already present).
if [[ ! -d "${TARGET_DIR}/.git" ]]; then
    echo "==> Cloning ${UPSTREAM_URL} ..."
    git clone "${UPSTREAM_URL}" "${TARGET_DIR}"
else
    echo "==> ${TARGET_DIR} already a git repo; skipping clone."
fi

# 2) Check out the pinned commit (fetch it first in case it is missing).
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

Done. ExTS is now available in ${TARGET_DIR}.

Next steps:
  cd ${TARGET_DIR}
  python -m venv .venv && . .venv/bin/activate   # optional but recommended
  pip install -e .                               # installs treequest (Apache-2.0)
  pip install boto3                              # AWS Bedrock client used by the benchmark

Then, from anywhere with that env active:
  python -c "import treequest as tq; print(tq.ExTS)"
  python -m pytest tests/test_exts.py                   # the ExTS unit tests (no data/creds)

The real ExTS-vs-baselines LiveCodeBench benchmark is at
${SCRIPT_DIR}/benchmarks/run_livecodebench.py. It needs the LiveCodeBench dataset
+ checker (lcb_runner, via LCB_RUNNER_PATH) and AWS Bedrock credentials (see the
script header and the README); it is NOT runnable without them:
  python "${SCRIPT_DIR}/benchmarks/run_livecodebench.py" --num-problems 5 --budget 32
EOF
