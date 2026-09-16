#!/usr/bin/env bash
#
# Fetch the pinned upstream K-MSE and apply the ExTS integration patch.
#
# We do NOT vendor third-party code. This script clones the official upstream
# and applies exts.patch on top of it. Nothing here contains secrets or
# absolute paths; API keys are read from the environment at run time (see
# README.md and run_benchmark.sh).
#
# The patch was generated against upstream commit PINNED_COMMIT below. At the
# time of writing this is also upstream's default-branch HEAD, so it applies
# cleanly. If upstream later drifts, the script retries with `git apply --3way`.
set -euo pipefail

# Official upstream (ACL 2025, MIT-licensed; Copyright (c) 2025 Xiang Zhuang).
UPSTREAM_URL="https://github.com/HICAI-ZJU/K-MSE.git"
PINNED_COMMIT="94d4cb307f205454dbd8bd875df927b79c575498"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SCRIPT_DIR}/upstream"          # git-ignored (examples/*/upstream/)
PATCH="${SCRIPT_DIR}/exts.patch"

if [ -d "${DEST}/.git" ]; then
    echo "Upstream already present at ${DEST}; leaving it untouched."
    echo "Delete it and re-run this script to start fresh."
    exit 0
fi

echo "Cloning upstream K-MSE into ${DEST} ..."
git clone "${UPSTREAM_URL}" "${DEST}"

cd "${DEST}"
echo "Checking out pinned commit ${PINNED_COMMIT} ..."
if ! git cat-file -e "${PINNED_COMMIT}^{commit}" 2>/dev/null; then
    echo "ERROR: pinned commit ${PINNED_COMMIT} not found upstream (repo may have been rewritten)." >&2
    echo "       Refusing to fall back to a moving default-branch HEAD; aborting." >&2
    exit 1
fi
git checkout "${PINNED_COMMIT}"

echo "Applying ExTS patch ..."
if git apply --check "${PATCH}" 2>/dev/null; then
    git apply "${PATCH}"
    echo "Patch applied cleanly."
else
    echo "Clean apply failed (upstream drift). Retrying with --3way ..."
    git apply --3way "${PATCH}"
    echo "Patch applied via 3-way merge (inspect for conflict markers)."
fi

cat <<EOF

Done. ExTS integration is now in ${DEST}/k-mse/code/
  - exts.py        (the ExTS algorithm, copied in by the patch)
  - exts_mcts.py   (MCTSrExTS: the ExTS controller/adapter)
  - run.py         (adds --mcts_variant {original,exts})

Next steps (see README.md):
  1. Download the heavy assets NOT included here:
       - code/models/ckpt.pth   (~29 MB molecule-spectrum scorer)
       - data/data.json, data/kb.json, data/Spectrum_imgs/   (benchmark)
     These ship in the upstream repo you just cloned; if missing, obtain them
     from ${UPSTREAM_URL}.
  2. Configure your OpenAI-compatible provider in code/utils/chatbot.py
     (read the key from OPENAI_API_KEY rather than hard-coding it).
  3. Run:  cd ${DEST}/k-mse && bash "${SCRIPT_DIR}/run_benchmark.sh"
EOF
