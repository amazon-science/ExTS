# Worked ExTS integrations

Each subdirectory integrates ExTS into a real external project, as a reference for
the integration patterns taught by the skill in
[`../skills/exts-integration/`](../skills/exts-integration/).

## Packaging

To respect upstream licenses and keep this repository small, examples do not
vendor third-party code. Each folder ships an `exts.patch` (the ExTS additions), a
`setup.sh` that clones a pinned upstream commit and applies the patch, a short
`README.md`, and a minimal benchmark script. Heavy datasets, model checkpoints,
and API keys are documented and read from the environment, not committed.

## Examples

| Folder | Upstream | Pattern |
|---|---|---|
| [`gepa/`](./gepa) | GEPA (paper artifact) | Controller |
| [`gepa-lib/`](./gepa-lib) | GEPA (production library) | Controller (injected selector) |
| [`kmse/`](./kmse) | K-MSE | Controller |
| [`treequest/`](./treequest) | TreeQuest / AB-MCTS | Framework port |
| [`aflow/`](./aflow) | AFlow | coming soon |
| [`ifco/`](./ifco) | IFCO | coming soon |

The controller examples keep the original selector as the default so that
original vs ExTS is an apples-to-apples comparison; the framework-port example
adds ExTS as a sibling algorithm next to the host's existing ones.
