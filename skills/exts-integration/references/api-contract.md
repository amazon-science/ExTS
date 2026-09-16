# ExTS API contract

`exts.py` exposes two classes: `ExTSNode` (a dataclass, rarely touched directly)
and `ExTSTree` (the thing you drive). The tree is domain-agnostic — it only knows
integer `program_idx` values and the scalar rewards you report.

## Lifecycle

```python
from exts import ExTSTree
import random

tree = ExTSTree(
    exploration_type="puct",     # or "uct"
    rng=random.Random(seed),     # pass an RNG for reproducibility
    # ... any hyperparameters (see the paper, arXiv:2608.23848) ...
    # e.g. exts_kwargs unpacked here: score_temperature=0.2, max_children=4, ...
)
```

| Call | When | Notes |
|---|---|---|
| `initialize_root(program_idx, program_score)` | once, at start | Usually `program_idx=0`. Follow it with a `backpropagate_success(0, root_score)` so the root has a visit and a reward on record. |
| `idx = select()` | each iteration | Returns the `program_idx` of the node to expand next. Guaranteed to be an index already in the tree. |
| `add_child(parent_idx, child_idx, child_score)` | after generating a candidate | `child_idx` must be new and unique. `child_score` is the child's current best-known score (used for gating/widening). |
| `backpropagate_success(idx, raw_score)` | after a successful, scored expansion | Updates global score bounds and increments visits + successes + appends `raw_score` for `idx` and all ancestors. |
| `backpropagate_failure(idx)` | after a failed expansion | Increments only visits along the ancestor chain. Drives the success-rate scaler down for failure-prone branches. |
| `sync_scores(per_program_scores)` | whenever host scores change | `per_program_scores[i]` is the current score for `program_idx == i`. Refreshes every node's `program_score`. |
| `get_node(idx)` / `get_stats()` | diagnostics | — |
| `save(path)` / `ExTSTree.load(path)` | persistence | JSON round-trip of the whole tree + config. |

## Semantics you must get right

- **Raw vs. normalized scores.** Always pass the host's **raw** score to
  `backpropagate_success`. ExTS min-max-normalizes each observation internally
  against the tree's evolving global `[score_min, score_max]`. Do not pre-scale.

- **Success vs. failure.** "Success" means the expansion yielded a usable,
  scored candidate — *not* that the score was high. A low but valid score is a
  success. "Failure" is for expansions that produced nothing usable (generation
  error, invalid output, timeout). If your loop always produces a scored child,
  you will never call `backpropagate_failure`, and that is fine.

- **`program_idx` space.** Indices must be integers. If the host uses them as an
  index into a list (as the reference examples do), keep them dense from 0. If
  host IDs are not integers, maintain a `host_id -> idx` mapping and translate at
  the boundary.

- **When to call `sync_scores`.** Gated expansion and progressive widening
  compare a node's `program_score` against quantiles over *all* nodes' scores.
  If the host re-evaluates or re-ranks candidates over time (score drift), call
  `sync_scores` each iteration. If scores are fixed at insertion (low drift), you
  can skip it for a small efficiency gain.

- **Root.** The root is always allowed to expand (it bypasses the expansion gate)
  and always earns progressive-widening bonuses. Seed it before the loop.

- **Determinism.** Selection uses `rng` for tie-breaking and virtual-child
  sampling. Pass a seeded `random.Random` for reproducible runs.

## Rebuilding a tree from existing host state

If the host may already hold several candidates before ExTS starts (e.g. a warm
start), initialize the root then replay the known parent→child edges with
`add_child` + `backpropagate_success` in creation order, so the tree mirrors the
host's population. The reference examples factor this into a small
`rebuild_*_tree_from_state(...)` helper.

**Incremental re-sync (injected selectors).** When the host owns the loop and only
hands you its `state` each call (you implemented its selector protocol), you replay
*incrementally on every call*, not once: keep a cursor of how many candidates you
have mirrored, and each call fold in every candidate created since —
`add_child(parent, idx, score)` under its recorded parent, then
`backpropagate_success`, then `sync_scores`. This needs two things from host state:
**parent→child lineage** and **per-candidate scores**; if lineage is missing, track
the index you last returned and treat new candidates as its children. Such a host
won't pass its `seed` into your object, so seed the `random.Random` you construct
yourself. See the injected-selector variant in
[`controller-pattern.md`](controller-pattern.md).
