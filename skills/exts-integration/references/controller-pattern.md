# Controller pattern

Use this when the host system selects its next candidate from a flat structure —
greedily (the current best), linearly (the last / best-of-N), or from a Pareto
front — and you want ExTS to make that choice instead. You keep the host's
generate + score code and drive an `ExTSTree` from the loop. If the host instead
exposes a *pluggable selector object* you can hand it, see the injected-selector
variant below — it needs zero edits to the host's loop.

## Recipe

1. **Vendor the algorithm.** Copy `exts.py` into the host repo (e.g. next to the
   optimizer module). No dependencies beyond the standard library.

2. **Add a strategy toggle.** Introduce a config field / CLI flag, e.g.
   `selection_strategy: Literal["original", "exts"] = "original"`, plus an
   optional `exts_kwargs: dict | None` passed straight into `ExTSTree(...)`.
   Default to the original.

3. **Construct the tree when the strategy is ExTS.** At the point where the host
   seeds its search, build the tree and initialize the root from the first
   candidate's score. If the host already holds >1 candidate, replay them (see
   *rebuild* in `api-contract.md`).

4. **Route selection through ExTS.** At the host's "pick next candidate" site,
   branch on the flag: original code path, or `idx = tree.select()`.

5. **Add the tree hooks.** Wherever the host adds a new candidate, call
   `add_child(...)`; on a successful score call `backpropagate_success(...)`; on
   a failed/aborted expansion call `backpropagate_failure(...)`; after scores
   update call `sync_scores(...)`. Optionally `save()` the tree to the run dir.

## Generic template

```python
# --- setup (once) ---
mcts_tree = None
if self.selection_strategy == "exts":
    from exts import ExTSTree
    mcts_tree = ExTSTree(rng=self.rng, **(self.exts_kwargs or {}))
    mcts_tree.initialize_root(program_idx=0, program_score=scores[0])
    mcts_tree.backpropagate_success(0, raw_score=scores[0])
    if len(candidates) > 1:                     # warm start
        rebuild_tree_from_state(mcts_tree, candidates, scores)

# --- selecting the next candidate (each iteration) ---
def select_next_candidate():
    if mcts_tree is not None:
        return mcts_tree.select()
    # ... original greedy / linear / pareto logic, unchanged ...
    return original_selection()

# --- inside the loop ---
parent_idx = select_next_candidate()
child = generate(parent_idx)                    # host's generate step (unchanged)
if child is None:
    if mcts_tree is not None:
        mcts_tree.backpropagate_failure(parent_idx)
    continue

child_idx = len(scores)
child_score = evaluate(child)                   # host's score step (unchanged)
scores.append(child_score)
store_candidate(child_idx, child)

if mcts_tree is not None:
    mcts_tree.add_child(parent_idx, child_idx, child_score)
    mcts_tree.backpropagate_success(child_idx, raw_score=child_score)
    mcts_tree.sync_scores(scores)
```

## Variant: integrating through a host extension point (injected selector)

Many mature systems don't want you editing their loop at all — they expose a
**pluggable selector/strategy object**: a protocol such as
`select_candidate_idx(state) -> int` (or `choose(...)`) that you implement and hand
to the host. This is still the controller pattern, but with two differences: the
"toggle" is *which object you pass* (no flag, zero edits to host code), and you get
only the host's `state` each call instead of per-step hooks. So you **self-sync the
tree from state on every call** — keep a cursor of how far you've mirrored and fold
in each candidate created since.

This needs two things from the host's state: **(a) parent→child lineage** and
**(b) per-candidate scores**. If lineage isn't recorded, track it yourself: remember
the index you returned last and treat new candidates as its children. If the host
never surfaces rejected proposals, *infer* failures the same way (a returned parent
that produced no new child → `backpropagate_failure`).

```python
class ExTSCandidateSelector:                 # implements the host's selector protocol
    def __init__(self, rng=None, **exts_kwargs):
        self.tree = None
        self.rng = rng or random.Random(0)   # host 'seed' may NOT reach an injected
                                              # object — seed yourself for reproducibility
        self.exts_kwargs = exts_kwargs
        self._synced = 0                      # how many candidates mirrored so far
        self._pending = None                  # parent last returned (for failure inference)

    def select_candidate_idx(self, state) -> int:
        scores = state.per_candidate_scores   # (b) per-candidate scores (dense, by idx)
        parents = state.parent_of_candidate   # (a) lineage: parent idx (or None) per candidate
        if self.tree is None:
            self.tree = ExTSTree(rng=self.rng, **self.exts_kwargs)
            self.tree.initialize_root(0, scores[0])
            self.tree.backpropagate_success(0, raw_score=scores[0])
            self._synced = 1
        # failure inference: the parent we returned produced no new candidate
        if self._pending is not None and len(scores) == self._synced:
            self.tree.backpropagate_failure(self._pending)
        # mirror every candidate created since the last call
        for idx in range(self._synced, len(scores)):
            parent = parents[idx]
            parent = 0 if parent is None else parent   # multi-parent: use primary/first
            self.tree.add_child(parent, idx, scores[idx])
            self.tree.backpropagate_success(idx, raw_score=scores[idx])
        self._synced = len(scores)
        self.tree.sync_scores(scores)
        chosen = self.tree.select()
        self._pending = chosen
        return chosen
```

Adapt the attribute names to the host's `state`. When you vendor `exts.py` into a
real package, match the host's import conventions — a package that bans relative
imports needs `exts.py` inside the package and imported absolutely
(`from yourpkg.strategies.exts import ExTSTree`).

## How the worked examples do it

### GEPA paper artifact (`examples/gepa`) — edit-the-loop controller

- The optimizer loop lives in `gepa_artifact/gepa/gepa.py`; candidate selection
  is factored into `select_next_candidate_to_update(...)`.
- The integration widens the existing `candidate_selection_strategy` flag with an
  `'exts'` value and adds an `exts_kwargs` dict. When set, the selection function
  returns `mcts_tree.select()` instead of the Pareto-front sampler.
- Tree hooks are placed at every existing add-candidate site (`add_child`), every
  early-exit / failed-proposal path (`backpropagate_failure`), after each full
  evaluation (`backpropagate_success` + `sync_scores`), and the tree is `save`-d
  into the run directory.
- The original `'pareto'` (default) and `'greedy'` paths are untouched, so the
  three strategies are directly comparable.

### K-MSE (`examples/kmse`) — MCTS-R → ExTS

- K-MSE's original search is an MCTS-R loop (`k-mse/code/mcts.py`, class `MCTSr`)
  with UCT-weighted importance sampling over candidate nodes.
- Rather than editing `MCTSr`, the integration adds a **sibling adapter class**
  (an `MCTSrExTS`) whose `run()` mirrors the original but drives an `ExTSTree`:
  `initialize_root` → per-rollout `select` → reuse the original
  `self_refine`/`_evaluate` LLM pipeline → `add_child` → `backpropagate_success`.
- A `--mcts_variant {original,exts}` flag in `run.py` dispatches to the original
  class or the ExTS adapter. The original selection logic is not modified.

The K-MSE style (a parallel adapter class + a dispatch flag) is the cleanest when
the host's selection and its expand/score pipeline are entangled in one class —
you subclass/duplicate the driver rather than threading branches through it.

### GEPA library (`examples/gepa-lib`) — injected-selector controller

The production `gepa` package (distinct from the paper artifact above) makes this
even cleaner: `optimize(...)` accepts `candidate_selection_strategy` as either a
built-in string *or any object* implementing its `CandidateSelector` protocol
(`select_candidate_idx(state) -> int`). ExTS attaches as an `ExTSCandidateSelector`
passed to `optimize` — **zero edits to any GEPA file**; the default `"pareto"`
selector is untouched. The selector self-syncs an `ExTSTree` from
`state.parent_program_for_candidate` (lineage) and `state.per_program_tracked_scores`
(dense, integer-keyed scores), and infers failures for rejected proposals. Swapping
original ↔ ExTS is a one-object change, which the bundled HotpotQA harness uses to
compare across seeds.
