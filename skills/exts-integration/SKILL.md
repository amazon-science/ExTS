---
name: exts-integration
description: >-
  Integrate the ExTS Monte-Carlo tree-search selection policy into a codebase
  that has an iterative candidate-improvement loop. Use this when a system picks
  its next candidate to refine greedily (take the best), linearly (refine the
  last / best-of-N), from a Pareto front, or with an existing tree search, and
  you want to swap in or add ExTS as the selection strategy. Triggers: "add
  ExTS", "integrate ExTS", "tree search selection", "replace greedy/Pareto
  selection", "MCTS candidate selection", "adaptive branching search".
---

# Integrating ExTS

ExTS is a candidate-**selection policy** for validation-heavy iterative search.
It decides *which existing candidate to expand next* and *whether to widen or
deepen*, balancing exploitation of strong candidates against exploration of
under-tried ones. This skill integrates it into a host system.

The algorithm is a single dependency-free file, `exts.py` (also bundled at
`assets/exts.py` in this skill). Its full contract and semantics are in
[`references/api-contract.md`](references/api-contract.md).

## Step 1 — Locate the host's selection point

Find the loop that repeatedly (a) **chooses** an existing candidate and (b)
**generates** a refined candidate from it, then **scores** and stores it. You are
replacing/augmenting step (a) only. Grep for terms like `select`, `best`,
`argmax`, `pareto`, `frontier`, `sample`, `candidate`, `population`, `rollout`,
`iteration`, `expand`.

Answer one question: **does the host already have its own tree-search
abstraction** (a class with select/expand/backpropagate or step/ask/tell), or is
selection a flat pick over a list/frontier?

## Step 2 — Choose the pattern

- **Host exposes a pluggable selector/strategy object** (a `select_*` / `choose_*`
  protocol you implement and hand to the host) → **Injected-selector controller.**
  Implement the protocol, keep an `ExTSTree` inside it, self-sync the tree from the
  host's `state` on each call, and return `tree.select()` — with **zero edits to the
  host's loop**. This is the cleanest integration when it's available.
  → Follow [`references/controller-pattern.md`](references/controller-pattern.md)
  ("Variant: integrating through a host extension point").

- **Flat / greedy / linear / best-of-N / Pareto selection you control → Controller pattern.**
  Drop `exts.py` in, keep the host's generate+score code untouched, and drive an
  `ExTSTree` from the loop yourself. This is the common case.
  → Follow [`references/controller-pattern.md`](references/controller-pattern.md).

- **Host has its own tree-search framework → Framework-port pattern.**
  Re-implement ExTS's scoring/selection against the host's algorithm interface so
  it becomes a sibling of the host's existing algorithms.
  → Follow [`references/port-pattern.md`](references/port-pattern.md).

## Step 3 — Preserve the original behind a toggle

**Never delete the original selector.** Add ExTS alongside it, selected by a flag
(e.g. `--selection-strategy {original,exts}` or a config field) **or**, when the
host takes a strategy object, simply by which selector you pass to its injection
point — no flag needed. This keeps the integration reviewable, makes
original-vs-ExTS an apples-to-apples comparison, and is exactly how the worked
examples are structured. The original path stays the default (it is the host's
default) unless the user says otherwise.

## Step 4 — Map identities and rewards

ExTS works in integer `program_idx` space. Reuse the host's existing candidate
IDs if they are dense integers from 0; otherwise keep a `host_id ↔ idx` dict.
Reward semantics that must be correct:

- A **successful** expansion (a usable, scored candidate) → `add_child(...)` then
  `backpropagate_success(child_idx, raw_score)`.
- A **failed** expansion (no candidate produced / invalid) → `backpropagate_failure(parent_idx)`.
  If your loop *always* produces a scored child, you simply never call this. But if
  you only observe the host's **state** (an injected selector) and never see rejected
  proposals, *infer* failures: remember the parent you returned and, on the next
  sync, `backpropagate_failure` any that produced no new child — otherwise the
  `(n⁺/n)^α` scaler is stuck at 1.0 in reject-heavy loops.
- Whenever the host's per-candidate scores change, call `sync_scores(list_of_scores)`
  so gating and progressive widening use current rankings.
- **Multiple parents.** `add_child` is single-parent; for candidates with several
  parents (merge / crossover), attach under the primary / first parent.

Use the host's **raw** score as `raw_score` (ExTS normalizes internally). Pass
the host's current best-known score as `program_score`. If the host's candidate IDs
are already dense integers from 0, they map straight onto `program_idx` with no
translation.

## Step 5 — Tune and verify

Start from the defaults. Pick starting hyperparameters from
the paper ([arXiv:2608.23848](https://arxiv.org/abs/2608.23848)) based on your task's failure rate,
score spread, and refinement diversity. Then:

- Add/extend a test that runs a handful of iterations end-to-end on both the
  original and the ExTS path and asserts the loop completes and produces a best
  candidate.
- Confirm `select()` never returns an unknown index and that the tree's total
  visits equal the number of backprop calls.
- **Comparing original vs ExTS.** Sweep several seeds (e.g. 0 / 42 / 1024),
  building a fresh `ExTSTree(rng=random.Random(seed))` per run; keep model, keys,
  budget, and data subset in environment variables; write each run's score to a
  git-ignored results dir and aggregate mean ± std. When you *inject* a selector the
  host's own `seed` may not reach it, so seed the `random.Random` yourself.

## Worked examples

Concrete, reviewed integrations live in the repo's `examples/`:

- **Injected-selector controller:** `examples/gepa-lib` (production GEPA *library* —
  ExTS as an `ExTSCandidateSelector` object passed to `optimize`, self-syncing from
  host state, **zero core edits**).
- **Controller (edit the loop):** `examples/gepa` (GEPA paper *artifact* — Pareto →
  ExTS via a `candidate_selection_strategy` flag + `exts_kwargs`), `examples/kmse`
  (MCTS-R → ExTS via `--mcts_variant`, with an adapter class mirroring the original
  `run()`).
- **Framework port:** `examples/treequest` (ExTS implemented as a `treequest`
  `Algorithm` sibling of AB-MCTS).

Read the one closest to your host before writing code.
