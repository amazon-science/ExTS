# Framework-port pattern

Use this when the host **already has its own tree-search framework** — an
abstract algorithm interface (select/expand/backpropagate, or step/ask/tell) with
its own tree/node data structures — and you want ExTS to be selectable *beside*
the host's existing algorithms (e.g. next to AB-MCTS). Here you do not vendor
`exts.py` verbatim; instead you re-implement ExTS's scoring and selection logic
against the host's interface, reusing the host's tree.

Think of `exts.py` as the **reference specification** of the math (see
the paper, arXiv:2608.23848) and port it, not copy it.

## Recipe

1. **Study the host interface.** Identify the abstract base every algorithm
   implements and the shared tree/node types. Note whether the framework is
   *stateful* (the algorithm object holds the tree) or *functional/stateless*
   (state is threaded through each call and returned).

2. **Create a sibling algorithm class** implementing that interface. Hold ExTS's
   per-node bookkeeping (visits, successful_visits, reward_history, program_score)
   and the global `score_min/score_max` either on the node type (if extensible)
   or in a side-table keyed by node id inside the algorithm's state object.

3. **Port the four pieces of math** from `exts.py`, unchanged in behavior:
   - node value `_uct_value` (exploitation `(n⁺/n)^α · mean φ(norm(r))` +
     exploration UCT/PUCT),
   - the virtual-child expansion decision `_virtual_child_uct`,
   - gated expansion `_can_expand` and progressive widening `_max_children_for_node`,
   - success/failure backpropagation.
   Keep temperature shaping `_phi`, per-observation normalization, and random
   tie-breaking identical.

4. **Map select/expand/backprop onto the host's methods.** In a step-based
   interface: the "select" descent (with the virtual-child widen-vs-deepen
   decision) happens in the ask/select half; recording the generated candidate's
   score happens in the tell/backprop half.

5. **Register it.** Export the new class from the framework's public surface
   (its `__init__`/`__all__`) so users can pick it like any built-in algorithm.
   Add a visualization adapter and a test mirroring the existing algorithms if the
   framework has those.

## How the worked example does it — TreeQuest (`examples/treequest`)

- TreeQuest's abstract base is `Algorithm` (`src/treequest/algos/base.py`), which
  is **stateless**: all mutable state lives in an `AlgoStateT` object passed to
  and returned from each call. Algorithms implement `init_tree`, `step`,
  `ask_batch`, `tell`, and `get_state_score_pairs`; the shared data structures are
  `Node`/`Tree` (`algos/tree.py`) and `Trial`/`TrialStore` (`trial.py`).
- ExTS is added as `src/treequest/algos/exts.py` defining an `@dataclass`
  `ExTSState` (holding the `Tree`, `visit_counts`, `successful_visits`,
  `reward_histories`, `score_min/max`, and a `trial_store`) and a class
  `class ExTS(Algorithm[StateT, ExTSState[StateT]])`.
- The mapping: `ask_batch` runs the UCT descent + virtual-child decision to pick
  which node to expand (returning `Trial`s); `tell` folds a `(state, score)`
  result back in via success/failure backprop. `score == 0.0` is treated as a
  failed expansion (a framework convention), everything else as success.
- ExTS requires **no new dependencies** (stdlib `math`/`random` plus TreeQuest
  internals), so `pyproject.toml` is untouched. The only edit to a tracked file is
  exporting `ExTS` from `src/treequest/__init__.py` (`import` + `__all__`).

## Fidelity check

After porting, verify the ported ExTS reproduces the reference `exts.py` on a
small synthetic problem: same seed, same sequence of (parent, score) outcomes,
same selection sequence. Any divergence usually traces to (a) normalization
timing (normalize each observation lazily at selection, not at insertion), (b)
the `n_fair = children_visits / m` divisor in the virtual child, or (c) missing
random tie-breaking among equal-valued children.
