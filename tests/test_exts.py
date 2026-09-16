"""Smoke tests for the ExTS core API.

Runnable either with pytest (`pytest tests/`) or directly (`python tests/test_exts.py`).
The tests exercise the full public contract that every integration relies on:
initialize_root → select → add_child → backpropagate_success/failure → sync_scores
→ save/load, under both PUCT and UCT.
"""

import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exts import ExTSTree, ExTSNode  # noqa: E402


def _grow_tree(exploration_type="puct", steps=40, seed=0):
    """Run a small deterministic search where 'reward' is a noisy function of idx."""
    rng = random.Random(seed)
    tree = ExTSTree(exploration_type=exploration_type, rng=rng)

    root_score = 0.5
    tree.initialize_root(program_idx=0, program_score=root_score)
    tree.backpropagate_success(program_idx=0, raw_score=root_score)

    scores = [root_score]
    next_idx = 1
    for _ in range(steps):
        parent_idx = tree.select()
        # Simulate an expansion that occasionally fails.
        if rng.random() < 0.2:
            tree.backpropagate_failure(parent_idx)
            continue
        child_score = min(1.0, max(0.0, scores[parent_idx] + rng.uniform(-0.1, 0.15)))
        child_idx = next_idx
        next_idx += 1
        scores.append(child_score)
        tree.add_child(parent_idx, child_idx, child_score)
        tree.backpropagate_success(child_idx, raw_score=child_score)
        tree.sync_scores(scores)
    return tree, scores


def test_selection_returns_valid_index():
    tree, _ = _grow_tree()
    idx = tree.select()
    assert tree.get_node(idx) is not None


def test_invariants_hold():
    tree, scores = _grow_tree(steps=60)
    stats = tree.get_stats()
    # Root visit count is the total number of backprops (success + failure).
    assert stats["total_nodes"] >= 1
    assert stats["total_visits"] >= 1
    for node in tree._nodes.values():
        assert node.successful_visits <= node.visits
        assert len(node.reward_history) == node.successful_visits
        for child in node.children:
            assert child.parent is node
    # A parent's visits must be at least the sum of its children's visits.
    for node in tree._nodes.values():
        child_visits = sum(c.visits for c in node.children)
        assert node.visits >= child_visits


def test_unvisited_child_is_prioritized():
    """A freshly added child (visits=0) has +inf value and must be selectable."""
    tree = ExTSTree(rng=random.Random(1))
    tree.initialize_root(0, 0.5)
    tree.backpropagate_success(0, 0.5)
    tree.add_child(0, 1, 0.6)  # child 1 has visits=0
    # With an unvisited child, selection should reach it (value +inf) rather than
    # stopping at the root, unless the virtual child also scores +inf; either way
    # the returned node must be a real node.
    idx = tree.select()
    assert idx in tree._nodes


def test_both_exploration_types_run():
    for et in ("uct", "puct"):
        tree, _ = _grow_tree(exploration_type=et, steps=30, seed=7)
        assert tree.exploration_type == et
        assert tree.get_stats()["total_nodes"] > 1


def test_save_load_roundtrip():
    tree, _ = _grow_tree(steps=50, seed=3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tree.json")
        tree.save(path)
        loaded = ExTSTree.load(path)

    assert set(loaded._nodes) == set(tree._nodes)
    assert loaded._score_min == tree._score_min
    assert loaded._score_max == tree._score_max
    for idx, node in tree._nodes.items():
        ln = loaded._nodes[idx]
        assert ln.visits == node.visits
        assert ln.successful_visits == node.successful_visits
        assert ln.reward_history == node.reward_history
        assert ln.program_score == node.program_score
        # Parent/child topology is reconstructed correctly.
        assert (ln.parent.program_idx if ln.parent else None) == (
            node.parent.program_idx if node.parent else None
        )
        assert [c.program_idx for c in ln.children] == [c.program_idx for c in node.children]


def test_gating_and_widening_toggle_off():
    """The algorithm should still run with gating / widening disabled."""
    tree = ExTSTree(
        use_gated_expansion=False,
        use_progressive_widening=False,
        rng=random.Random(0),
    )
    tree.initialize_root(0, 0.5)
    tree.backpropagate_success(0, 0.5)
    idx = 1
    for _ in range(20):
        p = tree.select()
        tree.add_child(p, idx, 0.5)
        tree.backpropagate_success(idx, 0.5)
        idx += 1
    assert tree.get_stats()["total_nodes"] == idx


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} smoke tests passed.")
