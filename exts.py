"""
ExTS (Exploit More, Explore Smarter for Budget-Constrained Agentic Search): MCTS for validation-heavy search.

Score formula:
    score(v) = (n_v+ / n_v)^α · mean(φ(norm(r_i))) + exploration(v)

    UCT exploration:  C · √(ln(n_parent) / n_v)
    PUCT exploration: C · n_parent^p / (1 + n_v)

Each node stores its list of raw reward observations. At selection time,
each observation is individually min-max normalized using the tree's
current global [score_min, score_max], shaped through φ, then averaged.

Features:
- Per-observation exponential temperature shaping with current global range
- Alpha-controlled success rate scaler
- Gated expansion (quantile-based)
- Gated progressive widening
- Virtual child with bootstrap-style sampled prior

See the paper (arXiv:2608.23848) for the derivation, design rationale,
and per-dataset hyperparameter guidance.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Optional

# ── Defaults ─────────────────────────────────────────────────────────────
DEFAULT_UCT_EXPLORATION_CONSTANT = 1.414
DEFAULT_PUCT_EXPLORATION_CONSTANT = 1.0
DEFAULT_SCORE_TEMPERATURE = 0.3
DEFAULT_SUCCESS_RATE_ALPHA = 1.414
DEFAULT_MAX_CHILDREN = 3
DEFAULT_EXPANSION_GATE_QUANTILE = 0.25
DEFAULT_USE_GATED_EXPANSION = True
DEFAULT_USE_PROGRESSIVE_WIDENING = True
DEFAULT_WIDENING_BASE = 2.0
DEFAULT_WIDENING_VISIT_THRESHOLD = 32
DEFAULT_WIDENING_SCORE_QUANTILE = 0.75
DEFAULT_EXPLORATION_TYPE = "puct"
DEFAULT_PUCT_PARENT_EXP = 0.5


@dataclass
class ExTSNode:
    """A node in the ExTS tree."""

    program_idx: int
    parent: Optional[ExTSNode] = None
    children: list[ExTSNode] = field(default_factory=list)

    visits: int = 0
    successful_visits: int = 0
    reward_history: list[float] = field(default_factory=list)

    # Latest score for gating and progressive widening decisions
    program_score: Optional[float] = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class ExTSTree:
    """ExTS tree for candidate selection in validation-heavy search.

    Parameters:
        exploration_constant: Exploration weight C. Typical values:
            UCT:  ~1.414 (√2). Range [0.5, 2.0].
            PUCT: ~1.0. Range [0.5, 1.5]. Lower than UCT because PUCT
                  lacks a policy prior P(s,a) that would dampen exploration.
        score_temperature: Shaping function convexity (lower = sharper).
        success_rate_alpha: Exponent on (n+/n). 0=ignore failures, 1=linear, >1=amplified.
        max_children: Base max children per node M_0.
        expansion_gate_quantile: Quantile threshold for gated expansion.
        use_gated_expansion: Enable expansion gating.
        use_progressive_widening: Enable gated progressive widening.
        widening_base: Log base for progressive widening bonus.
        widening_visit_threshold: Visits before widening starts.
        widening_score_quantile: Score quantile gate for PW bonus.
        exploration_type: "uct" or "puct".
            UCT:  C · √(ln(n_parent) / n_v)        — logarithmic decay.
            PUCT: C · n_parent^p / (1 + n_v)       — linear decay, faster convergence.
        puct_parent_exp: Exponent p on n_parent for PUCT. Default 0.5.
    """

    def __init__(
        self,
        exploration_constant: Optional[float] = None,
        score_temperature: float = DEFAULT_SCORE_TEMPERATURE,
        success_rate_alpha: float = DEFAULT_SUCCESS_RATE_ALPHA,
        max_children: int = DEFAULT_MAX_CHILDREN,
        expansion_gate_quantile: float = DEFAULT_EXPANSION_GATE_QUANTILE,
        use_gated_expansion: bool = DEFAULT_USE_GATED_EXPANSION,
        use_progressive_widening: bool = DEFAULT_USE_PROGRESSIVE_WIDENING,
        widening_base: float = DEFAULT_WIDENING_BASE,
        widening_visit_threshold: int = DEFAULT_WIDENING_VISIT_THRESHOLD,
        widening_score_quantile: float = DEFAULT_WIDENING_SCORE_QUANTILE,
        exploration_type: str = DEFAULT_EXPLORATION_TYPE,
        puct_parent_exp: float = DEFAULT_PUCT_PARENT_EXP,
        rng: Optional[random.Random] = None,
    ):
        if exploration_constant is None:
            if exploration_type == "puct":
                exploration_constant = DEFAULT_PUCT_EXPLORATION_CONSTANT
            else:
                exploration_constant = DEFAULT_UCT_EXPLORATION_CONSTANT
        self.exploration_constant = exploration_constant
        self.score_temperature = score_temperature
        self.success_rate_alpha = success_rate_alpha
        self.max_children = max_children
        self.expansion_gate_quantile = expansion_gate_quantile
        self.use_gated_expansion = use_gated_expansion
        self.use_progressive_widening = use_progressive_widening
        self.widening_base = widening_base
        self.widening_visit_threshold = widening_visit_threshold
        self.widening_score_quantile = widening_score_quantile
        self.exploration_type = exploration_type  # "uct" or "puct"
        self.puct_parent_exp = puct_parent_exp  # exponent on n_parent for PUCT
        self.rng = rng or random.Random()

        self._nodes: dict[int, ExTSNode] = {}
        self.root: Optional[ExTSNode] = None

        # Global raw score bounds, updated on each successful backprop
        self._score_min: float = float("inf")
        self._score_max: float = float("-inf")

    # ── Shaping ──────────────────────────────────────────────────────────

    def _phi(self, x: float) -> float:
        """φ(x) = (e^(x/T) - 1) / (e^(1/T) - 1), x ∈ [0, 1]."""
        T = self.score_temperature
        if T <= 0:
            return x
        exp_inv_T = math.exp(1.0 / T)
        return (math.exp(x / T) - 1.0) / (exp_inv_T - 1.0)

    def _shaped_mean(self, node: ExTSNode) -> float:
        """Mean of individually φ-shaped, min-max-normalized rewards."""
        if not node.reward_history:
            return 0.0
        if self._score_max <= self._score_min:
            return 1.0
        total = 0.0
        for r in node.reward_history:
            x = (r - self._score_min) / (self._score_max - self._score_min)
            x = max(0.0, min(1.0, x))
            total += self._phi(x)
        return total / len(node.reward_history)

    # ── Score helpers ────────────────────────────────────────────────────

    def _compute_score_quantile(self, quantile: float) -> Optional[float]:
        """Quantile over program_scores of all nodes."""
        scores = sorted(
            n.program_score
            for n in self._nodes.values()
            if n.program_score is not None
        )
        if not scores:
            return None
        idx = min(int(len(scores) * quantile), len(scores) - 1)
        return scores[idx]

    # ── Public API ───────────────────────────────────────────────────────

    def initialize_root(self, program_idx: int, program_score: float) -> None:
        root = ExTSNode(program_idx=program_idx, program_score=program_score)
        self._nodes[program_idx] = root
        self.root = root

    def select(self) -> int:
        """Select a node for expansion. Returns program_idx."""
        node = self._uct_select(self.root)
        return node.program_idx

    def add_child(
        self,
        parent_program_idx: int,
        child_program_idx: int,
        child_program_score: float,
    ) -> ExTSNode:
        parent = self._nodes[parent_program_idx]
        child = ExTSNode(
            program_idx=child_program_idx,
            parent=parent,
            program_score=child_program_score,
        )
        parent.children.append(child)
        self._nodes[child_program_idx] = child
        return child

    def backpropagate_success(self, program_idx: int, raw_score: float) -> None:
        self._score_min = min(self._score_min, raw_score)
        self._score_max = max(self._score_max, raw_score)

        node = self._nodes[program_idx]
        while node is not None:
            node.visits += 1
            node.successful_visits += 1
            node.reward_history.append(raw_score)
            node = node.parent

    def backpropagate_failure(self, program_idx: int) -> None:
        node = self._nodes[program_idx]
        while node is not None:
            node.visits += 1
            node = node.parent

    def sync_scores(self, per_program_scores: list[float]) -> None:
        for prog_idx, node in self._nodes.items():
            if prog_idx < len(per_program_scores):
                node.program_score = per_program_scores[prog_idx]

    def get_node(self, program_idx: int) -> Optional[ExTSNode]:
        return self._nodes.get(program_idx)

    # ── UCT scoring ──────────────────────────────────────────────────────

    def _exploration_term(self, parent_visits: int, node_visits: float) -> float:
        """Compute exploration bonus.

        UCT:  C · √(ln(n_parent) / n_v)
        PUCT: C · n_parent^p / (1 + n_v)   where p = puct_parent_exp
        """
        if self.exploration_type == "puct":
            return self.exploration_constant * (parent_visits ** self.puct_parent_exp) / (1 + node_visits)
        return self.exploration_constant * math.sqrt(
            math.log(parent_visits) / node_visits
        )

    def _uct_value(self, node: ExTSNode) -> float:
        if node.visits == 0:
            return float("inf")

        parent_visits = max(1, node.parent.visits) if node.parent else 1

        if node.successful_visits > 0 and self._score_max > self._score_min:
            shaped = self._shaped_mean(node)
            success_rate = node.successful_visits / node.visits
            exploitation = (success_rate ** self.success_rate_alpha) * shaped
        elif node.successful_visits > 0:
            success_rate = node.successful_visits / node.visits
            exploitation = success_rate ** self.success_rate_alpha
        else:
            exploitation = 0.0

        exploration = self._exploration_term(parent_visits, node.visits)
        return exploitation + exploration

    def _virtual_child_uct(self, node: ExTSNode) -> float:
        """Virtual child UCT modeling a subtree with n_fair visits.

        Samples k = round(n_fair × success_rate) scores from the parent's
        reward pool (reward_history + program_score) to estimate what a young
        subtree would look like. Sampling with replacement preserves
        bootstrap-style stochasticity at small pool sizes.
        """
        parent_visits = max(1, node.visits)
        m = len(node.children)

        # n_fair = average visit count among existing children, so the virtual
        # child's UCT is directly comparable to a real child at the same
        # maturity level (dividing by m rather than m+1 avoids deflating the
        # baseline and over-biasing toward expansion).
        children_visits = sum(c.visits for c in node.children)
        n_fair = max(1, children_visits / m)

        q_prior = 0.0
        if node.reward_history and self._score_max > self._score_min:
            success_rate = node.successful_visits / node.visits if node.visits > 0 else 1.0
            k = max(1, round(n_fair * success_rate))

            # Include program_score in the pool so a child inherits some of the
            # parent's tracked static quality; its weight is 1/(len(history)+1),
            # which naturally decays as more rewards accumulate. Sampling is with
            # replacement to preserve stochasticity when the pool is small.
            pool = list(node.reward_history)
            if node.program_score is not None:
                pool.append(node.program_score)

            samples = self.rng.choices(pool, k=k)
            total = 0.0
            for r in samples:
                x = (r - self._score_min) / (self._score_max - self._score_min)
                x = max(0.0, min(1.0, x))
                total += self._phi(x)
            sampled_mean = total / k
            q_prior = (success_rate ** self.success_rate_alpha) * sampled_mean

        # Fallback when all scores are identical (_score_max == _score_min):
        # mirror the real-child exploitation term (success_rate^alpha) so the
        # virtual vs real comparison stays fair and expansion is not blocked.
        elif node.successful_visits > 0 and node.visits > 0:
            success_rate = node.successful_visits / node.visits
            q_prior = success_rate ** self.success_rate_alpha

        return q_prior + self._exploration_term(parent_visits, n_fair)

    # ── Selection ────────────────────────────────────────────────────────

    def _uct_select(self, node: ExTSNode) -> ExTSNode:
        if node.is_leaf:
            return node

        can_expand = self._can_expand(node)

        # Random tiebreaker ensures uniform exploration among equally-scored
        # children — in particular unvisited children, which all score UCT=inf
        # and would otherwise be starved in favor of the first-added sibling.
        best_child = max(node.children, key=lambda c: (self._uct_value(c), self.rng.random()))
        best_child_uct = self._uct_value(best_child)

        if can_expand:
            virtual_uct = self._virtual_child_uct(node)
            if virtual_uct > best_child_uct:
                return node

        return self._uct_select(best_child)

    def _can_expand(self, node: ExTSNode) -> bool:
        if len(node.children) >= self._max_children_for_node(node):
            return False

        if not self.use_gated_expansion or node is self.root:
            return True

        q_alpha = self._compute_score_quantile(self.expansion_gate_quantile)
        if q_alpha is None:
            return True

        if node.program_score is None:
            return False

        return node.program_score >= q_alpha

    def _max_children_for_node(self, node: ExTSNode) -> int:
        """M(v) = M_0 + ⌊log_b(n_v / n_0)⌋ · 𝟙[s_v ≥ Q_γ ∨ root]."""
        if not self.use_progressive_widening:
            return self.max_children

        n_v = node.visits
        if n_v < self.widening_visit_threshold:
            return self.max_children

        bonus = int(
            math.log(n_v / self.widening_visit_threshold)
            / math.log(self.widening_base)
        )

        if node is self.root:
            return self.max_children + bonus

        q_gamma = self._compute_score_quantile(self.widening_score_quantile)
        if (
            q_gamma is not None
            and node.program_score is not None
            and node.program_score >= q_gamma
        ):
            return self.max_children + bonus

        return self.max_children

    # ── Stats ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        all_nodes = list(self._nodes.values())
        return {
            "total_nodes": len(all_nodes),
            "total_visits": self.root.visits if self.root else 0,
            "max_depth": max(
                (self._node_depth(n) for n in all_nodes), default=0
            ),
            "leaf_count": sum(1 for n in all_nodes if n.is_leaf),
            "score_min": self._score_min if self._score_min != float("inf") else None,
            "score_max": self._score_max if self._score_max != float("-inf") else None,
        }

    def _node_depth(self, node: ExTSNode) -> int:
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        tree_data = {
            "config": {
                "exploration_constant": self.exploration_constant,
                "score_temperature": self.score_temperature,
                "success_rate_alpha": self.success_rate_alpha,
                "max_children": self.max_children,
                "expansion_gate_quantile": self.expansion_gate_quantile,
                "use_gated_expansion": self.use_gated_expansion,
                "use_progressive_widening": self.use_progressive_widening,
                "widening_base": self.widening_base,
                "widening_visit_threshold": self.widening_visit_threshold,
                "widening_score_quantile": self.widening_score_quantile,
                "exploration_type": self.exploration_type,
                "puct_parent_exp": self.puct_parent_exp,
            },
            "score_min": self._score_min if self._score_min != float("inf") else None,
            "score_max": self._score_max if self._score_max != float("-inf") else None,
            "root_idx": self.root.program_idx if self.root else None,
            "nodes": {
                str(prog_idx): {
                    "program_idx": node.program_idx,
                    "parent": node.parent.program_idx if node.parent else None,
                    "children": [c.program_idx for c in node.children],
                    "visits": node.visits,
                    "successful_visits": node.successful_visits,
                    "reward_history": node.reward_history,
                    "program_score": node.program_score,
                }
                for prog_idx, node in self._nodes.items()
            },
        }
        with open(filepath, "w") as f:
            json.dump(tree_data, f, indent=2)

    @staticmethod
    def load(filepath: str) -> ExTSTree:
        with open(filepath) as f:
            data = json.load(f)

        cfg = data["config"]
        exploration_type = cfg.get("exploration_type", DEFAULT_EXPLORATION_TYPE)
        default_c = DEFAULT_PUCT_EXPLORATION_CONSTANT if exploration_type == "puct" else DEFAULT_UCT_EXPLORATION_CONSTANT
        tree = ExTSTree(
            exploration_constant=cfg.get("exploration_constant", default_c),
            score_temperature=cfg.get("score_temperature", DEFAULT_SCORE_TEMPERATURE),
            success_rate_alpha=cfg.get("success_rate_alpha", DEFAULT_SUCCESS_RATE_ALPHA),
            max_children=cfg.get("max_children", DEFAULT_MAX_CHILDREN),
            expansion_gate_quantile=cfg.get("expansion_gate_quantile", DEFAULT_EXPANSION_GATE_QUANTILE),
            use_gated_expansion=cfg.get("use_gated_expansion", DEFAULT_USE_GATED_EXPANSION),
            use_progressive_widening=cfg.get("use_progressive_widening", DEFAULT_USE_PROGRESSIVE_WIDENING),
            widening_base=cfg.get("widening_base", DEFAULT_WIDENING_BASE),
            widening_visit_threshold=cfg.get("widening_visit_threshold", DEFAULT_WIDENING_VISIT_THRESHOLD),
            widening_score_quantile=cfg.get("widening_score_quantile", DEFAULT_WIDENING_SCORE_QUANTILE),
            exploration_type=exploration_type,
            puct_parent_exp=cfg.get("puct_parent_exp", DEFAULT_PUCT_PARENT_EXP),
        )

        if data.get("score_min") is not None:
            tree._score_min = data["score_min"]
        if data.get("score_max") is not None:
            tree._score_max = data["score_max"]

        nodes_data = data["nodes"]
        for key, nd in nodes_data.items():
            prog_idx = nd["program_idx"]
            node = ExTSNode(
                program_idx=prog_idx,
                visits=nd["visits"],
                successful_visits=nd["successful_visits"],
                reward_history=nd["reward_history"],
                program_score=nd["program_score"],
            )
            tree._nodes[prog_idx] = node

        for key, nd in nodes_data.items():
            prog_idx = nd["program_idx"]
            node = tree._nodes[prog_idx]
            if nd["parent"] is not None:
                node.parent = tree._nodes[nd["parent"]]
            node.children = [tree._nodes[c] for c in nd["children"]]

        tree.root = (
            tree._nodes[data["root_idx"]] if data["root_idx"] is not None else None
        )
        return tree
