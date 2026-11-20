"""Minimax / Alpha-Beta correctness tests (synthetic trees + real game states)."""

import math
import random
from dataclasses import replace

import pytest

from ai.evaluation import WIN_SCORE, evaluate
from ai.minimax import MinimaxAgent, SearchStats, alphabeta, minimax
from config import DEFAULT_CONFIG, MinimaxConfig
from game.actions import is_legal
from game.arena import get_arena
from game.rules import initial_state
from game.simulation import simulate_turn
from tests.conftest import make_state


class TreeProblem:
    """Explicit game tree. A node is either a number (leaf) or a list of children.
    Optional edge values: children given as (edge, subtree) tuples."""

    def __init__(self, root_is_max=True):
        self.root_is_max = root_is_max

    def is_terminal(self, node):
        return not isinstance(node[1], list)

    def evaluate(self, node):
        return node[1]

    def is_max(self, node):
        return node[0] % 2 == (0 if self.root_is_max else 1)

    def actions(self, node, depth):
        return list(range(len(node[1])))

    def result(self, node, i):
        ch = node[1][i]
        if isinstance(ch, tuple):
            e, sub = ch
        else:
            e, sub = 0.0, ch
        return e, (node[0] + 1, sub)


def wrap(tree):
    return (0, tree)


def random_tree(rng, depth, branch, edges=False):
    if depth == 0:
        return rng.randint(-50, 50)
    kids = [random_tree(rng, depth - 1, branch, edges) for _ in range(rng.randint(1, branch))]
    if edges:
        kids = [(rng.uniform(-5, 5), k) for k in kids]
    return kids


def test_textbook_tree_value_and_pruning():
    # Classic example (Russell & Norvig fig. 5.5): MAX root, three MIN children.
    tree = [[3, 12, 8], [2, 4, 6], [14, 5, 2]]
    p = TreeProblem()
    s1, s2 = SearchStats(), SearchStats()
    assert minimax(p, wrap(tree), 2, s1) == 3
    assert alphabeta(p, wrap(tree), 2, -math.inf, math.inf, s2) == 3
    assert s2.branches_pruned == 2          # the 4 and 6 under the second MIN node
    assert s2.nodes_evaluated < s1.nodes_evaluated


def test_minimizing_root():
    tree = [[3, 12], [2, 4]]                # MIN root over MAX children -> min(12, 4)
    assert minimax(TreeProblem(root_is_max=False), wrap(tree), 2, SearchStats()) == 4


@pytest.mark.parametrize("edges", [False, True])
def test_alphabeta_matches_minimax_on_random_trees(edges):
    rng = random.Random(1234 + edges)
    total_pruned = 0
    for _ in range(300):
        depth = rng.randint(1, 5)
        tree = random_tree(rng, depth, 4, edges)
        p = TreeProblem(root_is_max=rng.random() < 0.5)
        s1, s2 = SearchStats(), SearchStats()
        v1 = minimax(p, wrap(tree), depth, s1)
        v2 = alphabeta(p, wrap(tree), depth, -math.inf, math.inf, s2)
        assert v2 == pytest.approx(v1)
        assert s2.nodes_visited <= s1.nodes_visited
        total_pruned += s2.branches_pruned
    assert total_pruned > 0


def test_depth_limit_uses_evaluation():
    tree = [[1, 2], [3, 4]]
    p = TreeProblem()
    # depth 0 -> evaluate the root itself, which is a list -> treat via subclass
    class P(TreeProblem):
        def evaluate(self, node):
            return 99 if isinstance(node[1], list) else node[1]
    assert minimax(P(), wrap(tree), 1, SearchStats()) == 99


# ----------------------------------------------------------------------------- real game
SMALL = MinimaxConfig(max_depth=2, root_width=20, inner_width=8, node_budget=3000, time_budget=30.0)


def test_terminal_state_evaluation(classic):
    s = make_state([(0, 0.0, 0.0), (1, 30.0, 0.0, False)]).with_changes(winner=0)
    assert evaluate(s, 0) >= WIN_SCORE
    assert evaluate(s, 1) <= -WIN_SCORE
    assert evaluate(s.with_changes(winner=-1), 0) == 0.0


def test_nonterminal_scores_are_bounded():
    s = initial_state(get_arena("neon"), per_player=6)
    assert abs(evaluate(s, 0)) < WIN_SCORE / 4


def test_prefers_immediate_win(classic):
    s = make_state([(0, 0.0, -4.0), (0, 5.0, 0.0), (1, 8.6, 0.0)])
    agent = MinimaxAgent(0, SMALL)
    d = agent.decide(s)
    assert is_legal(s, d.action, DEFAULT_CONFIG.rules)
    res = simulate_turn(s, d.action, classic)
    assert res.state.winner == 0
    assert d.stats["evaluation"] >= WIN_SCORE - 1000


def test_avoids_obvious_self_elimination(classic):
    # Player 1 has one marble at the rim. Most shots that move it outward lose.
    s = make_state([(0, 0.0, -9.0), (1, 0.0, 4.0), (1, 3.0, 4.0)])
    from game.actions import generate_candidate_actions
    suicidal = [a for a in generate_candidate_actions(s, DEFAULT_CONFIG.rules)
                if simulate_turn(s, a, classic).state.winner == 1]
    assert suicidal                                   # the trap is real: some candidates lose at once
    d = MinimaxAgent(0, SMALL).decide(s)
    res = simulate_turn(s, d.action, classic)
    assert res.state.winner != 1
    # A shot that loses at once is worth -(WIN + margin) minus its own fuzzy-risk edge penalty
    # (risk ~0.95 for a self-elimination). The chosen line may still be a *later* forced loss,
    # which is worth more than that.
    immediate_loss = -(WIN_SCORE + 50 * 2 - 1)
    assert d.stats["evaluation"] > immediate_loss - 0.9 * SMALL.fuzzy_weight


def test_minimizing_reply_is_considered(classic):
    """At depth 2 the backed-up value must not exceed the depth-1 optimistic value."""
    s = initial_state(classic)
    d1 = MinimaxAgent(0, replace(SMALL, max_depth=1)).decide(s)
    d2 = MinimaxAgent(0, SMALL).decide(s)
    assert d2.stats["depth_reached"] == 2
    assert d2.stats["evaluation"] <= d1.stats["evaluation"] + 1e-9


def test_search_statistics_are_consistent(classic):
    s = initial_state(classic)
    agent = MinimaxAgent(0, replace(SMALL, max_depth=1, root_width=10))
    sims_before = agent.simulations
    d = agent.decide(s)
    st = d.stats
    assert st["depth_reached"] == 1
    assert st["nodes_visited"] == 1 + 10          # root + its 10 children
    assert st["nodes_evaluated"] == 10
    assert st["simulations"] == agent.simulations - sims_before == 10
    d2 = MinimaxAgent(0, SMALL).decide(s)
    assert d2.stats["branches_pruned"] > 0 and d2.stats["cutoffs"] > 0
    assert d2.stats["nodes_visited"] >= d2.stats["nodes_evaluated"]


def test_alphabeta_agrees_with_plain_minimax_on_game_state(classic):
    """Run the real marble problem through both search functions."""
    from ai.base_agent import Clock
    from ai.minimax import MarbleSearchProblem
    s = make_state([(0, 0.0, -5.0), (0, 2.0, -4.0), (1, 0.0, 5.0), (1, -2.0, 6.5)])
    cfg = replace(SMALL, root_width=8, inner_width=5, node_budget=10**6)
    agent = MinimaxAgent(0, cfg)
    agent._prepare(s)
    p = MarbleSearchProblem(agent, 0, Clock(None), cfg)
    p.root_state = s
    v_mm = minimax(p, s, 2, SearchStats())
    st = SearchStats()
    v_ab = alphabeta(p, s, 2, -math.inf, math.inf, st)
    assert v_ab == pytest.approx(v_mm)
    assert st.branches_pruned > 0


def test_node_budget_limits_search_and_falls_back(classic):
    s = initial_state(classic)
    d = MinimaxAgent(0, replace(SMALL, node_budget=3)).decide(s)
    assert d.stats["budget_exhausted"]
    assert d.fallback
    assert is_legal(s, d.action, DEFAULT_CONFIG.rules)
    assert d.stats["simulations"] <= 4 + 1   # 3 budgeted + the risk lookup of the chosen move


def test_time_budget_zero_still_returns_legal_action(classic):
    s = initial_state(classic)
    d = MinimaxAgent(0, SMALL).decide(s, time_budget=0.0)
    assert is_legal(s, d.action, DEFAULT_CONFIG.rules)


def test_iterative_deepening_keeps_last_complete_depth(classic):
    s = initial_state(classic)
    d = MinimaxAgent(0, replace(SMALL, max_depth=3, root_width=20, inner_width=8, node_budget=200)).decide(s)
    assert d.stats["budget_exhausted"]
    assert d.stats["depth_reached"] in (1, 2)
    assert not d.fallback


def test_wrong_player_rejected(classic):
    with pytest.raises(ValueError):
        MinimaxAgent(1, SMALL).decide(initial_state(classic))
