"""MCTS / UCT correctness tests."""

import math
from dataclasses import replace

import pytest

from ai.mcts import MCTSAgent, MCTSNode, uct_value
from config import DEFAULT_CONFIG, MCTSConfig
from game.actions import Action, is_legal
from game.rules import initial_state
from game.simulation import simulate_turn
from tests.conftest import make_state

SMALL = MCTSConfig(iterations=60, time_budget=60.0, rollout_depth=1, candidate_width=24)


def test_uct_formula():
    assert uct_value(3.0, 5, 20, 1.0) == pytest.approx(0.6 + math.sqrt(math.log(20) / 5))
    assert uct_value(0.0, 0, 10, 1.0) == math.inf
    # exploitation dominates with c = 0
    assert uct_value(4.0, 5, 50, 0.0) == pytest.approx(0.8)


def test_backpropagation_uses_mover_perspective(classic):
    s = initial_state(classic)
    root = MCTSNode(s, None, None, mover=1)            # root: player 0 to move
    child = MCTSNode(s, root, None, mover=0)           # move made by player 0
    grand = MCTSNode(s, child, None, mover=1)          # reply by player 1
    MCTSAgent.backpropagate(grand, 0.8, root_player=0)
    assert (root.visits, child.visits, grand.visits) == (1, 1, 1)
    assert child.value_sum == pytest.approx(0.8)       # good for player 0
    assert grand.value_sum == pytest.approx(0.2)       # same outcome seen by player 1
    assert root.value_sum == pytest.approx(0.2)


def test_selection_picks_highest_uct(classic):
    s = initial_state(classic)
    agent = MCTSAgent(0, replace(SMALL, exploration=0.0, widening_c=0.5, widening_alpha=0.0))
    agent._prepare(s)
    root = MCTSNode(s, None, None, 1)
    root.untried = []                                  # fully expanded
    a, b = MCTSNode(s, root, None, 0), MCTSNode(s, root, None, 0)
    a.visits, a.value_sum = 4, 1.0
    b.visits, b.value_sum = 4, 3.0
    root.children = [a, b]
    root.visits = 8
    b.untried = []
    picked, _ = agent.select(root)
    assert picked is b


def test_expansion_adds_one_child(classic):
    s = initial_state(classic)
    agent = MCTSAgent(0, SMALL)
    agent._prepare(s)
    root = MCTSNode(s, None, None, 1)
    n_untried = len(agent._untried(root))
    first = root.untried[0]
    child = agent.expand(root)
    assert child.action == first
    assert child.mover == 0 and child.parent is root
    assert len(root.children) == 1 and len(root.untried) == n_untried - 1
    assert child.state.to_move == 1


def test_visit_counts_are_consistent(classic):
    s = initial_state(classic)
    agent = MCTSAgent(0, SMALL, seed=3)
    d = agent.decide(s)
    root = agent.last_root
    assert root.visits == d.stats["iterations"] == SMALL.iterations
    assert sum(c.visits for c in root.children) == root.visits
    assert d.stats["selected_visits"] == max(c.visits for c in root.children)
    assert 0.0 <= d.stats["selected_reward"] <= 1.0

    def check(node):
        if node.children:
            assert node.visits >= sum(c.visits for c in node.children)
        for c in node.children:
            check(c)
    check(root)


def test_progressive_widening_limits_children(classic):
    s = initial_state(classic)
    agent = MCTSAgent(0, SMALL, seed=1)
    agent.decide(s)
    root = agent.last_root
    assert len(root.children) <= math.ceil(SMALL.widening_c * root.visits ** SMALL.widening_alpha)


def test_terminal_reward_values(classic):
    agent = MCTSAgent(0, SMALL)
    s = make_state([(0, 0.0, 0.0), (1, 0.0, 5.0)])
    assert agent.terminal_reward(s.with_changes(winner=0)) == 1.0
    assert agent.terminal_reward(s.with_changes(winner=1)) == 0.0
    assert agent.terminal_reward(s.with_changes(winner=-1)) == 0.5


def test_legal_action_and_fixed_seed_reproducibility(classic):
    s = initial_state(classic)
    d1 = MCTSAgent(0, SMALL, seed=11).decide(s)
    d2 = MCTSAgent(0, SMALL, seed=11).decide(s)
    assert is_legal(s, d1.action, DEFAULT_CONFIG.rules)
    assert d1.action == d2.action
    for k in ("iterations", "simulations", "selected_visits", "selected_reward", "avg_rollout_depth"):
        assert d1.stats[k] == d2.stats[k]


def test_tiny_budgets(classic):
    s = initial_state(classic)
    d = MCTSAgent(0, replace(SMALL, iterations=1)).decide(s)
    assert is_legal(s, d.action, DEFAULT_CONFIG.rules) and not d.fallback
    assert d.stats["iterations"] == 1
    d0 = MCTSAgent(0, replace(SMALL, iterations=0)).decide(s)
    assert d0.fallback and is_legal(s, d0.action, DEFAULT_CONFIG.rules)
    dt = MCTSAgent(0, SMALL).decide(s, time_budget=0.0)
    assert is_legal(s, dt.action, DEFAULT_CONFIG.rules)


def test_finds_immediate_win(classic):
    s = make_state([(0, 0.0, -4.0), (0, 5.0, 0.0), (1, 8.6, 0.0)])
    for seed in (1, 2, 3):
        d = MCTSAgent(0, replace(SMALL, iterations=120), seed=seed).decide(s)
        assert simulate_turn(s, d.action, classic).state.winner == 0


def test_terminal_children_are_scored_exactly(classic):
    s = make_state([(0, 0.0, -4.0), (0, 5.0, 0.0), (1, 8.6, 0.0)])
    agent = MCTSAgent(0, replace(SMALL, iterations=80), seed=5)
    agent.decide(s)
    wins = [c for c in agent.last_root.children if c.state.winner == 0]
    assert wins and all(c.mean == 1.0 for c in wins)


def test_simulation_statistics(classic):
    s = initial_state(classic)
    agent = MCTSAgent(0, SMALL, seed=2)
    before = agent.simulations
    d = agent.decide(s)
    assert d.stats["simulations"] == agent.simulations - before > 0
    assert 0 <= d.stats["avg_rollout_depth"] <= SMALL.rollout_depth
    assert d.stats["root_children"] == len(agent.last_root.children)
