"""Plain-language decision explanations must come from measured numbers."""

import math

from ai.base_agent import Decision
from ai.explain import BAD, GOOD, NEUTRAL, explain, explanation_dict
from ai.mcts import MCTSAgent
from ai.minimax import MinimaxAgent
from config import MCTSConfig, MinimaxConfig
from game.actions import Action
from game.simulation import simulate_turn
from tests.conftest import make_state

FAST_MM = MinimaxConfig(max_depth=2, root_width=10, inner_width=5, node_budget=10**6, time_budget=None)
FAST_MC = MCTSConfig(iterations=30, rollout_depth=1, candidate_width=8, time_budget=None)


def test_knockout_and_ranking_are_reported_for_minimax(classic):
    # enemy marble right on the rim in front of our marble: the best shot knocks it out
    s = make_state([(0, 0.0, 6.0), (0, -3.0, -3.0), (1, 0.0, 8.9), (1, 4.0, -2.0)])
    d = MinimaxAgent(0, FAST_MM).decide(s)
    res = simulate_turn(s, d.action, classic)
    ex = explain(s, d, classic)
    assert ex.headline.startswith("P1 Minimax chose:")
    knocked = [m for m in res.eliminated if s.marbles[m].owner == 1]
    if knocked:
        assert any(r.kind == GOOD and "Knocks out enemy marble" in r.text for r in ex.reasons)
    assert ex.ranking and sum(r.selected for r in ex.ranking) == 1
    assert ex.ranking[0].rank == 1
    assert "shots" in ex.search and "turn(s) ahead" in ex.search
    assert 1 <= len(ex.reasons) <= 6


def test_mcts_explanation_uses_visits(classic):
    s = make_state([(0, 0.0, -3.0), (0, 4.0, -4.0), (1, 2.0, 7.0), (1, -3.0, 2.0)])
    d = MCTSAgent(0, FAST_MC, seed=3).decide(s)
    ex = explain(s, d, classic)
    assert ex.algorithm == "MCTS" and "simulated futures" in ex.search
    assert all("visits" in r.score for r in ex.ranking)
    assert any("fuzzy risk" in r.text for r in ex.reasons)


def test_edge_reasons_measure_distances(classic):
    s = make_state([(0, 0.0, 5.0), (0, -5.0, -5.0), (1, 0.0, 6.5), (1, 5.0, -5.0)])
    a = Action(0, math.pi / 2, 0.45)                       # gently push the enemy marble outward
    d = Decision(a, "Minimax + Alpha-Beta", 0, {"fuzzy_risk": 0.2}, [], "", False)
    ex = explain(s, d, classic)
    pushed = [r for r in ex.reasons if "toward the edge" in r.text]
    after = simulate_turn(s, a, classic).state
    if after.marbles[2].alive:
        assert pushed and pushed[0].kind == GOOD and "->" in pushed[0].text


def test_fallback_and_quiet_shots():
    s = make_state([(0, 0.0, 0.0), (1, 0.0, 5.0)])
    fb = Decision(Action(0, math.pi, 0.15), "MCTS (UCT)", 0, {}, [], "", True)
    ex = explain(s, fb)
    assert ex.reasons[0].kind == BAD and "fallback" in ex.reasons[0].text
    quiet = Decision(Action(0, math.pi, 0.15), "MCTS (UCT)", 0, {}, [], "", False)
    ex2 = explain(s, quiet)
    assert any(r.kind == NEUTRAL for r in ex2.reasons)
    d = explanation_dict(ex2)
    assert d["headline"] and isinstance(d["reasons"], list)
