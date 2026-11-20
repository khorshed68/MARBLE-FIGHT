"""Plain-language decision explanations must come from measured numbers."""

import math

import pytest

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


# ----------------------------------------------------------------------------- heuristic breakdown
def _start():
    from game.match import Match, PlayerSpec
    m = Match("classic", PlayerSpec("minimax"), PlayerSpec("mcts"), seed=11)
    return m.state, m.arena, m.cfg


def test_evaluation_terms_add_up_to_the_evaluation(classic):
    from ai.evaluation import evaluate, evaluation_terms
    from ai.opponent_model import EvalBias
    states = [make_state([(0, 0.0, 6.0), (0, -3.0, -3.0), (1, 0.0, 8.9), (1, 4.0, -2.0)]),
              make_state([(0, 9.0, 0.5), (1, -2.0, 1.0), (1, 0.5, -8.0)], to_move=1)]
    bias = EvalBias(1.3, 0.8, {0: 0.4})
    for s in states:
        for p in (0, 1):
            for b in (None, bias):
                rows = evaluation_terms(s, p, classic.radius, bias=b)
                assert sum(r[-1] for r in rows) == pytest.approx(evaluate(s, p, classic.radius, bias=b))
    won = make_state([(0, 0.0, 0.0), (1, 0.0, 30.0, False)])
    from dataclasses import replace
    won = replace(won, winner=0)
    rows = evaluation_terms(won, 0)
    assert rows[0][0] == "terminal" and rows[0][-1] == pytest.approx(evaluate(won, 0))


def test_heuristic_report_matches_the_agents_numbers():
    from ai.explain import heuristic_report
    from ai.search_trace import trace_decision
    st, arena, cfg = _start()
    agent = MinimaxAgent(0, MinimaxConfig(max_depth=1, root_width=24, inner_width=10, node_budget=10**6,
                                          time_budget=None))
    dec = agent.decide(st)
    rep = heuristic_report(st, dec.action, 0, arena, cfg, dec.stats)
    assert rep.one_ply == pytest.approx(dec.stats["evaluation"], abs=0.01)      # depth-1 value of the agent
    assert rep.one_ply == pytest.approx(rep.static_value + rep.edge)
    assert rep.edge == pytest.approx(-rep.fuzzy_weight * rep.risk)
    assert rep.risk == pytest.approx(dec.stats["fuzzy_risk"], abs=1e-3)
    assert rep.reward == pytest.approx(1 / (1 + math.exp(-rep.one_ply / rep.reward_scale)))
    assert rep.fired and all(0 < w <= 1 for _, w in rep.fired)
    assert {n for n, _, _ in rep.inputs} == {"edge", "power", "align", "stop", "threat", "friction"}
    assert [w for _, _, w, _ in rep.rule_calcs] == pytest.approx([w for _, w in rep.fired])
    num, den = rep.centroid
    assert num / den == pytest.approx(rep.risk)                                  # the centroid shown is exact
    # the principal-variation breakdown reproduces the tree value exactly
    from ai.evaluation import evaluation_terms
    tr = trace_decision(st, player=0, depth=2, root_width=4, inner_width=3, game_cfg=cfg)
    pv = tr.principal_variation()
    leaf = tr.nodes[pv[-1]]
    total = sum(r[-1] for r in evaluation_terms(leaf.state, 0, arena.radius)) + sum(tr.nodes[i].edge for i in pv[1:])
    assert total == pytest.approx(tr.value)


def test_bias_is_rebuilt_from_the_decision_stats():
    from ai.explain import bias_from_stats
    b = bias_from_stats({"opponent_profile": {"defense_mult": 1.2, "threat_mult": 0.9, "protect": {"2": 0.3}}})
    assert (b.defense_mult, b.threat_mult, b.protect) == (1.2, 0.9, {2: 0.3})
    n = bias_from_stats({})
    assert (n.defense_mult, n.threat_mult, n.protect) == (1.0, 1.0, {})
