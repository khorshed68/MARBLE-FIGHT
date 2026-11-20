"""Fuzzy-logic risk evaluator and opponent-history model tests."""

import itertools
import math

import pytest

from ai.evaluation import evaluate
from ai.fuzzy_risk import (EDGE, POWER, RISK, RULES, STOP, FuzzyInputs, FuzzyRiskEvaluator, trapmf, trimf)
from ai.opponent_model import EvalBias, OpponentModel
from config import DEFAULT_CONFIG
from game.actions import Action
from game.simulation import simulate_turn
from tests.conftest import make_state

F = FuzzyRiskEvaluator()


def base(**kw):
    d = dict(edge=5.0, power=0.6, align=0.0, stop=5.0, threat=0.1, friction=1.0)
    d.update(kw)
    return FuzzyInputs(**d)


# ----------------------------------------------------------------------------- membership functions
def test_trapezoid_and_triangle():
    assert trapmf(0.5, 0, 1, 2, 3) == pytest.approx(0.5)
    assert trapmf(1.5, 0, 1, 2, 3) == 1.0
    assert trapmf(2.5, 0, 1, 2, 3) == pytest.approx(0.5)
    assert trapmf(-1, 0, 1, 2, 3) == 0.0 and trapmf(4, 0, 1, 2, 3) == 0.0
    assert trapmf(0.0, 0, 0, 1, 2) == 1.0            # left shoulder
    assert trimf(0.5, 0, 0.5, 1) == 1.0
    assert trimf(0.25, 0, 0.5, 1) == pytest.approx(0.5)


def test_sets_overlap_and_cover_their_universe():
    for sets, lo, hi in ((EDGE, 0, 10), (POWER, 0, 1.4), (STOP, 0, 10)):
        for i in range(101):
            x = lo + (hi - lo) * i / 100
            degrees = [s.mu(x) for s in sets.values()]
            assert all(0.0 <= d <= 1.0 for d in degrees)
            assert max(degrees) > 0.0                 # every value belongs somewhere
    # neighbouring sets overlap (soft boundaries, not crisp classes)
    assert EDGE["near"].mu(3.0) > 0 and EDGE["medium"].mu(3.0) > 0
    assert POWER["medium"].mu(0.8) > 0 and POWER["high"].mu(0.8) > 0


# ----------------------------------------------------------------------------- pipeline stages
def test_fuzzification_values():
    fz = F.fuzzify(base(edge=1.0, power=1.0))
    assert fz["edge"]["near"] == 1.0 and fz["edge"]["far"] == 0.0
    assert fz["power"]["high"] == 1.0


def test_rule_activation_uses_min_for_and():
    fz = F.fuzzify(base(edge=2.5, power=0.875))
    fired = dict((r.text, w) for r, w in F.evaluate_rules(fz))
    rule = "IF edge is near AND power is high THEN risk is very_high"
    assert fired[rule] == pytest.approx(min(EDGE["near"].mu(2.5), POWER["high"].mu(0.875)))


def test_or_and_compound_rule():
    fz = F.fuzzify(base(align=-1.0, power=0.2))
    fired = dict((r.text, w) for r, w in F.evaluate_rules(fz))
    assert fired["IF align is inward AND (power is low OR power is medium) AND stop is NOT unsafe THEN risk is low"] == pytest.approx(1.0)


def test_aggregation_is_max_of_clipped_sets():
    low_rule = next(r for r in RULES if r.consequent == "low")
    high_rule = next(r for r in RULES if r.consequent == "very_high")
    agg = F.aggregate([(low_rule, 0.3), (high_rule, 0.6)])
    for u, m in zip(F.universe, agg):
        assert m == pytest.approx(max(min(RISK["low"].mu(u), 0.3), min(RISK["very_high"].mu(u), 0.6)))


def test_centroid_defuzzification():
    sym = [trimf(u, 0.2, 0.5, 0.8) for u in F.universe]
    assert F.defuzzify(sym) == pytest.approx(0.5, abs=1e-9)
    assert F.defuzzify([0.0] * len(F.universe)) == 0.5


def test_output_range_and_some_rule_always_fires():
    grid = itertools.product([0, 1.5, 3, 6, 10], [0.2, 0.6, 1.0, 1.35], [-1, 0, 1], [0, 1.5, 3, 8], [0, 0.5, 1],
                             [0.4, 1.0, 2.2])
    for e, p, a, s, t, f in grid:
        inp = FuzzyInputs(e, p, a, s, t, f)
        fired = F.evaluate_rules(F.fuzzify(inp))
        assert max(w for _, w in fired) > 0
        r = F.evaluate(inp).risk
        assert 0.0 <= r <= 1.0


def test_risk_rises_with_power_near_edge():
    risks = [F.evaluate(base(edge=1.0, align=0.9, stop=2.0, power=p)).risk for p in (0.2, 0.5, 0.8, 1.0, 1.3)]
    assert all(b >= a - 1e-9 for a, b in zip(risks, risks[1:]))
    assert risks[-1] > risks[0] + 0.05


def test_risk_falls_as_stop_point_gets_safer():
    risks = [F.evaluate(base(stop=s)).risk for s in (0.0, 1.0, 2.0, 3.0, 5.0, 8.0)]
    assert all(b <= a + 1e-9 for a, b in zip(risks, risks[1:]))
    assert risks[0] > 0.9 and risks[-1] < 0.35


def test_threat_increases_risk():
    assert F.evaluate(base(stop=3.0, threat=0.9)).risk > F.evaluate(base(stop=3.0, threat=0.0)).risk


def test_explain_lists_active_rules():
    res = F.evaluate(base(edge=1.0, power=1.0), explain=True)
    assert res.activations and all(w > 0 for _, w in res.activations)
    assert "edge" in res.memberships


def test_shot_risk_from_real_simulation(classic):
    s = make_state([(0, 0.0, -8.5), (1, 0.0, 4.0)])
    out = Action(0, -math.pi / 2, 1.0)       # straight off the board
    safe = Action(0, math.pi / 2, 0.35)      # gentle move to the centre
    r_out = F.shot_risk(s, out, simulate_turn(s, out, classic).state, classic, DEFAULT_CONFIG)
    r_safe = F.shot_risk(s, safe, simulate_turn(s, safe, classic).state, classic, DEFAULT_CONFIG)
    assert r_out > 0.8 > r_safe


# ----------------------------------------------------------------------------- opponent model
def test_empty_history_is_neutral():
    m = OpponentModel(0)
    assert m.bias() == EvalBias.neutral()
    assert m.aggression == 0.5


def test_aggressive_history_raises_defense_and_protects_target(classic):
    s = make_state([(0, 0.0, -6.0), (0, 4.0, -4.0), (1, 0.0, 4.0)], to_move=1)
    m = OpponentModel(0)
    for _ in range(4):
        a = Action(2, math.atan2(-6.0 - 4.0, 0.0), 1.0)          # aims at marble 0, full power
        m.observe(s, a.marble_id, a.angle, a.power, simulate_turn(s, a, classic).state)
    b = m.bias()
    assert m.aggressive == 4 and b.defense_mult > 1.0
    assert b.protect.get(0, 0) > 0
    assert m.profile()["most_targeted_marble"] == 0


def test_defensive_history_lowers_defense_weight(classic):
    s = make_state([(0, 6.0, -3.0), (1, 0.0, 7.0)], to_move=1)
    m = OpponentModel(0)
    for _ in range(5):
        m.observe(s, 1, -math.pi / 2, 0.3, s)      # retreats toward the centre
    assert m.defensive == 5 and m.bias().defense_mult < 1.0


def test_model_only_observes_opponent(classic):
    s = make_state([(0, 0.0, -6.0), (1, 0.0, 7.0)], to_move=0)
    with pytest.raises(ValueError):
        OpponentModel(0).observe(s, 0, 0.0, 0.5, s)


def test_snapshot_roundtrip_and_reset():
    m = OpponentModel(1)
    m.shots, m.aggressive, m.targeted = 3, 2, {4: 2}
    m2 = OpponentModel.from_snapshot(m.snapshot())
    assert m2.snapshot() == m.snapshot()
    m.reset()
    assert m.shots == 0 and m.observer == 1


def test_bias_changes_evaluation():
    s = make_state([(0, 0.0, -9.0), (1, 0.0, 3.0)])
    neutral = evaluate(s, 0)
    cautious = evaluate(s, 0, bias=EvalBias(defense_mult=1.6, threat_mult=1.0, protect={0: 0.6}))
    assert cautious < neutral
