"""Tests for setup, turn order, abilities and terminal conditions."""

import math
from dataclasses import replace

import pytest

from config import DEFAULT_CONFIG
from game.actions import Action, IllegalActionError, generate_candidate_actions, is_legal, validate_action
from game.arena import Arena, arena_ids, get_arena
from game.rules import SetupError, initial_state, resolve_terminal, turn_limit_score
from game.simulation import simulate_turn
from game.state import DRAW, GameState
from tests.conftest import make_state


@pytest.mark.parametrize("arena_id", ["classic", "neon", "magnetic"])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_initial_placement_is_legal_and_symmetric(arena_id, n):
    arena = get_arena(arena_id)
    s = initial_state(arena, per_player=n)
    assert s.count(0) == n and s.count(1) == n
    r = DEFAULT_CONFIG.physics.marble_radius
    ms = s.marbles
    for i, a in enumerate(ms):
        assert math.hypot(a.x, a.y) + r < arena.radius
        for b in ms[i + 1:]:
            assert math.hypot(a.x - b.x, a.y - b.y) >= 2 * r
        for ox, oy, orr in arena.obstacle_positions(0.0):
            assert math.hypot(a.x - ox, a.y - oy) >= r + orr
    p1 = sorted((round(m.x, 9), round(m.y, 9)) for m in s.alive(0))
    p2 = sorted((round(-m.x, 9), round(-m.y, 9)) for m in s.alive(1))
    assert p1 == p2   # point symmetry -> fair start


def test_default_setup_three_marbles_player1_first():
    s = initial_state(get_arena("classic"))
    assert s.count(0) == 3 and s.to_move == 0 and s.turn == 0 and s.winner is None
    assert s.abilities == ((2, 2, 2), (2, 2, 2))


def test_invalid_marble_count():
    with pytest.raises(SetupError):
        initial_state(get_arena("classic"), per_player=0)


def test_all_arenas_load():
    assert {"classic", "neon", "magnetic"} <= set(arena_ids())


def test_only_active_player_may_shoot(classic):
    s = initial_state(classic)
    own = s.alive(0)[0].mid
    validate_action(s, Action(own, 0.0, 0.5), DEFAULT_CONFIG.rules, player=0)
    with pytest.raises(IllegalActionError, match="turn"):
        validate_action(s, Action(own, 0.0, 0.5), DEFAULT_CONFIG.rules, player=1)
    enemy = s.alive(1)[0].mid
    assert not is_legal(s, Action(enemy, 0.0, 0.5), DEFAULT_CONFIG.rules)


def test_players_alternate(classic):
    s = initial_state(classic)
    order = []
    for k in range(4):
        order.append(s.to_move)
        mid = s.alive(s.to_move)[0].mid
        s = simulate_turn(s, Action(mid, math.atan2(-s.marbles[mid].y, -s.marbles[mid].x), 0.2), classic).state
    assert order == [0, 1, 0, 1]
    assert s.turn == 4


def test_ability_uses_decrement_and_run_out(classic):
    s = make_state([(0, 0.0, -2.0), (1, 0.0, 6.0)], abilities=((1, 1, 1), (1, 1, 1)))
    s2 = simulate_turn(s, Action(0, math.pi, 0.2, "power"), classic).state
    assert s2.abilities[0] == (1, 0, 1)
    s3 = simulate_turn(s2, Action(1, 0.0, 0.2), classic).state
    assert not is_legal(s3, Action(0, 0.0, 0.2, "power"), DEFAULT_CONFIG.rules)
    assert is_legal(s3, Action(0, 0.0, 0.2, "spin_left"), DEFAULT_CONFIG.rules)


def test_candidate_actions_are_legal_and_respect_abilities(classic):
    s = initial_state(classic)
    acts = generate_candidate_actions(s, DEFAULT_CONFIG.rules)
    assert len(acts) > 50
    assert all(is_legal(s, a, DEFAULT_CONFIG.rules) for a in acts)
    assert {a.marble_id for a in acts} == {m.mid for m in s.alive(0)}
    empty = s.with_changes(abilities=((0, 0, 0), (2, 2, 2)))
    assert all(a.ability == "none" for a in generate_candidate_actions(empty, DEFAULT_CONFIG.rules))
    # deterministic
    assert generate_candidate_actions(s, DEFAULT_CONFIG.rules) == acts


def test_victory_detection(classic):
    s = make_state([(0, 5.0, 0.0), (1, 8.6, 0.0)])
    r = simulate_turn(s, Action(0, 0.0, 1.0), classic)
    assert r.state.winner == 0 and r.state.is_terminal
    assert "Player 2" in r.state.end_reason


def test_defeat_by_own_marble(classic):
    s = make_state([(0, 8.0, 0.0), (1, -3.0, 0.0)])
    r = simulate_turn(s, Action(0, 0.0, 1.0), classic)
    assert r.state.winner == 1


def test_simultaneous_elimination_is_draw():
    # explicit friction-1.0 arena so the geometry does not depend on arena tuning
    plain = Arena("plain", "Plain", "", 10.0, 1.0)
    s = make_state([(0, 6.0, 0.0), (1, 8.3, -0.7)])
    r = simulate_turn(s, Action(0, 0.0, 1.0), plain)
    assert set(r.eliminated) == {0, 1}
    assert r.state.winner == DRAW


def test_turn_limit_scoring_by_marbles_then_safety(classic):
    s = make_state([(0, 0.0, 0.0), (0, 1.0, 1.0), (1, 9.0, 0.0)], turn=60, turn_limit=60)
    assert resolve_terminal(s, classic)[0] == 0
    s = make_state([(0, 9.0, 0.0), (1, 2.0, 0.0)], turn=60, turn_limit=60)   # equal count, P2 safer
    s0, s1 = turn_limit_score(s, classic)
    assert s0[0] == s1[0] and s1[1] > s0[1]
    assert resolve_terminal(s, classic)[0] == 1


def test_turn_limit_exact_tie_is_draw(classic):
    s = make_state([(0, 3.0, 0.0), (1, -3.0, 0.0)], turn=60, turn_limit=60)
    assert resolve_terminal(s, classic) == (DRAW, "Turn limit reached with an exactly tied score")


def test_turn_limit_reached_through_play(classic):
    s = make_state([(0, 0.0, -3.0), (1, 0.0, 3.0)], turn_limit=2)
    s = simulate_turn(s, Action(0, math.pi, 0.2), classic).state
    assert not s.is_terminal
    s = simulate_turn(s, Action(1, 0.0, 0.2), classic).state
    assert s.is_terminal and s.turn == 2


def test_restart_gives_identical_state(classic):
    assert initial_state(classic) == initial_state(classic)


def test_state_roundtrip_serialisation(classic):
    s = initial_state(classic)
    assert GameState.from_dict(s.to_dict()) == s
