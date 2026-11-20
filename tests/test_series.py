"""Two-round series: first player alternates, overall result and tie-breaks."""

import pytest

from game.match import Match, PlayerSpec
from game.series import Series
from game.state import DRAW


def rnd(first, winner, score, turns=10):
    return {"first_player": first, "winner": winner, "score": score, "turns": turns, "end_reason": "test"}


def test_first_player_alternates_each_round():
    s = Series(2, first_player=0)
    assert s.next_first_player() == 0 and s.round_number == 1
    s.record(rnd(0, 0, [2, 0]))
    assert s.next_first_player() == 1 and s.round_number == 2
    s2 = Series(2, first_player=1)
    assert [s2.first_player_of(i) for i in range(4)] == [1, 0, 1, 0]


def test_more_round_wins_decides():
    s = Series(2)
    s.record(rnd(0, 1, [0, 1]))
    assert s.winner() is None and not s.is_over
    s.record(rnd(1, 1, [0, 3]))
    assert s.is_over and s.winner() == 1 and s.round_wins() == (0, 2)
    assert "more rounds" in s.reason()


def test_split_rounds_are_decided_by_total_marbles():
    s = Series(2)
    s.record(rnd(0, 0, [3, 0]))           # P1 wins round 1 keeping 3 marbles
    s.record(rnd(1, 1, [0, 1]))           # P2 wins round 2 keeping 1
    assert s.round_wins() == (1, 1)
    assert s.total_marbles() == (3, 1) and s.winner() == 0
    assert "marbles remaining" in s.reason()


def test_draws_count_as_no_round_win_and_full_tie_is_a_draw():
    s = Series(2)
    s.record(rnd(0, DRAW, [1, 1]))
    s.record(rnd(1, DRAW, [2, 2]))
    assert s.round_wins() == (0, 0) and s.winner() == DRAW
    t = Series(2)
    t.record(rnd(0, 0, [2, 0]))
    t.record(rnd(1, 1, [0, 2]))
    assert t.winner() == DRAW


def test_invalid_use_is_rejected():
    with pytest.raises(ValueError):
        Series(0)
    s = Series(2)
    with pytest.raises(ValueError):
        s.record(rnd(1, 0, [1, 0]))       # round 1 must start with player 0
    with pytest.raises(ValueError):
        s.record({"first_player": 0, "winner": None, "score": [1, 1], "turns": 3})   # unfinished
    s.record(rnd(0, 0, [1, 0]))
    s.record(rnd(1, 0, [1, 0]))
    with pytest.raises(ValueError):
        s.record(rnd(0, 0, [1, 0]))       # already over
    s.reset()
    assert s.rounds == [] and s.round_number == 1


def test_headless_two_round_series_with_real_matches():
    fast = {"minimax": {"max_depth": 1, "root_width": 8, "inner_width": 4},
            "mcts": {"iterations": 12, "rollout_depth": 1, "candidate_width": 10}}
    p1 = PlayerSpec("minimax", "easy", overrides=fast["minimax"])
    p2 = PlayerSpec("mcts", "easy", overrides=fast["mcts"])
    s = Series(2)
    for i in range(2):
        m = Match("classic", p1, p2, marbles=2, seed=5 + i, first_player=s.next_first_player(), turn_limit=12)
        assert m.state.to_move == s.next_first_player()           # the swap really happens in the game state
        m.run_to_end()
        s.record(m.summary())
    assert s.is_over and s.winner() in (0, 1, DRAW)
    d = s.to_dict()
    assert [r["first_player"] for r in d["rounds"]] == [0, 1]
    assert d["round_wins"] == list(s.round_wins())
