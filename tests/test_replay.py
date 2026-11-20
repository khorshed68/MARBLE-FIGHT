"""Replay recording, deterministic re-simulation and error handling."""

import copy
import math
from dataclasses import replace

import pytest

from game.actions import Action
from game.match import Match, PlayerSpec
from game.replay import (REPLAY_VERSION, ReplayError, ReplayPlayer, config_fingerprint, list_replays, load_replay,
                         record_from_match, save_replay, verify_replay)

FAST = {"minimax": {"max_depth": 2, "root_width": 10, "inner_width": 5},
        "mcts": {"iterations": 25, "rollout_depth": 1, "candidate_width": 16}}


def ai(kind):
    return PlayerSpec(kind, "easy", overrides=FAST[kind])


def finished_match(arena="neon", seed=4, p1="minimax", p2="mcts", turn_limit=20):
    m = Match(arena, ai(p1), ai(p2), seed=seed, turn_limit=turn_limit)
    m.run_to_end()
    return m


@pytest.mark.parametrize("arena", ["classic", "neon", "magnetic"])
def test_replay_reproduces_the_recorded_match(arena):
    m = finished_match(arena)
    rec = record_from_match(m)
    v = verify_replay(rec)
    assert v.ok, v.message
    assert v.turns_checked == len(m.history) == m.state.turn
    assert v.final_state == m.state
    assert v.final_state.winner == m.state.winner


def test_replay_player_steps_match_live_history():
    m = finished_match("magnetic", seed=8)
    rec = record_from_match(m)
    p = ReplayPlayer(rec)
    for h in m.history:
        res = p.step(record_frames=True)
        assert res.frames                                     # playable frames are regenerated, not stored
        assert res.state.fingerprint() == h.fingerprint
        assert tuple(res.eliminated) == tuple(h.eliminated)
    assert p.done
    with pytest.raises(ReplayError):
        p.step()


def test_human_vs_human_replay():
    m = Match("classic", PlayerSpec("human"), PlayerSpec("human"), seed=1, turn_limit=4)
    for k in range(4):
        mid = m.state.alive(m.state.to_move)[0].mid
        mm = m.state.marbles[mid]
        m.apply(Action(mid, math.atan2(-mm.y, -mm.x) + 0.3 * k, 0.4 + 0.1 * k, "spin_left" if k == 1 else "none"))
    assert m.is_over
    assert verify_replay(record_from_match(m, "pvp")).ok


def test_same_seed_and_config_reproduce_the_match():
    a = finished_match("neon", seed=12)
    b = finished_match("neon", seed=12)
    assert [h.action for h in a.history] == [h.action for h in b.history]
    assert a.state == b.state


def test_tampered_action_diverges_at_that_turn():
    rec = record_from_match(finished_match("magnetic", seed=3))
    assert len(rec["turns"]) >= 3
    bad = copy.deepcopy(rec)
    bad["turns"][2]["action"]["power"] = max(0.15, bad["turns"][2]["action"]["power"] - 0.2)
    v = verify_replay(bad)
    assert not v.ok and "Turn 3" in v.message and v.turns_checked == 2


def test_missing_and_illegal_actions_are_reported_not_invented():
    rec = record_from_match(finished_match())
    miss = copy.deepcopy(rec)
    del miss["turns"][1]["action"]
    v = verify_replay(miss)
    assert not v.ok and "missing" in v.message
    illegal = copy.deepcopy(rec)
    illegal["turns"][0]["action"]["marble_id"] = [m for m in range(6) if m >= 3][0]   # an opponent marble
    v = verify_replay(illegal)
    assert not v.ok and ("illegal" in v.message or "recorded player" in v.message)
    short = copy.deepcopy(rec)
    short["turns"] = short["turns"][:-1]
    assert not verify_replay(short).ok                       # result no longer matches


def test_config_change_is_detected(monkeypatch):
    rec = record_from_match(finished_match("classic", seed=5))
    from game import arena as A
    arenas = A._all_arenas()
    original = arenas["classic"]
    monkeypatch.setitem(arenas, "classic", replace(original, base_friction=original.base_friction * 0.5))
    p = ReplayPlayer(rec)
    assert not p.config_matches
    v = verify_replay(rec)
    assert not v.ok and "configuration changed" in v.message


def test_save_load_and_validation(tmp_path):
    rec = record_from_match(finished_match())
    path = save_replay(rec, "unit", str(tmp_path))
    assert load_replay(path) == rec
    assert list_replays(str(tmp_path)) == [path]
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ReplayError):
        load_replay(str(bad))
    old = tmp_path / "old.json"
    import json
    old.write_text(json.dumps({**rec, "version": REPLAY_VERSION + 1}))
    with pytest.raises(ReplayError, match="version"):
        load_replay(str(old))


def test_config_fingerprint_is_stable():
    from config import DEFAULT_CONFIG
    assert config_fingerprint(DEFAULT_CONFIG, "neon") == config_fingerprint(DEFAULT_CONFIG, "neon")
    assert config_fingerprint(DEFAULT_CONFIG, "neon") != config_fingerprint(DEFAULT_CONFIG, "classic")
