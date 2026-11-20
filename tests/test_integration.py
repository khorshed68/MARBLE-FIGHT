"""End-to-end tests: matches, modes, tournament, playback, worker and fallback."""

import json
import math

import pytest

from ai.base_agent import Decision
from ai.worker import AIWorker, compute_decision
from analytics.match_logger import build_record, load_history, log_match
from analytics.metrics import aggregate, head_to_head
from analytics.tournament import TournamentConfig, render_report, run_tournament, schedule
from game.actions import Action, is_legal
from game.match import Match, PlayerSpec, decision_seed
from ui.playback import Playback

FAST = {"minimax": {"max_depth": 2, "root_width": 10, "inner_width": 5},
        "mcts": {"iterations": 25, "rollout_depth": 1, "candidate_width": 16}}


def ai(kind, **kw):
    return PlayerSpec(kind, "easy", overrides={**FAST[kind], **kw})


def test_minimax_vs_mcts_match_completes():
    m = Match("classic", ai("minimax"), ai("mcts"), seed=5, turn_limit=30)
    m.run_to_end()
    s = m.summary()
    assert m.is_over and s["winner"] in (0, 1, -1)
    assert s["turns"] == len(m.history) <= 30
    assert s["players"][0]["minimax_nodes"] > 0
    assert s["players"][1]["mcts_iterations"] > 0
    assert s["players"][0]["decisions"] + s["players"][1]["decisions"] == s["turns"]
    removed = 6 - sum(s["score"])
    assert s["eliminated_total"] == removed


@pytest.mark.parametrize("p1,p2,arena", [("mcts", "minimax", "neon"), ("minimax", "minimax", "magnetic"),
                                         ("mcts", "mcts", "classic")])
def test_all_ai_pairings_progress(p1, p2, arena):
    m = Match(arena, ai(p1), ai(p2), seed=1, turn_limit=8)
    m.run_to_end()
    assert m.is_over
    for h in m.history:
        assert h.decision is not None and not h.decision["fallback"]


def test_seeded_matches_are_reproducible():
    def play():
        m = Match("magnetic", ai("mcts"), ai("minimax"), seed=9, turn_limit=10)
        m.run_to_end()
        return [h.action for h in m.history], m.state
    a1, s1 = play()
    a2, s2 = play()
    assert a1 == a2 and s1 == s2


def test_human_vs_ai_progression():
    m = Match("classic", PlayerSpec("human"), ai("minimax"), seed=2)
    assert not m.is_ai_turn()
    assert m.play_ai_turn() is None                       # AI may not move on the human's turn
    mid = m.state.alive(0)[0].mid
    res = m.apply(Action(mid, math.pi / 2, 0.4))
    assert res.frames and len(res.frames) > 1             # recorded for rendering
    assert m.is_ai_turn()
    r2 = m.play_ai_turn()
    assert r2 is not None and m.state.turn == 2 and m.state.to_move == 0
    # the AI observed the human's shot through its opponent model
    assert m.agent(1).opponent_model.shots == 1


def test_human_illegal_action_is_rejected():
    m = Match("classic", PlayerSpec("human"), ai("minimax"))
    enemy = m.state.alive(1)[0].mid
    with pytest.raises(ValueError):
        m.apply(Action(enemy, 0.0, 0.5))
    assert m.state.turn == 0


def test_pause_resume_and_restart():
    m = Match("classic", ai("minimax"), ai("mcts"), seed=3)
    m.paused = True
    assert m.play_ai_turn() is None and m.state.turn == 0
    m.paused = False
    assert m.play_ai_turn() is not None and m.state.turn == 1
    m.restart()
    assert m.state.turn == 0 and not m.history and m.state.count(0) == 3
    assert m.agent(1).opponent_model.shots == 0           # history reset between matches


def test_fallback_decision_is_legal():
    m = Match("neon", ai("minimax"), ai("mcts"))
    d = m.fallback_decision("timeout")
    assert d.fallback and is_legal(m.state, d.action, m.cfg.rules)
    m.apply(d.action, d)
    assert m.fallbacks == 1


def test_decision_seeds_differ_per_turn():
    assert decision_seed(1, 0, 0) != decision_seed(1, 1, 0) != decision_seed(1, 1, 1)


def test_playback_matches_authoritative_result():
    m = Match("neon", ai("minimax"), ai("mcts"), seed=4)
    d = m.compute_decision()
    res = m.apply(d.action, d, record=True)
    pb = Playback(res.frames, res.contacts, list(zip(res.eliminated, res.elimination_times)), speed=2.0)
    events = []
    while not pb.done:
        events += pb.advance(1 / 60)
        pos, alive, _ = pb.sample()
    pos, alive, _ = pb.sample()
    for i, mm in enumerate(m.state.marbles):
        assert alive[i] == mm.alive
        if mm.alive:
            assert pos[i] == pytest.approx((mm.x, mm.y))
    elim_events = [e.a for e in events if e.kind == "eliminated"]
    assert elim_events == list(res.eliminated)


def test_worker_process_and_thread_return_legal_decisions():
    m = Match("classic", ai("mcts"), ai("minimax"), seed=6)
    spec = m.specs[0].agent_spec()
    direct = Decision.from_dict(compute_decision({"state": m.state.to_dict(), "spec": spec, "seed": 7,
                                                  "opponent_model": None, "time_budget": 10.0}))
    assert is_legal(m.state, direct.action, m.cfg.rules)
    for use_proc in (False, True):
        w = AIWorker(use_processes=use_proc)
        try:
            rid = w.request(m.state, spec, 7, None, 10.0)
            import time
            t0 = time.time()
            while True:
                status, dec, err = w.poll(rid)
                if status != "pending":
                    break
                assert time.time() - t0 < 60
                time.sleep(0.02)
            assert status == "done", err
            assert dec.action == direct.action        # same seed -> same decision in any executor
        finally:
            w.shutdown()


def test_match_logging_and_metrics(tmp_path):
    path = tmp_path / "hist.jsonl"
    m = Match("classic", ai("minimax"), ai("mcts"), seed=8, turn_limit=6)
    m.run_to_end()
    log_match(m, "arena", path=str(path))
    with open(path, "a") as fh:
        fh.write("{corrupt line\n")
    recs = load_history(str(path))
    assert len(recs) == 1
    agg = aggregate(recs)
    assert agg["total_matches"] == 1
    mm, mc = agg["algorithms"]["minimax"], agg["algorithms"]["mcts"]
    assert mm["matches"] == mc["matches"] == 1
    assert mm["wins"] + mm["losses"] + mm["draws"] == 1
    assert mm["total_nodes"] == m.summary()["players"][0]["minimax_nodes"]
    assert head_to_head(recs)["matches"] == 1
    assert aggregate([]) == {"total_matches": 0, "algorithms": {}, "arenas": {}, "pairings": []}


def test_tournament_schedule_rotates_seats_and_arenas():
    jobs = schedule(TournamentConfig(matches=4, arenas=["classic", "neon"]))
    assert [(j["p1"], j["p2"]) for j in jobs] == [("minimax", "mcts"), ("mcts", "minimax")] * 2
    assert [j["arena"] for j in jobs] == ["classic", "neon", "classic", "neon"]
    assert len({j["seed"] for j in jobs}) == 4


def test_tournament_runs_real_matches(tmp_path, monkeypatch):
    import analytics.tournament as T
    monkeypatch.setattr(T, "TOURNAMENT_DIR", str(tmp_path / "t"))
    monkeypatch.setattr(T, "REPORTS_DIR", str(tmp_path / "r"))
    cfg = TournamentConfig(matches=2, arenas=["classic"], difficulty="easy", turn_limit=8, workers=1,
                           overrides=FAST, log_history=False, save_replays=False)
    res = run_tournament(cfg)
    assert res["completed"] == 2
    h = res["head_to_head"]
    assert h["minimax"] + h["mcts"] + h["draw"] == 2
    data = json.load(open(res["files"]["json"]))
    assert data["completed"] == 2
    report = render_report(res)
    assert "Measured results" in report and "Head to head" in report
