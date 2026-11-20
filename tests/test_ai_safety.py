"""AI computation safety: budgets, cancellation, fallback, transition cache, worker responsiveness."""

import math
import random
import threading
import time
from dataclasses import replace

import pytest

from ai.base_agent import Clock
from ai.mcts import MCTSAgent
from ai.minimax import MinimaxAgent
from ai.worker import AIWorker
from config import DEFAULT_CONFIG, MCTSConfig, MinimaxConfig
from game.actions import Action, generate_candidate_actions, is_legal
from game.arena import get_arena
from game.rules import initial_state
from game.simulation import simulate_turn
from tests.conftest import make_state

HUGE_MM = MinimaxConfig(max_depth=4, root_width=60, inner_width=30, node_budget=10**9, time_budget=120.0)
HUGE_MC = MCTSConfig(iterations=10**9, time_budget=120.0)


def start(arena="classic"):
    return initial_state(get_arena(arena))


# ----------------------------------------------------------------------------- clock / cancellation
def test_clock_cancel_and_budget():
    flag = {"v": False}
    c = Clock(None, lambda: flag["v"])
    assert not c.expired()
    flag["v"] = True
    assert c.expired() and c.cancelled
    assert Clock(0.0).expired() and not Clock(0.0).cancelled


@pytest.mark.parametrize("agent", [MinimaxAgent(0, HUGE_MM), MCTSAgent(0, HUGE_MC, seed=1)])
def test_immediate_cancel_returns_legal_action_fast(agent):
    s = start()
    t0 = time.perf_counter()
    d = agent.decide(s, cancel=lambda: True)
    assert time.perf_counter() - t0 < 1.0
    assert d.stats["cancelled"] is True
    assert is_legal(s, d.action, DEFAULT_CONFIG.rules)


@pytest.mark.parametrize("make", [lambda: MinimaxAgent(0, HUGE_MM), lambda: MCTSAgent(0, HUGE_MC, seed=2)])
def test_cancel_during_search_from_another_thread(make):
    s = start("neon")
    ev = threading.Event()
    threading.Timer(0.3, ev.set).start()
    t0 = time.perf_counter()
    d = make().decide(s, cancel=ev.is_set)
    took = time.perf_counter() - t0
    assert 0.25 < took < 1.5                       # stopped right after the cancel request
    assert d.stats["cancelled"] and is_legal(s, d.action, DEFAULT_CONFIG.rules)


def test_mcts_keeps_partial_result_after_cancel():
    s = start()
    calls = {"n": 0}

    def cancel_after_some():
        calls["n"] += 1
        return calls["n"] > 30
    d = MCTSAgent(0, HUGE_MC, seed=3).decide(s, cancel=cancel_after_some)
    assert d.stats["cancelled"] and 0 < d.stats["iterations"] < 31
    assert not d.fallback                           # best action from the search completed so far


def test_time_budget_exhaustion_is_reported():
    s = start("magnetic")
    d = MCTSAgent(0, HUGE_MC, seed=1).decide(s, time_budget=0.2)
    assert d.stats["budget_exhausted"] and not d.stats["cancelled"]
    assert is_legal(s, d.action, DEFAULT_CONFIG.rules)
    d2 = MinimaxAgent(0, HUGE_MM).decide(s, time_budget=0.2)
    assert d2.stats["budget_exhausted"] and is_legal(s, d2.action, DEFAULT_CONFIG.rules)


def test_fallback_action_is_always_legal():
    rng = random.Random(4)
    agent = MinimaxAgent(0, HUGE_MM)
    for _ in range(40):
        ms = [(0, rng.uniform(-7, 7), rng.uniform(-7, 7)), (1, rng.uniform(-7, 7), rng.uniform(-7, 7)),
              (0, 30, 0, False)]
        s = make_state(ms, to_move=rng.randint(0, 1), abilities=((0, 0, 0), (0, 0, 0)))
        if s.alive(s.to_move):
            assert is_legal(s, agent.fallback_action(s), DEFAULT_CONFIG.rules)


# ----------------------------------------------------------------------------- transition cache
def test_transition_cache_matches_uncached_simulation():
    s = start("magnetic")
    agent = MinimaxAgent(0, HUGE_MM)
    agent._prepare(s)
    for a in generate_candidate_actions(s, DEFAULT_CONFIG.rules)[:15]:
        cached1, r1 = agent.transition(s, a)
        cached2, r2 = agent.transition(s, a)             # served from the cache
        fresh = simulate_turn(s, a, get_arena("magnetic")).state
        assert cached1 is cached2 and r1 == r2
        assert cached1 == fresh


def test_minimax_with_and_without_cache_agree():
    s = start("neon")
    cfg = MinimaxConfig(max_depth=2, root_width=12, inner_width=6, node_budget=10**9, time_budget=None)
    a = MinimaxAgent(0, cfg)
    b = MinimaxAgent(0, cfg)
    b.use_cache = False
    da, db = a.decide(s), b.decide(s)
    assert da.action == db.action
    assert da.stats["evaluation"] == db.stats["evaluation"]
    assert da.stats["nodes_visited"] == db.stats["nodes_visited"]
    assert da.stats["simulations"] < db.stats["simulations"]      # the cache really saves work


def test_mcts_with_and_without_cache_agree():
    s = start()
    cfg = MCTSConfig(iterations=40, rollout_depth=1, candidate_width=16, time_budget=None)
    a, b = MCTSAgent(0, cfg, seed=9), MCTSAgent(0, cfg, seed=9)
    b.use_cache = False
    da, db = a.decide(s), b.decide(s)
    assert da.action == db.action and da.stats["selected_visits"] == db.stats["selected_visits"]


# ----------------------------------------------------------------------------- worker
def _wait(w, rid, limit=60.0):
    t0 = time.time()
    while True:
        st, d, e = w.poll(rid)
        if st != "pending":
            return st, d, e
        assert time.time() - t0 < limit
        time.sleep(0.01)


@pytest.mark.parametrize("use_proc", [False, True])
def test_worker_cancel_then_next_request(use_proc):
    w = AIWorker(use_processes=use_proc)
    try:
        s = start()
        spec = {"kind": "mcts", "difficulty": "hard", "overrides": {"iterations": 10**9, "time_budget": 120.0}}
        rid = w.request(s, spec, 1, None, 120.0)
        time.sleep(0.8)
        w.cancel_all()
        t0 = time.time()
        assert _wait(w, rid, 10)[0] == "cancelled"
        assert time.time() - t0 < 3.0
        rid2 = w.request(s, {"kind": "minimax", "difficulty": "easy"}, 1, None, 10.0)
        st, d, e = _wait(w, rid2)
        assert st == "done", e
        assert is_legal(s, d.action, DEFAULT_CONFIG.rules)
    finally:
        w.shutdown()


def test_worker_reports_errors_instead_of_crashing():
    w = AIWorker(use_processes=False)
    try:
        rid = w.request(start(), {"kind": "nonexistent-agent"}, 1, None, 1.0)
        st, d, e = _wait(w, rid)
        assert st == "error" and d is None and "nonexistent" in e
    finally:
        w.shutdown()


def test_worker_hard_restart_on_stuck_job():
    w = AIWorker(use_processes=True, stuck_after=0.0)
    try:
        s = start()
        spec = {"kind": "mcts", "overrides": {"iterations": 10**9, "time_budget": 120.0}}
        rid = w.request(s, spec, 1, None, 120.0)
        time.sleep(0.5)
        w.cancel_all()
        w.poll(rid)                                   # stuck_after=0 -> process is restarted
        assert w.restarts == 1 and w.proc.is_alive()
        rid2 = w.request(s, {"kind": "minimax", "difficulty": "easy"}, 1, None, 10.0)
        assert _wait(w, rid2)[0] == "done"
    finally:
        w.shutdown()


def test_main_loop_stays_responsive_during_search():
    """A 'frame loop' on the main thread keeps ticking while the worker process searches."""
    w = AIWorker(use_processes=True)
    try:
        s = start("magnetic")
        warm = w.request(s, {"kind": "minimax", "difficulty": "easy"}, 1, None, 5.0)
        _wait(w, warm)
        rid = w.request(s, {"kind": "mcts", "difficulty": "hard", "overrides": {"time_budget": 2.0}}, 1, None, 2.0)
        last, worst, frames = time.perf_counter(), 0.0, 0
        while w.poll(rid)[0] == "pending":
            # simulated frame: a little main-thread work
            sum(math.sqrt(i) for i in range(2000))
            now = time.perf_counter()
            worst = max(worst, now - last)
            last = now
            frames += 1
            time.sleep(0.005)
        assert frames > 50
        assert worst < 0.2, f"main loop stalled for {worst:.3f}s"
    finally:
        w.shutdown()
