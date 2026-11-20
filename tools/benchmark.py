"""Benchmark both agents on the same saved game states.

    python -m tools.benchmark                  # normal budgets, writes reports/benchmark_<time>.md
    python -m tools.benchmark --difficulty hard

Steps:
1. Build (or load) ``data/benchmark_states.json``: positions reached by playing
   a few deterministic greedy turns from the start of every arena.
2. For every state, run Minimax and MCTS (same budget preset) and record the
   decision time, physics simulations and algorithm-specific counters.
3. Run plain Minimax and Alpha-Beta on the *same* search problem (depth 2,
   fixed widths) and record both values and node counts, demonstrating that
   pruning returns the identical value with fewer nodes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import replace

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.base_agent import Clock
from ai.factory import make_agent
from ai.minimax import MarbleSearchProblem, MinimaxAgent, SearchStats, alphabeta, minimax
from config import DATA_DIR, DEFAULT_CONFIG, MinimaxConfig, REPORTS_DIR
from game.actions import generate_candidate_actions
from game.arena import arena_ids, get_arena
from game.rules import initial_state
from game.simulation import simulate_turn
from game.state import GameState

STATES_FILE = os.path.join(DATA_DIR, "benchmark_states.json")


def build_states() -> list[dict]:
    """Deterministic positions: start + after 2 and 4 greedy turns, per arena."""
    out = []
    for aid in arena_ids():
        arena = get_arena(aid)
        s = initial_state(arena)
        out.append({"name": f"{aid}-start", "state": s.to_dict()})
        for turn in range(4):
            acts = generate_candidate_actions(s, DEFAULT_CONFIG.rules, arena.radius)
            # greedy-ish: highest priority shot that does not end the match
            for a in acts:
                r = simulate_turn(s, a, arena)
                if not r.state.is_terminal:
                    s = r.state
                    break
            if turn in (1, 3):
                out.append({"name": f"{aid}-turn{turn + 1}", "state": s.to_dict()})
    return out


def load_states(rebuild: bool = False) -> list[dict]:
    if not rebuild and os.path.exists(STATES_FILE):
        with open(STATES_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    states = build_states()
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATES_FILE, "w", encoding="utf-8") as fh:
        json.dump(states, fh, indent=1)
    return states


def pruning_comparison(state: GameState) -> dict:
    cfg = MinimaxConfig(max_depth=2, root_width=12, inner_width=8, node_budget=10**9, time_budget=None)
    agent = MinimaxAgent(state.to_move, cfg)
    agent._prepare(state)
    prob = MarbleSearchProblem(agent, state.to_move, Clock(None), cfg)
    prob.root_state = state
    s1, s2 = SearchStats(), SearchStats()
    t0 = time.perf_counter()
    v_plain = minimax(prob, state, 2, s1)
    t1 = time.perf_counter()
    agent._cache.clear()
    agent.simulations = 0
    v_ab = alphabeta(prob, state, 2, -math.inf, math.inf, s2)
    t2 = time.perf_counter()
    return {"value_minimax": round(v_plain, 4), "value_alphabeta": round(v_ab, 4),
            "nodes_minimax": s1.nodes_visited, "nodes_alphabeta": s2.nodes_visited,
            "leaves_minimax": s1.nodes_evaluated, "leaves_alphabeta": s2.nodes_evaluated,
            "pruned": s2.branches_pruned, "time_minimax": round(t1 - t0, 4), "time_alphabeta": round(t2 - t1, 4)}


def run(difficulty: str, rebuild: bool) -> dict:
    states = load_states(rebuild)
    rows = []
    for item in states:
        st = GameState.from_dict(item["state"])
        row = {"state": item["name"], "to_move": st.to_move}
        for kind in ("minimax", "mcts"):
            agent = make_agent(kind, st.to_move, difficulty, seed=123)
            d = agent.decide(st)
            s = d.stats
            row[kind] = {k: s.get(k) for k in ("decision_time", "simulations", "nodes_visited", "branches_pruned",
                                               "cutoffs", "depth_reached", "evaluation", "iterations",
                                               "avg_rollout_depth", "selected_visits", "selected_reward",
                                               "fuzzy_risk", "budget_exhausted")}
            row[kind]["action"] = d.action.describe()
        row["pruning"] = pruning_comparison(st)
        rows.append(row)
        print(f"{item['name']:<18} minimax {row['minimax']['decision_time']:.3f}s "
              f"mcts {row['mcts']['decision_time']:.3f}s | plain {row['pruning']['nodes_minimax']} nodes "
              f"vs alpha-beta {row['pruning']['nodes_alphabeta']} (same value: "
              f"{row['pruning']['value_minimax'] == row['pruning']['value_alphabeta']})", flush=True)
    return {"difficulty": difficulty, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}


def render(res: dict) -> str:
    rows = res["rows"]
    mm_t = [r["minimax"]["decision_time"] for r in rows]
    mc_t = [r["mcts"]["decision_time"] for r in rows]
    lines = [f"# Marble Fight — Benchmark on saved states ({res['difficulty']} budget)", "",
             f"Measured on {res['timestamp']} on this machine. States: `data/benchmark_states.json` "
             f"({len(rows)} positions).", "",
             "## Decision cost per state", "",
             "| State | MM time (s) | MM sims | MM nodes | MM pruned | MM depth | MCTS time (s) | MCTS sims | MCTS iters | MCTS visits |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m, c = r["minimax"], r["mcts"]
        lines.append(f"| {r['state']} | {m['decision_time']:.3f} | {m['simulations']} | {m['nodes_visited']} | "
                     f"{m['branches_pruned']} | {m['depth_reached']} | {c['decision_time']:.3f} | {c['simulations']} | "
                     f"{c['iterations']} | {c['selected_visits']} |")
    lines += ["", f"Mean decision time: Minimax {statistics.mean(mm_t):.3f} s (max {max(mm_t):.3f}), "
                  f"MCTS {statistics.mean(mc_t):.3f} s (max {max(mc_t):.3f}).", "",
              "## Alpha-Beta vs plain Minimax (depth 2, root width 12, reply width 8)", "",
              "| State | Value (plain) | Value (alpha-beta) | Nodes plain | Nodes alpha-beta | Saved | Branches pruned |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        p = r["pruning"]
        saved = 1 - p["nodes_alphabeta"] / p["nodes_minimax"] if p["nodes_minimax"] else 0
        lines.append(f"| {r['state']} | {p['value_minimax']} | {p['value_alphabeta']} | {p['nodes_minimax']} | "
                     f"{p['nodes_alphabeta']} | {saved * 100:.1f}% | {p['pruned']} |")
    same = all(r["pruning"]["value_minimax"] == r["pruning"]["value_alphabeta"] for r in rows)
    lines += ["", f"Identical root value in every state: **{same}**.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark Minimax and MCTS on saved states")
    ap.add_argument("--difficulty", default="normal", choices=["easy", "normal", "hard"])
    ap.add_argument("--rebuild-states", action="store_true")
    args = ap.parse_args(argv)
    res = run(args.difficulty, args.rebuild_states)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(REPORTS_DIR, f"benchmark_{args.difficulty}_{stamp}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render(res))
    with open(path.replace(".md", ".json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("Saved", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
