"""Automatic multi-match tournaments with real, reproducible results.

Usage (from the project folder)::

    python -m analytics.tournament --matches 12 --pairing minimax:mcts --arenas classic,neon,magnetic

Methodology (see AI_ALGORITHMS.md, section 16):

* For pairing ``A:B`` match *i* seats A as Player 1 when *i* is even and B as
  Player 1 when *i* is odd, so each algorithm moves first equally often.
* Arenas rotate through the given list (``i mod len(arenas)``).
* Match *i* uses seed ``base_seed + i``; MCTS derives a per-decision seed from
  it. Budgets are node/iteration limits (machine-independent); the time budget
  is only a safety cap, so fixed seeds reproduce the same games unless a
  machine is so slow that the cap is hit (reported as ``budget_exhausted``).
* Matches may run in parallel worker processes; each match is independent so
  parallelism does not change results.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

if __package__ in (None, ""):  # allow `python analytics/tournament.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.match_logger import append_record, build_record
from analytics.metrics import aggregate, head_to_head
from config import REPORTS_DIR, TOURNAMENT_DIR


@dataclass
class TournamentConfig:
    matches: int = 6
    pairings: list[tuple[str, str]] = field(default_factory=lambda: [("minimax", "mcts")])
    arenas: list[str] = field(default_factory=lambda: ["classic", "neon", "magnetic"])
    difficulty: str = "normal"
    rotate_start: bool = True
    base_seed: int = 42
    marbles: int = 3
    turn_limit: int | None = None
    use_fuzzy: bool = True
    use_opponent_model: bool = True
    workers: int = 0                     # 0 = automatic, 1 = sequential
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)  # per-kind budget overrides
    log_history: bool = True
    save_replays: bool = True


def schedule(cfg: TournamentConfig) -> list[dict[str, Any]]:
    jobs = []
    for i in range(cfg.matches):
        a, b = cfg.pairings[i % len(cfg.pairings)]
        if cfg.rotate_start and (i // len(cfg.pairings)) % 2 == 1:
            a, b = b, a
        jobs.append({
            "index": i, "p1": a, "p2": b, "arena": cfg.arenas[i % len(cfg.arenas)], "seed": cfg.base_seed + i,
            "difficulty": cfg.difficulty, "marbles": cfg.marbles, "turn_limit": cfg.turn_limit,
            "use_fuzzy": cfg.use_fuzzy, "use_opponent_model": cfg.use_opponent_model,
            "overrides": cfg.overrides,
        })
    return jobs


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    """Play one complete AI-vs-AI match (top-level so it can run in a process)."""
    from game.match import Match, PlayerSpec

    def spec(kind: str) -> PlayerSpec:
        return PlayerSpec(kind, job["difficulty"], job["use_fuzzy"], job["use_opponent_model"],
                          dict(job.get("overrides", {}).get(kind, {})))

    m = Match(job["arena"], spec(job["p1"]), spec(job["p2"]), marbles=job["marbles"], seed=job["seed"],
              turn_limit=job["turn_limit"])
    m.run_to_end()
    from game.replay import record_from_match
    rec = build_record(m, "tournament")
    rec["job"] = job
    rec["replay"] = record_from_match(m, "tournament")
    return rec


def run_tournament(cfg: TournamentConfig, progress: Callable[[int, int, dict], None] | None = None,
                   cancel: Callable[[], bool] | None = None, save: bool = True) -> dict[str, Any]:
    tid = time.strftime("T%Y%m%d-%H%M%S")
    jobs = schedule(cfg)
    workers = cfg.workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1, 6))
    records: list[dict[str, Any]] = []
    t0 = time.time()

    def done(rec: dict[str, Any]) -> None:
        rec["tournament_id"] = tid
        replay = rec.pop("replay", None)
        if replay is not None and cfg.save_replays:
            from game.replay import save_replay
            save_replay(replay, f"{tid}-m{rec['job']['index']:02d}")
        records.append(rec)
        if cfg.log_history:
            append_record(rec)
        if progress:
            progress(len(records), len(jobs), rec)

    if workers <= 1:
        for job in jobs:
            if cancel and cancel():
                break
            done(run_one(job))
    else:
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(run_one, j) for j in jobs]
            for fut in cf.as_completed(futs):
                if cancel and cancel():
                    for f in futs:
                        f.cancel()
                    break
                done(fut.result())
    records.sort(key=lambda r: r["job"]["index"])
    result = {
        "tournament_id": tid,
        "config": {**asdict(cfg), "pairings": [list(p) for p in cfg.pairings]},
        "wall_time_s": round(time.time() - t0, 2),
        "completed": len(records),
        "scheduled": len(jobs),
        "aggregate": aggregate(records),
        "head_to_head": head_to_head(records),
        "matches": [
            {"index": r["job"]["index"], "arena": r["summary"]["arena"], "p1": r["job"]["p1"], "p2": r["job"]["p2"],
             "seed": r["job"]["seed"], "winner": r["summary"]["winner"], "winner_label": r["summary"]["winner_label"],
             "score": r["summary"]["score"], "turns": r["summary"]["turns"],
             "duration_s": r["summary"]["duration_s"], "end_reason": r["summary"]["end_reason"]}
            for r in records
        ],
    }
    if save:
        result["files"] = save_result(result)
    return result


def save_result(result: dict[str, Any]) -> dict[str, str]:
    os.makedirs(TOURNAMENT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    tid = result["tournament_id"]
    jpath = os.path.join(TOURNAMENT_DIR, f"{tid}.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    mpath = os.path.join(REPORTS_DIR, f"tournament_{tid}.md")
    with open(mpath, "w", encoding="utf-8") as fh:
        fh.write(render_report(result))
    return {"json": jpath, "report": mpath}


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_report(result: dict[str, Any]) -> str:
    c = result["config"]
    agg = result["aggregate"]
    h2h = result["head_to_head"]
    lines = [
        f"# Marble Fight — Tournament Report {result['tournament_id']}",
        "",
        "> **Measured results.** Every number below was produced by actually playing the matches listed.",
        "",
        "## Configuration",
        "",
        f"- Matches: {result['completed']} completed of {result['scheduled']} scheduled",
        f"- Pairings: {', '.join(' vs '.join(p) for p in c['pairings'])}",
        f"- Arenas (rotation): {', '.join(c['arenas'])}",
        f"- Difficulty preset: {c['difficulty']}; marbles per player: {c['marbles']}",
        f"- Starting-seat rotation: {c['rotate_start']}; base seed: {c['base_seed']}",
        f"- Fuzzy risk: {c['use_fuzzy']}; history-based adaptation: {c['use_opponent_model']}",
        f"- Wall time: {result['wall_time_s']} s",
        "",
        "## Head to head (Minimax vs MCTS)",
        "",
        f"Minimax wins: {h2h['minimax']} · MCTS wins: {h2h['mcts']} · Draws: {h2h['draw']} · Matches: {h2h['matches']}",
        "",
        "## Per-algorithm metrics",
        "",
        "| Metric | " + " | ".join(agg["algorithms"].keys()) + " |",
        "|---|" + "---|" * len(agg["algorithms"]),
    ]
    keys = [("matches", 0), ("wins", 0), ("losses", 0), ("draws", 0), ("win_rate", 3), ("decisions", 0),
            ("avg_decision_time", 3), ("max_decision_time", 3), ("total_nodes", 0), ("total_leaves", 0),
            ("total_prunes", 0), ("total_cutoffs", 0), ("avg_depth", 2), ("max_depth", 0), ("max_tree_depth", 0), ("total_iterations", 0), ("total_simulations", 0), ("avg_rollout_depth", 2),
            ("avg_selected_visits", 1), ("avg_value", 3), ("avg_fuzzy_risk", 3), ("fallbacks", 0)]
    for k, nd in keys:
        lines.append(f"| {k} | " + " | ".join(_f(a.get(k), nd) for a in agg["algorithms"].values()) + " |")
    lines += ["", "## Results by arena and pairing", "",
              "| Pairing | Arena | Matches | Avg turns | Wins | Draws |", "|---|---|---|---|---|---|"]
    for v in agg["pairings"]:
        wins = ", ".join(f"{k} {n}" for k, n in sorted(v["wins"].items())) or "-"
        lines.append(f"| {v['pairing']} | {v['arena']} | {v['matches']} | {_f(v['avg_turns'], 1)} | {wins} | {v['draws']} |")
    lines += ["", "`n/a` = metric not measured for that agent (not zero).", ""]
    lines += ["", "## Matches", "", "| # | Arena | P1 | P2 | Seed | Winner | Score | Turns | Time (s) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m in result["matches"]:
        lines.append(f"| {m['index']} | {m['arena']} | {m['p1']} | {m['p2']} | {m['seed']} | {m['winner_label']} | "
                     f"{m['score'][0]}-{m['score'][1]} | {m['turns']} | {m['duration_s']} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a Marble Fight AI tournament.")
    ap.add_argument("--matches", type=int, default=6)
    ap.add_argument("--pairing", action="append", default=None,
                    help="e.g. minimax:mcts (repeatable). Default minimax:mcts")
    ap.add_argument("--arenas", default="classic,neon,magnetic")
    ap.add_argument("--difficulty", default="normal", choices=["easy", "normal", "hard"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--marbles", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--no-fuzzy", action="store_true")
    ap.add_argument("--no-adapt", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="do not append to data/match_history.jsonl")
    ap.add_argument("--no-replays", action="store_true", help="do not save a replay file per match")
    args = ap.parse_args(argv)
    pairings = [tuple(p.split(":")) for p in (args.pairing or ["minimax:mcts"])]
    cfg = TournamentConfig(matches=args.matches, pairings=pairings,  # type: ignore[arg-type]
                           arenas=[a.strip() for a in args.arenas.split(",") if a.strip()],
                           difficulty=args.difficulty, rotate_start=not args.no_rotate, base_seed=args.seed,
                           marbles=args.marbles, use_fuzzy=not args.no_fuzzy, use_opponent_model=not args.no_adapt,
                           workers=args.workers, log_history=not args.no_log, save_replays=not args.no_replays)

    def prog(i: int, n: int, rec: dict) -> None:
        s = rec["summary"]
        print(f"[{i}/{n}] {rec['job']['p1']} vs {rec['job']['p2']} on {s['arena']}: "
              f"{s['winner_label']} ({s['score'][0]}-{s['score'][1]}, {s['turns']} turns)", flush=True)

    res = run_tournament(cfg, prog)
    h = res["head_to_head"]
    print(f"\nHead to head: Minimax {h['minimax']}  MCTS {h['mcts']}  Draws {h['draw']}")
    for k, a in res["aggregate"]["algorithms"].items():
        print(f"{k:8s} win rate {_f(a['win_rate'])}  avg decision {_f(a['avg_decision_time'])} s")
    print(f"Saved: {res['files']['json']}\n       {res['files']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
