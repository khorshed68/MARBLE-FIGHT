"""Aggregate measured match records into the numbers shown in the Algorithm Lab.

Every number is computed from logged matches; nothing is invented.

* A metric that was never measured is ``None`` (shown as "n/a"); a measured
  zero stays ``0``. For example Alpha-Beta prunes are ``None`` for MCTS and for
  humans, but can be a genuine ``0`` for a Minimax that never had a cutoff.
* With no history the functions return empty structures and the UI shows an
  empty state.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

ALGOS = ("minimax", "mcts", "human")


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _max(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def _sum_or_none(xs: list) -> int | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) if xs else None


def pairing_key(kinds: list[str]) -> str:
    return " vs ".join(sorted(kinds))


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return per-algorithm, per-arena and per-(pairing, arena) statistics."""
    per: dict[str, dict[str, Any]] = {}
    for k in ALGOS:
        per[k] = {"matches": 0, "wins": 0, "losses": 0, "draws": 0, "decision_times": [], "nodes": [],
                  "leaves": [], "prunes": [], "cutoffs": [], "depths": [], "iterations": [], "simulations": [],
                  "rollout_depths": [], "selected_visits": [], "values": [], "risks": [], "fallbacks": 0,
                  "decisions": 0, "tree_depths": []}
    arenas: dict[str, dict[str, Any]] = defaultdict(lambda: {"matches": 0, "turns": [], "results": defaultdict(int)})
    pairings: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"matches": 0, "turns": [], "wins": defaultdict(int), "draws": 0})
    for rec in records:
        s = rec.get("summary", {})
        players = s.get("players", [])
        winner = s.get("winner")
        kinds = [p.get("kind") for p in players]
        for seat, kind in enumerate(kinds):
            if kind not in per:
                continue
            d = per[kind]
            d["matches"] += 1
            if winner == seat:
                d["wins"] += 1
            elif winner == -1:
                d["draws"] += 1
            elif winner is not None:
                d["losses"] += 1
        arena = s.get("arena", "?")
        a = arenas[arena]
        a["matches"] += 1
        a["turns"].append(s.get("turns", 0))
        if winner == -1:
            a["results"]["draw"] += 1
        elif winner in (0, 1) and len(kinds) == 2:
            a["results"][kinds[winner]] += 1
        if len(kinds) == 2:
            pr = pairings[(pairing_key(kinds), arena)]
            pr["matches"] += 1
            pr["turns"].append(s.get("turns", 0))
            if winner == -1:
                pr["draws"] += 1
            elif winner in (0, 1):
                # mirror matches are reported by seat, otherwise by algorithm
                label = f"P{winner + 1}" if kinds[0] == kinds[1] else kinds[winner]
                pr["wins"][label] += 1
        specs = rec.get("specs", [])
        for dec in rec.get("decisions", []):
            kind = dec.get("kind")
            if kind not in per:
                continue
            seat = dec.get("player")
            fuzzy_on = specs[seat].get("use_fuzzy", True) if isinstance(seat, int) and seat < len(specs) else True
            d = per[kind]
            d["decisions"] += 1
            if dec.get("fallback"):
                d["fallbacks"] += 1
            d["decision_times"].append(dec.get("decision_time"))
            d["simulations"].append(dec.get("simulations"))
            if fuzzy_on:   # risk is not measured when the fuzzy evaluator is disabled
                d["risks"].append(dec.get("fuzzy_risk"))
            if kind == "minimax":
                d["nodes"].append(dec.get("nodes_visited"))
                d["leaves"].append(dec.get("nodes_evaluated"))
                d["prunes"].append(dec.get("branches_pruned"))
                d["cutoffs"].append(dec.get("cutoffs"))
                d["depths"].append(dec.get("depth_reached"))
                d["values"].append(dec.get("evaluation"))
            elif kind == "mcts":
                d["iterations"].append(dec.get("iterations"))
                d["rollout_depths"].append(dec.get("avg_rollout_depth"))
                d["selected_visits"].append(dec.get("selected_visits"))
                d["values"].append(dec.get("selected_reward"))
                d["tree_depths"].append(dec.get("max_tree_depth"))

    out_algos = {}
    for k, d in per.items():
        if d["matches"] == 0 and d["decisions"] == 0:
            continue
        decided = d["wins"] + d["losses"] + d["draws"]
        out_algos[k] = {
            "matches": d["matches"], "wins": d["wins"], "losses": d["losses"], "draws": d["draws"],
            "win_rate": d["wins"] / decided if decided else None,
            "decisions": d["decisions"],
            "avg_decision_time": _mean(d["decision_times"]),
            "max_decision_time": _max(d["decision_times"]),
            "total_nodes": _sum_or_none(d["nodes"]),
            "total_leaves": _sum_or_none(d["leaves"]),
            "total_prunes": _sum_or_none(d["prunes"]),
            "total_cutoffs": _sum_or_none(d["cutoffs"]),
            "avg_depth": _mean(d["depths"]),
            "max_depth": _max(d["depths"]),
            "total_iterations": _sum_or_none(d["iterations"]),
            "total_simulations": _sum_or_none(d["simulations"]),
            "avg_rollout_depth": _mean(d["rollout_depths"]),
            "avg_selected_visits": _mean(d["selected_visits"]),
            "max_tree_depth": _max(d["tree_depths"]),
            "avg_value": _mean(d["values"]),
            "avg_fuzzy_risk": _mean(d["risks"]),
            "fallbacks": d["fallbacks"] if d["decisions"] else None,
        }
    out_arenas = {
        k: {"matches": v["matches"], "avg_turns": _mean(v["turns"]), "results": dict(v["results"])}
        for k, v in arenas.items()
    }
    out_pairings = [
        {"pairing": pk, "arena": ar, "matches": v["matches"], "avg_turns": _mean(v["turns"]),
         "wins": dict(v["wins"]), "draws": v["draws"]}
        for (pk, ar), v in sorted(pairings.items())
    ]
    return {"total_matches": len(records), "algorithms": out_algos, "arenas": out_arenas, "pairings": out_pairings}


def head_to_head(records: list[dict[str, Any]]) -> dict[str, int]:
    """Minimax-vs-MCTS results only (both seats AI, different algorithms)."""
    res = {"minimax": 0, "mcts": 0, "draw": 0, "matches": 0}
    for rec in records:
        s = rec.get("summary", {})
        kinds = [p.get("kind") for p in s.get("players", [])]
        if sorted(kinds) != ["mcts", "minimax"]:
            continue
        res["matches"] += 1
        w = s.get("winner")
        if w == -1:
            res["draw"] += 1
        elif w in (0, 1):
            res[kinds[w]] += 1
    return res
