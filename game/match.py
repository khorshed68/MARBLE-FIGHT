"""Headless match controller shared by the 3-D game, the tournament runner and tests.

A :class:`Match` owns the authoritative :class:`GameState`, the two seats
(human or AI agent), the turn history and the event feed. The UI never moves
marbles itself: it asks the match to ``apply`` an action and then plays back
the frames recorded by the physics engine.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from ai.base_agent import BaseAgent, Decision
from ai.factory import AGENT_LABELS, agent_from_spec, with_marbles
from config import DEFAULT_CONFIG, GameConfig
from game.actions import ABILITY_LABELS, Action, validate_action
from game.arena import Arena, get_arena
from game.rules import initial_state
from game.simulation import TurnResult, simulate_turn
from game.state import DRAW, PLAYER_NAMES, GameState


@dataclass
class PlayerSpec:
    kind: str = "minimax"              # "human", "minimax" or "mcts"
    difficulty: str = "normal"
    use_fuzzy: bool = True
    use_opponent_model: bool = True
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def is_human(self) -> bool:
        return self.kind == "human"

    @property
    def label(self) -> str:
        return "Human" if self.is_human else AGENT_LABELS.get(self.kind, self.kind)

    def agent_spec(self) -> dict[str, Any]:
        return {"kind": self.kind, "difficulty": self.difficulty, "use_fuzzy": self.use_fuzzy,
                "use_opponent_model": self.use_opponent_model, "overrides": dict(self.overrides)}

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "difficulty": self.difficulty, "use_fuzzy": self.use_fuzzy,
                "use_opponent_model": self.use_opponent_model, "overrides": dict(self.overrides)}


@dataclass
class TurnRecord:
    turn: int
    player: int
    action: dict[str, Any]
    decision: dict[str, Any] | None
    eliminated: tuple[int, ...]
    sim_time: float
    truncated: bool
    contacts: int
    fingerprint: str = ""          # state after the shot (for replay verification)


def decision_seed(match_seed: int, turn: int, player: int) -> int:
    """Deterministic per-decision seed for stochastic agents (MCTS)."""
    return (match_seed * 1_000_003 + turn * 7_919 + player * 104_729) & 0x7FFFFFFF


class Match:
    def __init__(self, arena_id: str, p1: PlayerSpec, p2: PlayerSpec, cfg: GameConfig = DEFAULT_CONFIG,
                 marbles: int | None = None, first_player: int = 0, seed: int = 0,
                 turn_limit: int | None = None) -> None:
        self.arena: Arena = get_arena(arena_id)
        self.specs = (p1, p2)
        self.cfg = with_marbles(cfg, marbles) if marbles else cfg
        self.first_player = first_player
        self.seed = seed
        self.turn_limit = turn_limit
        self.paused = False
        self.restart()

    # ------------------------------------------------------------------ lifecycle
    def restart(self) -> None:
        self.state: GameState = initial_state(self.arena, self.cfg, first_player=self.first_player,
                                              turn_limit=self.turn_limit)
        self.agents: list[BaseAgent | None] = [
            None if s.is_human else agent_from_spec(s.agent_spec(), i, seed=self.seed, game_cfg=self.cfg)
            for i, s in enumerate(self.specs)
        ]
        self.history: list[TurnRecord] = []
        self.events: list[str] = [f"Match start on {self.arena.name}: {self.seat_label(0)} vs {self.seat_label(1)}"]
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.truncations = 0
        self.fallbacks = 0
        self.paused = False

    def seat_label(self, player: int) -> str:
        return f"{PLAYER_NAMES[player]} ({self.specs[player].label})"

    @property
    def is_over(self) -> bool:
        return self.state.is_terminal

    def is_ai_turn(self) -> bool:
        return not self.is_over and self.agents[self.state.to_move] is not None

    def agent(self, player: int) -> BaseAgent | None:
        return self.agents[player]

    def current_seed(self) -> int:
        return decision_seed(self.seed, self.state.turn, self.state.to_move)

    # ------------------------------------------------------------------ decisions
    def compute_decision(self, time_budget: float | None = None) -> Decision:
        """Run the current AI synchronously (headless use)."""
        p = self.state.to_move
        agent = self.agents[p]
        if agent is None:
            raise RuntimeError("compute_decision called on a human turn")
        if hasattr(agent, "reseed"):
            agent.reseed(self.current_seed())  # type: ignore[attr-defined]
        return agent.decide(self.state, time_budget)

    def fallback_decision(self, reason: str) -> Decision:
        p = self.state.to_move
        agent = self.agents[p]
        action = agent.fallback_action(self.state) if agent else Action(self.state.alive(p)[0].mid, 0.0, 0.35)
        algo = agent.algorithm if agent else "human"
        return Decision(action, algo, p, {"algorithm": algo, "decision_time": None, "timeout": True}, [],
                        f"Fallback shot: {reason}", True)

    # ------------------------------------------------------------------ transitions
    def apply(self, action: Action, decision: Decision | None = None, record: bool = True) -> TurnResult:
        """Validate and execute one shot for the player to move."""
        before = self.state
        validate_action(before, action, self.cfg.rules)
        result = simulate_turn(before, action, self.arena, self.cfg, record=record)
        after = result.state
        for i, ag in enumerate(self.agents):
            if ag is not None and i != before.to_move:
                ag.observe_opponent(before, action, after)
        self.state = after
        if result.truncated:
            self.truncations += 1
        if decision is not None and decision.fallback:
            self.fallbacks += 1
        self.history.append(TurnRecord(
            before.turn, before.to_move, action.to_dict(), decision.to_dict() if decision else None,
            result.eliminated, result.sim_time, result.truncated, result.marble_contacts + result.obstacle_contacts,
            after.fingerprint(),
        ))
        self._log_events(before, action, result, decision)
        if after.is_terminal and self.finished_at is None:
            self.finished_at = time.time()
        return result

    def play_ai_turn(self, record: bool = False, time_budget: float | None = None) -> TurnResult | None:
        """Decide and apply one AI turn. Does nothing while paused or on a human turn."""
        if self.paused or not self.is_ai_turn():
            return None
        d = self.compute_decision(time_budget)
        return self.apply(d.action, d, record=record)

    def run_to_end(self, max_turns: int = 10_000) -> GameState:
        n = 0
        while not self.is_over and n < max_turns:
            if self.play_ai_turn() is None:
                break
            n += 1
        return self.state

    # ------------------------------------------------------------------ events & stats
    def _log_events(self, before: GameState, action: Action, res: TurnResult, decision: Decision | None) -> None:
        who = PLAYER_NAMES[before.to_move]
        deg = math.degrees(action.angle) % 360
        txt = f"T{before.turn + 1}: {who} shoots M{action.marble_id} at {deg:.0f}° power {action.power:.2f}"
        if action.ability != "none":
            txt += f" [{ABILITY_LABELS[action.ability]}]"
        self.events.append(txt)
        for mid in res.eliminated:
            owner = before.marbles[mid].owner
            if owner == before.to_move:
                self.events.append(f"   {who} knocked out its own marble M{mid}!")
            else:
                self.events.append(f"   KNOCKOUT! {who} eliminated {PLAYER_NAMES[owner]}'s marble M{mid}")
        if res.truncated:
            self.events.append("   (simulation hit the safety time cap; marbles frozen in place)")
        if decision is not None and decision.fallback:
            self.events.append(f"   {who} used a fallback shot ({decision.explanation})")
        if res.state.is_terminal:
            self.events.append(f"MATCH OVER: {res.state.end_reason}")

    def summary(self) -> dict[str, Any]:
        """Statistics of the (finished or running) match, computed from the history."""
        st = self.state
        end = self.finished_at or time.time()
        per: list[dict[str, Any]] = []
        for p in (0, 1):
            ds = [h.decision for h in self.history if h.player == p and h.decision]
            times = [d["stats"].get("decision_time") for d in ds if d["stats"].get("decision_time") is not None]
            s = {
                "seat": PLAYER_NAMES[p],
                "controller": self.specs[p].label,
                "kind": self.specs[p].kind,
                "difficulty": self.specs[p].difficulty,
                "remaining": st.count(p),
                "eliminated_by_opponent": sum(1 for h in self.history if h.player != p
                                              for mid in h.eliminated if st.marbles[mid].owner == p),
                "self_eliminations": sum(1 for h in self.history if h.player == p
                                         for mid in h.eliminated if st.marbles[mid].owner == p),
                "decisions": len(ds),
                "avg_decision_time": round(sum(times) / len(times), 4) if times else None,
                "max_decision_time": round(max(times), 4) if times else None,
                "fallbacks": sum(1 for d in ds if d.get("fallback")),
            }
            if self.specs[p].kind == "minimax":
                s["minimax_nodes"] = sum(d["stats"].get("nodes_visited", 0) or 0 for d in ds)
                s["alphabeta_prunes"] = sum(d["stats"].get("branches_pruned", 0) or 0 for d in ds)
                s["alphabeta_cutoffs"] = sum(d["stats"].get("cutoffs", 0) or 0 for d in ds)
            if self.specs[p].kind == "mcts":
                s["mcts_iterations"] = sum(d["stats"].get("iterations", 0) or 0 for d in ds)
            s["physics_simulations"] = sum(d["stats"].get("simulations", 0) or 0 for d in ds)
            per.append(s)
        winner = st.winner
        return {
            "arena": self.arena.arena_id,
            "arena_name": self.arena.name,
            "winner": winner,
            "winner_label": ("Draw" if winner == DRAW else self.seat_label(winner)) if winner is not None else None,
            "end_reason": st.end_reason,
            "turns": st.turn,
            "duration_s": round(end - self.started_at, 2),
            "score": [st.count(0), st.count(1)],
            "players": per,
            "eliminated_total": sum(len(h.eliminated) for h in self.history),
            "truncations": self.truncations,
            "fallbacks": self.fallbacks,
            "seed": self.seed,
            "first_player": self.first_player,
            "marbles_per_player": self.cfg.rules.marbles_per_player,
        }

    def decision_records(self) -> list[dict[str, Any]]:
        """Flat per-decision stats for analytics."""
        out = []
        for h in self.history:
            if not h.decision:
                continue
            s = dict(h.decision["stats"])
            s.pop("opponent_profile", None)
            s.update(turn=h.turn, player=h.player, kind=self.specs[h.player].kind, fallback=h.decision.get("fallback"))
            out.append(s)
        return out
