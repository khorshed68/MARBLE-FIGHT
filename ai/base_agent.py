"""Common agent interface, decision record and shared helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ai.evaluation import EvalWeights, DEFAULT_WEIGHTS, evaluate
from ai.fuzzy_risk import FuzzyRiskEvaluator
from ai.opponent_model import EvalBias, OpponentModel
from config import DEFAULT_CONFIG, GameConfig
from game.actions import Action, generate_candidate_actions, is_legal
from game.arena import Arena, get_arena
from game.simulation import TurnResult, simulate_turn
from game.state import GameState


class BudgetExceeded(Exception):
    """Raised inside a search when its node or time budget runs out."""


@dataclass
class Decision:
    action: Action
    algorithm: str
    player: int
    stats: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)  # a few scored alternatives for the overlay
    explanation: str = ""
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action.to_dict(), "algorithm": self.algorithm, "player": self.player,
                "stats": self.stats, "candidates": self.candidates, "explanation": self.explanation,
                "fallback": self.fallback}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Decision":
        return Decision(Action.from_dict(d["action"]), d["algorithm"], int(d["player"]), d.get("stats", {}),
                        d.get("candidates", []), d.get("explanation", ""), bool(d.get("fallback", False)))


class BaseAgent:
    """Shared machinery: simulation cache, evaluation, fuzzy risk, fallback."""

    algorithm = "base"

    def __init__(self, player: int, game_cfg: GameConfig = DEFAULT_CONFIG, use_fuzzy: bool = True,
                 use_opponent_model: bool = True, fuzzy_weight: float = 30.0,
                 weights: EvalWeights = DEFAULT_WEIGHTS, name: str | None = None) -> None:
        self.player = player
        self.game_cfg = game_cfg
        self.use_fuzzy = use_fuzzy
        self.use_opponent_model = use_opponent_model
        self.fuzzy_weight = fuzzy_weight
        self.weights = weights
        self.name = name or self.algorithm
        self.fuzzy = FuzzyRiskEvaluator()
        self.opponent_model = OpponentModel(player)  # owned by this agent, observes the other player's shots
        self._cache: dict[tuple, tuple[GameState, float]] = {}
        self.use_cache = True          # transition cache (state, action) -> (child, risk); see tests
        self._arena: Arena | None = None
        self.simulations = 0

    # ------------------------------------------------------------------ hooks
    def reset(self) -> None:
        """Forget everything between independent matches."""
        self.opponent_model = OpponentModel(self.player)
        self._cache.clear()

    def observe_opponent(self, before: GameState, action: Action, after: GameState) -> None:
        if self.use_opponent_model and before.to_move != self.player:
            self.opponent_model.arena_radius = get_arena(before.arena_id).radius
            self.opponent_model.observe(before, action.marble_id, action.angle, action.power, after)

    def decide(self, state: GameState, time_budget: float | None = None,
               cancel: Callable[[], bool] | None = None) -> Decision:  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers
    def _bias(self) -> EvalBias:
        return self.opponent_model.bias() if self.use_opponent_model else EvalBias.neutral()

    def _prepare(self, state: GameState) -> Arena:
        arena = get_arena(state.arena_id)
        if self._arena is None or self._arena.arena_id != arena.arena_id:
            self._cache.clear()
        self._arena = arena
        if len(self._cache) > 200_000:
            self._cache.clear()
        return arena

    def candidates(self, state: GameState, width: int) -> list[Action]:
        arena = self._arena or get_arena(state.arena_id)
        acts = generate_candidate_actions(state, self.game_cfg.rules, arena.radius, self.game_cfg.physics.marble_radius)
        return acts[:width] if width > 0 else acts

    def transition(self, state: GameState, action: Action) -> tuple[GameState, float]:
        """Simulate (cached) and return (child state, fuzzy risk of the shot for its shooter)."""
        key = (state.key(6), action.marble_id, round(action.angle, 9), round(action.power, 9), action.ability)
        hit = self._cache.get(key) if self.use_cache else None
        if hit is not None:
            return hit
        arena = self._arena or get_arena(state.arena_id)
        res: TurnResult = simulate_turn(state, action, arena, self.game_cfg, record=False, validate=False,
                                        record_contacts=False)
        self.simulations += 1
        risk = 0.0
        if self.use_fuzzy:
            risk = self.fuzzy.shot_risk(state, action, res.state, arena, self.game_cfg)
        out = (res.state, risk)
        if self.use_cache:
            self._cache[key] = out
        return out

    def edge_value(self, parent: GameState, child: GameState, risk: float, root_player: int) -> float:
        """Fuzzy risk as an additive edge cost, from the root player's perspective.

        A shot that ends the match in the shooter's favour is never penalised:
        fuzzy risk must not override a terminal win.
        """
        if not self.use_fuzzy or risk <= 0.0:
            return 0.0
        mover = parent.to_move
        if child.winner is not None and child.winner == mover:
            return 0.0
        pen = self.fuzzy_weight * risk
        return -pen if mover == root_player else pen

    def static_eval(self, state: GameState, perspective: int) -> float:
        arena = self._arena or get_arena(state.arena_id)
        return evaluate(state, perspective, arena.radius, self.game_cfg.physics.marble_radius, self.weights, self._bias())

    def fallback_action(self, state: GameState) -> Action:
        """A guaranteed-legal action: the highest-priority generated candidate."""
        acts = generate_candidate_actions(state, self.game_cfg.rules, include_abilities=False)
        for a in acts:
            if is_legal(state, a, self.game_cfg.rules):
                return a
        own = state.alive(state.to_move)
        return Action(own[0].mid, 0.0, self.game_cfg.rules.power_levels[0], "none", "fallback")


class Clock:
    """Time budget plus an optional cooperative cancellation check.

    ``cancel`` is any zero-argument callable (e.g. ``threading.Event().is_set`` or a
    ``multiprocessing.Event().is_set``). Searches poll :meth:`expired` between
    simulations, so a cancel request stops them within one physics simulation.
    """

    def __init__(self, budget: float | None, cancel: Callable[[], bool] | None = None) -> None:
        self.start = time.perf_counter()
        self.budget = budget
        self.cancel = cancel
        self.cancelled = False

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def expired(self) -> bool:
        if self.cancel is not None and not self.cancelled and self.cancel():
            self.cancelled = True
        return self.cancelled or (self.budget is not None and self.elapsed() > self.budget)
