"""Minimax with Alpha-Beta pruning.

The search core (:func:`minimax`, :func:`alphabeta`) is written against a tiny
abstract :class:`SearchProblem` interface, so the very same functions are
tested on hand-built game trees *and* used by :class:`MinimaxAgent` on real
marble states.

One ply = one complete player turn (choose shot -> resolve all physics until
the arena settles). Players never alternate inside a physics simulation.

Edge values: a move may carry an additive value (the fuzzy-risk penalty of the
shot). For a node, ``value = max/min over children of (edge + value(child))``.
Alpha-beta handles this by searching each child with the window shifted by the
edge value, which keeps pruning exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, TypeVar

from ai.base_agent import BaseAgent, BudgetExceeded, Clock, Decision
from config import DEFAULT_CONFIG, GameConfig, MinimaxConfig
from game.actions import Action
from game.state import GameState

N = TypeVar("N")
A = TypeVar("A")


class SearchProblem(Protocol[N, A]):
    def is_terminal(self, node: N) -> bool: ...
    def evaluate(self, node: N) -> float: ...                  # from the MAX player's perspective
    def is_max(self, node: N) -> bool: ...
    def actions(self, node: N, depth: int) -> Sequence[A]: ...  # ordered
    def result(self, node: N, action: A) -> tuple[float, N]: ...  # (edge value, child)


@dataclass
class SearchStats:
    nodes_visited: int = 0      # every node entered by the search
    nodes_evaluated: int = 0    # leaves scored with the heuristic / terminal value
    cutoffs: int = 0            # alpha-beta cut events
    branches_pruned: int = 0    # sibling moves skipped because of cutoffs
    max_depth_reached: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def minimax(problem: SearchProblem, node, depth: int, stats: SearchStats, ply: int = 0) -> float:
    """Plain (unpruned) minimax. Reference implementation for tests."""
    stats.nodes_visited += 1
    stats.max_depth_reached = max(stats.max_depth_reached, ply)
    if depth == 0 or problem.is_terminal(node):
        stats.nodes_evaluated += 1
        return problem.evaluate(node)
    acts = problem.actions(node, depth)
    if not acts:
        stats.nodes_evaluated += 1
        return problem.evaluate(node)
    if problem.is_max(node):
        best = -math.inf
        for a in acts:
            e, child = problem.result(node, a)
            best = max(best, e + minimax(problem, child, depth - 1, stats, ply + 1))
    else:
        best = math.inf
        for a in acts:
            e, child = problem.result(node, a)
            best = min(best, e + minimax(problem, child, depth - 1, stats, ply + 1))
    return best


def alphabeta(problem: SearchProblem, node, depth: int, alpha: float, beta: float, stats: SearchStats,
              ply: int = 0, root_values: list | None = None) -> float:
    """Fail-soft alpha-beta. Returns the exact minimax value when it lies in
    (alpha, beta); otherwise a bound on the correct side (which may lie outside
    the window: a fail-low value is an upper bound, a fail-high value a lower bound).

    ``root_values`` (only at the top call) collects (action, value) pairs.
    """
    stats.nodes_visited += 1
    stats.max_depth_reached = max(stats.max_depth_reached, ply)
    if depth == 0 or problem.is_terminal(node):
        stats.nodes_evaluated += 1
        return problem.evaluate(node)
    acts = problem.actions(node, depth)
    if not acts:
        stats.nodes_evaluated += 1
        return problem.evaluate(node)
    if problem.is_max(node):
        value = -math.inf
        for i, a in enumerate(acts):
            e, child = problem.result(node, a)
            v = e + alphabeta(problem, child, depth - 1, alpha - e, beta - e, stats, ply + 1)
            if root_values is not None:
                root_values.append((a, v))
            if v > value:
                value = v
            if value > alpha:
                alpha = value
            if alpha >= beta:
                stats.cutoffs += 1
                stats.branches_pruned += len(acts) - i - 1
                break
        return value
    value = math.inf
    for i, a in enumerate(acts):
        e, child = problem.result(node, a)
        v = e + alphabeta(problem, child, depth - 1, alpha - e, beta - e, stats, ply + 1)
        if root_values is not None:
            root_values.append((a, v))
        if v < value:
            value = v
        if value < beta:
            beta = value
        if alpha >= beta:
            stats.cutoffs += 1
            stats.branches_pruned += len(acts) - i - 1
            break
    return value


# ============================================================================ marble adapter
class MarbleSearchProblem:
    """Adapts the marble game to :class:`SearchProblem` for one decision."""

    def __init__(self, agent: "MinimaxAgent", root_player: int, clock: Clock, cfg: MinimaxConfig,
                 root_order: list[Action] | None = None) -> None:
        self.agent = agent
        self.root_player = root_player
        self.clock = clock
        self.cfg = cfg
        self.root_order = root_order
        self.root_state: GameState | None = None
        self.sims = 0

    def _charge(self) -> None:
        self.sims += 1
        if self.sims > self.cfg.node_budget or self.clock.expired():
            raise BudgetExceeded()

    def is_terminal(self, node: GameState) -> bool:
        return node.winner is not None

    def evaluate(self, node: GameState) -> float:
        return self.agent.static_eval(node, self.root_player)

    def is_max(self, node: GameState) -> bool:
        return node.to_move == self.root_player

    def actions(self, node: GameState, depth: int) -> list[Action]:
        is_root = node is self.root_state
        width = self.cfg.root_width if is_root else self.cfg.inner_width
        acts = self.agent.candidates(node, width)
        if is_root and self.root_order:
            # iterative deepening: best moves of the previous iteration first
            first = [a for a in self.root_order if a in acts]
            acts = first + [a for a in acts if a not in first]
        if depth > 1:
            # children are interior nodes: simulate all of them now and order by static value
            # (these simulations are reused by result(), nothing is wasted)
            scored = []
            for a in acts:
                e, child = self.result(node, a)
                scored.append((e + self.evaluate(child), a))
            maximize = self.is_max(node)
            if is_root and self.root_order:
                k = len([a for a in self.root_order if a in acts])
                head, tail = scored[:k], scored[k:]
                tail.sort(key=lambda t: -t[0] if maximize else t[0])
                scored = head + tail
            else:
                scored.sort(key=lambda t: -t[0] if maximize else t[0])
            return [a for _, a in scored]
        # children are leaves: keep the cheap geometric order and simulate lazily so that
        # alpha-beta cutoffs genuinely save physics simulations
        return acts

    def result(self, node: GameState, action: Action) -> tuple[float, GameState]:
        before = self.agent.simulations
        child, risk = self.agent.transition(node, action)
        if self.agent.simulations != before:
            self._charge()
        elif self.clock.expired():
            raise BudgetExceeded()
        return self.agent.edge_value(node, child, risk, self.root_player), child


class MinimaxAgent(BaseAgent):
    algorithm = "Minimax + Alpha-Beta"

    def __init__(self, player: int, config: MinimaxConfig | None = None, game_cfg: GameConfig = DEFAULT_CONFIG,
                 name: str | None = None, **_: Any) -> None:
        self.config = config or MinimaxConfig()
        super().__init__(player, game_cfg, self.config.use_fuzzy, self.config.use_opponent_model,
                         self.config.fuzzy_weight, name=name or "Minimax")

    def decide(self, state: GameState, time_budget: float | None = None,
               cancel: Callable[[], bool] | None = None) -> Decision:
        cfg = self.config
        clock = Clock(time_budget if time_budget is not None else cfg.time_budget, cancel)
        self._prepare(state)
        if state.to_move != self.player:
            raise ValueError("MinimaxAgent asked to move for the wrong player")
        sims_before = self.simulations
        total = SearchStats()
        best_action: Action | None = None
        best_value = -math.inf
        best_values: list[tuple[Action, float]] = []
        completed_depth = 0
        order: list[Action] | None = None
        budget_hit = False
        search_order_sims = 0

        for depth in range(1, max(1, cfg.max_depth) + 1):
            problem = MarbleSearchProblem(self, self.player, clock, cfg, order)
            problem.sims = search_order_sims
            problem.root_state = state
            stats = SearchStats()
            root_values: list[tuple[Action, float]] = []
            try:
                value = alphabeta(problem, state, depth, -math.inf, math.inf, stats, 0, root_values)
            except BudgetExceeded:
                budget_hit = True
                self._merge(total, stats)
                break
            search_order_sims = problem.sims
            self._merge(total, stats)
            completed_depth = depth
            # best = first action that achieved the returned value (exact at the root)
            for a, v in root_values:
                if v == value:
                    best_action, best_value = a, v
                    break
            best_values = root_values
            # next iteration: principal move first, then the rest by their (bounded) values
            ranked = sorted(root_values, key=lambda t: -t[1])
            order = [a for a, _ in ranked]
            if abs(value) >= 9_000:   # forced result found; deeper search cannot change it
                break

        fallback = best_action is None
        if fallback:
            best_action = self.fallback_action(state)
            best_value = float("nan")

        elapsed = clock.elapsed()
        _, risk = self.transition(state, best_action) if not fallback else (None, None)
        cands = []
        seen = set()
        for a, v in sorted(best_values, key=lambda t: -t[1])[:4]:
            if a in seen:
                continue
            seen.add(a)
            _, r = self.transition(state, a)
            cands.append({"action": a.to_dict(), "value": round(v, 2), "risk": round(r, 3),
                          "selected": a == best_action})
        stats = {
            "algorithm": self.algorithm,
            "depth_reached": completed_depth,
            "max_depth": cfg.max_depth,
            "nodes_visited": total.nodes_visited,
            "nodes_evaluated": total.nodes_evaluated,
            "cutoffs": total.cutoffs,
            "branches_pruned": total.branches_pruned,
            "simulations": self.simulations - sims_before,
            "evaluation": None if fallback else round(best_value, 2),
            "fuzzy_risk": None if (risk is None or not self.use_fuzzy) else round(risk, 3),
            "decision_time": round(elapsed, 4),
            "budget_exhausted": budget_hit,
            "cancelled": clock.cancelled,
            "root_candidates": len(best_values),
            "opponent_profile": self.opponent_model.profile() if self.use_opponent_model else None,
        }
        expl = (f"Searched {completed_depth} ply(s): {total.nodes_visited} nodes, {total.branches_pruned} branches pruned. "
                f"Chose {best_action.tag or 'shot'} with backed-up value {stats['evaluation']}.")
        if fallback:
            expl = "Budget ran out before depth 1 finished; using the highest-priority legal shot."
        if clock.cancelled:
            expl = "Search cancelled; " + expl
        return Decision(best_action, self.algorithm, self.player, stats, cands, expl, fallback)

    @staticmethod
    def _merge(total: SearchStats, s: SearchStats) -> None:
        total.nodes_visited += s.nodes_visited
        total.nodes_evaluated += s.nodes_evaluated
        total.cutoffs += s.cutoffs
        total.branches_pruned += s.branches_pruned
        total.max_depth_reached = max(total.max_depth_reached, s.max_depth_reached)
