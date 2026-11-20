"""Monte Carlo Tree Search with UCT (UCB1 applied to trees).

Each iteration runs the four classic phases:

1. **Selection** – from the root, repeatedly pick the child maximising
   ``UCT = W/N + c * sqrt(ln(N_parent) / N)`` while the node is fully expanded.
2. **Expansion** – add one untried action (progressive widening: a node with
   ``N`` visits may have at most ``ceil(widening_c * N ** widening_alpha)``
   children; untried actions are taken in candidate-priority order). The action
   is resolved with the shared physics transition ``simulate_turn``.
3. **Simulation (rollout)** – play ``rollout_depth`` further turns with a light
   heuristic policy (sample a few candidate shots, keep the one with the best
   quick evaluation minus fuzzy risk; epsilon-random otherwise). The final
   state is scored: terminal = 1 / 0.5 / 0, else logistic(evaluation / scale).
4. **Backpropagation** – add the reward to every node on the path.

Reward perspective: the reward ``r`` of an iteration is computed for the *root
player*. A node stores the total reward of the player who made the move that
leads *into* the node (``node.mover``): ``r`` if mover == root player, else
``1 - r``. Therefore every parent can simply maximise ``W/N`` of its children
in UCT, whichever player is to move.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable

from ai.base_agent import BaseAgent, Clock, Decision
from config import DEFAULT_CONFIG, GameConfig, MCTSConfig
from game.actions import Action
from game.state import DRAW, GameState


def uct_value(child_value_sum: float, child_visits: int, parent_visits: int, c: float) -> float:
    """UCT = exploitation + c * sqrt(ln(parent) / child). Unvisited children score +inf."""
    if child_visits == 0:
        return math.inf
    return child_value_sum / child_visits + c * math.sqrt(math.log(max(1, parent_visits)) / child_visits)


class MCTSNode:
    __slots__ = ("state", "parent", "action", "mover", "children", "untried", "visits", "value_sum",
                 "edge", "risk", "depth")

    def __init__(self, state: GameState, parent: "MCTSNode | None", action: Action | None, mover: int,
                 edge: float = 0.0, risk: float = 0.0) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.mover = mover                 # player who made `action` (reward owner); root: opponent of to_move
        self.children: list[MCTSNode] = []
        self.untried: list[Action] | None = None   # filled lazily
        self.visits = 0
        self.value_sum = 0.0
        self.edge = edge                   # fuzzy edge value of `action`, root-player perspective
        self.risk = risk
        self.depth = 0 if parent is None else parent.depth + 1

    @property
    def is_terminal(self) -> bool:
        return self.state.winner is not None

    @property
    def mean(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MCTSAgent(BaseAgent):
    algorithm = "MCTS (UCT)"

    def __init__(self, player: int, config: MCTSConfig | None = None, game_cfg: GameConfig = DEFAULT_CONFIG,
                 seed: int = 0, name: str | None = None, **_: Any) -> None:
        self.config = config or MCTSConfig()
        super().__init__(player, game_cfg, self.config.use_fuzzy, self.config.use_opponent_model,
                         self.config.fuzzy_weight, name=name or "MCTS")
        self.seed = seed
        self.rng = random.Random(seed)

    def reseed(self, seed: int) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ phases
    def _widening_limit(self, node: MCTSNode) -> int:
        c = self.config
        return max(1, math.ceil(c.widening_c * max(1, node.visits) ** c.widening_alpha))

    def _untried(self, node: MCTSNode) -> list[Action]:
        if node.untried is None:
            node.untried = list(self.candidates(node.state, self.config.candidate_width))
        return node.untried

    def select(self, node: MCTSNode) -> tuple[MCTSNode, float]:
        """Descend with UCT; returns (node to expand or terminal leaf, accumulated edge value)."""
        acc = node.edge if node.parent is not None else 0.0
        while not node.is_terminal:
            untried = self._untried(node)
            if untried and len(node.children) < self._widening_limit(node):
                return node, acc
            if not node.children:
                return node, acc
            c = self.config.exploration
            node = max(node.children, key=lambda ch: (uct_value(ch.value_sum, ch.visits, node.visits, c), -ch.depth))
            acc += node.edge
        return node, acc

    def expand(self, node: MCTSNode) -> MCTSNode:
        untried = self._untried(node)
        action = untried.pop(0)
        child_state, risk = self.transition(node.state, action)
        edge = self.edge_value(node.state, child_state, risk, self.player)
        child = MCTSNode(child_state, node, action, node.state.to_move, edge, risk)
        node.children.append(child)
        return child

    def terminal_reward(self, state: GameState) -> float:
        if state.winner == DRAW:
            return 0.5
        return 1.0 if state.winner == self.player else 0.0

    def rollout(self, state: GameState, acc_edge: float) -> tuple[float, int]:
        """Light-policy playout. Returns (reward for the root player, turns played)."""
        cfg = self.config
        depth = 0
        while depth < cfg.rollout_depth and state.winner is None:
            pool = self.candidates(state, 12)
            if not pool:
                break
            mover = state.to_move
            if self.rng.random() < cfg.rollout_epsilon:
                action = self.rng.choice(pool)
                child, risk = self.transition(state, action)
            else:
                picks = self.rng.sample(pool, min(cfg.rollout_samples, len(pool)))
                best = None
                for a in picks:
                    ch, rk = self.transition(state, a)
                    score = self.static_eval(ch, mover) - (self.fuzzy_weight * rk if self.use_fuzzy else 0.0)
                    if best is None or score > best[0]:
                        best = (score, a, ch, rk)
                _, action, child, risk = best  # type: ignore[misc]
            acc_edge += self.edge_value(state, child, risk, self.player)
            state = child
            depth += 1
        if state.winner is not None:
            return self.terminal_reward(state), depth
        v = self.static_eval(state, self.player) + acc_edge
        z = max(-50.0, min(50.0, v / cfg.reward_scale))
        return 1.0 / (1.0 + math.exp(-z)), depth

    @staticmethod
    def backpropagate(node: MCTSNode, reward_root: float, root_player: int) -> None:
        while node is not None:
            node.visits += 1
            node.value_sum += reward_root if node.mover == root_player else 1.0 - reward_root
            node = node.parent

    # ------------------------------------------------------------------ decision
    def decide(self, state: GameState, time_budget: float | None = None,
               cancel: Callable[[], bool] | None = None) -> Decision:
        cfg = self.config
        clock = Clock(time_budget if time_budget is not None else cfg.time_budget, cancel)
        self._prepare(state)
        if state.to_move != self.player:
            raise ValueError("MCTSAgent asked to move for the wrong player")
        sims_before = self.simulations
        root = MCTSNode(state, None, None, 1 - state.to_move)
        self.last_root = root            # kept for inspection/tests
        iterations = 0
        rollout_depth_sum = 0
        max_tree_depth = 0
        budget_hit = False
        while iterations < cfg.iterations:
            if clock.expired():
                budget_hit = True
                break
            leaf, acc = self.select(root)
            if not leaf.is_terminal and self._untried(leaf):
                leaf = self.expand(leaf)
                acc += leaf.edge
            if leaf.is_terminal:
                reward, d = self.terminal_reward(leaf.state), 0
            else:
                reward, d = self.rollout(leaf.state, acc)
            self.backpropagate(leaf, reward, self.player)
            iterations += 1
            rollout_depth_sum += d
            max_tree_depth = max(max_tree_depth, leaf.depth)

        if root.children:
            best = max(root.children, key=lambda ch: (ch.visits, ch.mean))
            action, fallback = best.action, False
        else:
            best, action, fallback = None, self.fallback_action(state), True

        cands = []
        for ch in sorted(root.children, key=lambda c: (-c.visits, -c.mean))[:4]:
            cands.append({"action": ch.action.to_dict(), "visits": ch.visits, "value": round(ch.mean, 3),
                          "risk": round(ch.risk, 3), "selected": ch is best})
        stats = {
            "algorithm": self.algorithm,
            "iterations": iterations,
            "simulations": self.simulations - sims_before,
            "avg_rollout_depth": round(rollout_depth_sum / iterations, 3) if iterations else None,
            "max_tree_depth": max_tree_depth,
            "root_children": len(root.children),
            "selected_visits": best.visits if best else None,
            "selected_reward": round(best.mean, 4) if best else None,
            "fuzzy_risk": round(best.risk, 3) if (best and self.use_fuzzy) else None,
            "decision_time": round(clock.elapsed(), 4),
            "budget_exhausted": budget_hit,
            "cancelled": clock.cancelled,
            "seed": self.seed,
            "opponent_profile": self.opponent_model.profile() if self.use_opponent_model else None,
        }
        if fallback:
            expl = "No iteration completed within the budget; using the highest-priority legal shot."
        else:
            expl = (f"{iterations} iterations, {len(root.children)} root shots explored. "
                    f"Chose {action.tag or 'shot'}: {best.visits} visits, mean reward {best.mean:.3f}.")
        return Decision(action, self.algorithm, self.player, stats, cands, expl, fallback)
