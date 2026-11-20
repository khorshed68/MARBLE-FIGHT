"""Construct agents from a short spec (used by the UI, tournaments and workers)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ai.base_agent import BaseAgent
from ai.mcts import MCTSAgent
from ai.minimax import MinimaxAgent
from config import DEFAULT_CONFIG, GameConfig, mcts_config_for, minimax_config_for

AGENT_KINDS = ("minimax", "mcts")
AGENT_LABELS = {"minimax": "Minimax + Alpha-Beta", "mcts": "MCTS (UCT)"}


def make_agent(kind: str, player: int, difficulty: str = "normal", seed: int = 0,
               use_fuzzy: bool = True, use_opponent_model: bool = True,
               game_cfg: GameConfig = DEFAULT_CONFIG, overrides: dict[str, Any] | None = None) -> BaseAgent:
    overrides = dict(overrides or {})
    overrides.update(use_fuzzy=use_fuzzy, use_opponent_model=use_opponent_model)
    if kind == "minimax":
        return MinimaxAgent(player, minimax_config_for(difficulty, **overrides), game_cfg)
    if kind == "mcts":
        return MCTSAgent(player, mcts_config_for(difficulty, **overrides), game_cfg, seed=seed)
    raise ValueError(f"Unknown agent kind '{kind}' (use one of {AGENT_KINDS})")


def agent_spec(kind: str, difficulty: str = "normal", use_fuzzy: bool = True, use_opponent_model: bool = True,
               overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """A plain-dict description that can be sent to a worker process."""
    return {"kind": kind, "difficulty": difficulty, "use_fuzzy": use_fuzzy,
            "use_opponent_model": use_opponent_model, "overrides": dict(overrides or {})}


def agent_from_spec(spec: dict[str, Any], player: int, seed: int = 0, game_cfg: GameConfig = DEFAULT_CONFIG) -> BaseAgent:
    return make_agent(spec["kind"], player, spec.get("difficulty", "normal"), seed,
                      spec.get("use_fuzzy", True), spec.get("use_opponent_model", True), game_cfg,
                      spec.get("overrides"))


def with_marbles(cfg: GameConfig, marbles: int) -> GameConfig:
    return replace(cfg, rules=replace(cfg.rules, marbles_per_player=marbles))
