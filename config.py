"""Central configuration for Marble Fight.

Every gameplay constant used by the physics engine, the rules, the AI agents and
the user interface lives here (arena layouts live in ``data/arenas.json``).
Changing a value here changes it consistently for the live match, the AI's
internal simulations and the tests, because all of them read from this module.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
ARENAS_FILE = os.path.join(DATA_DIR, "arenas.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
MATCH_HISTORY_FILE = os.path.join(DATA_DIR, "match_history.jsonl")
TOURNAMENT_DIR = os.path.join(DATA_DIR, "tournaments")

GAME_TITLE = "MARBLE FIGHT"
GAME_SUBTITLE = "Intelligent Marble Battle Arena"
GAME_TAGLINE = "Aim Smart. Strike Hard. Own the Arena."


@dataclass(frozen=True)
class PhysicsConfig:
    """Constants of the deterministic 2-D tabletop physics model.

    Units: 1 world unit is roughly 1 cm of a real Kancha court scaled up; time is
    in seconds. The arena radius is defined per arena in ``data/arenas.json``.
    """

    dt: float = 1.0 / 60.0                 # fixed simulation timestep (s)
    marble_radius: float = 0.45
    marble_mass: float = 1.0
    restitution_marble: float = 0.90       # coefficient of restitution, marble-marble
    restitution_obstacle: float = 0.80     # marble-bumper
    rolling_decel: float = 3.2             # constant rolling-friction deceleration (u/s^2)
    linear_drag: float = 0.15              # velocity-proportional drag (1/s)
    stop_speed: float = 0.06               # below this speed (and no active force) a marble sleeps
    settle_frames: int = 3                 # consecutive all-asleep frames required to settle
    max_sim_time: float = 14.0             # hard cap on one shot's simulation (s)
    max_substep_travel: float = 0.18       # max distance a marble may travel per sub-step
    max_speed: float = 25.0                # numerical safety clamp (u/s)
    elimination_tolerance: float = 0.0     # marble eliminated when |centre| > R + tolerance
    shot_speed_max: float = 10.5           # launch speed for power = 1.0 (u/s); balance-tested, see AI_ALGORITHMS.md
    # Ability physics
    power_shot_multiplier: float = 1.35    # Power Shot launch-speed multiplier
    precision_brake_multiplier: float = 4.0  # shooter friction multiplier after its first hit
    spin_curve: float = 0.9                # lateral acceleration factor (1/s) for Spin Shot
    spin_decay_time: float = 1.2           # spin e-folding time (s)


@dataclass(frozen=True)
class RulesConfig:
    """Match rules."""

    marbles_per_player: int = 3
    turn_limit: int = 60                   # total shots (both players) before scoring
    first_player: int = 0
    power_min: float = 0.15
    power_max: float = 1.0
    power_levels: tuple[float, ...] = (0.35, 0.65, 1.0)   # low, medium, high
    direction_count: int = 16
    ability_uses: tuple[tuple[str, int], ...] = (
        ("precision", 2),
        ("power", 2),
        ("spin", 2),
    )
    start_radius: float = 5.0              # radius of the ring the starting marbles sit on
    start_spread_deg: float = 25.0         # angular spacing between a player's start marbles


@dataclass(frozen=True)
class MinimaxConfig:
    """Budget and options of the Minimax agent."""

    max_depth: int = 2                     # plies; one ply = one complete turn
    root_width: int = 36                   # candidate shots simulated at the root
    inner_width: int = 12                  # candidate shots per inner node
    time_budget: float = 4.0               # seconds (safety cap; search is iterative deepening)
    node_budget: int = 4000                # max simulated nodes per decision
    use_fuzzy: bool = True
    use_opponent_model: bool = True
    fuzzy_weight: float = 30.0


@dataclass(frozen=True)
class MCTSConfig:
    """Budget and options of the MCTS agent."""

    iterations: int = 220                  # simulation budget (tree iterations)
    time_budget: float = 4.0               # seconds (safety cap)
    exploration: float = 1.0 / (2 ** 0.5)  # UCT exploration constant c
    rollout_depth: int = 2                 # turns simulated by the rollout policy
    rollout_samples: int = 3               # candidate shots compared per rollout step
    rollout_epsilon: float = 0.2           # probability of a random (unscored) rollout shot
    widening_c: float = 2.0                # progressive widening: k = ceil(c * N^alpha)
    widening_alpha: float = 0.5
    candidate_width: int = 48              # candidate shots considered at a node
    reward_scale: float = 160.0            # heuristic -> [0,1] logistic scale
    use_fuzzy: bool = True
    use_opponent_model: bool = True
    fuzzy_weight: float = 30.0


@dataclass(frozen=True)
class GameConfig:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)


DEFAULT_CONFIG = GameConfig()

# Difficulty presets change the real search budget, not just a label.
DIFFICULTY_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "easy": {
        "minimax": {"max_depth": 2, "root_width": 12, "inner_width": 6, "node_budget": 400, "time_budget": 1.5},
        "mcts": {"iterations": 40, "rollout_depth": 1, "candidate_width": 20, "time_budget": 1.5},
    },
    "normal": {
        "minimax": {"max_depth": 3, "root_width": 24, "inner_width": 10, "node_budget": 3000, "time_budget": 3.0},
        "mcts": {"iterations": 160, "rollout_depth": 2, "time_budget": 3.0},
    },
    "hard": {
        "minimax": {"max_depth": 3, "root_width": 36, "inner_width": 14, "node_budget": 7000, "time_budget": 8.0},
        "mcts": {"iterations": 500, "rollout_depth": 2, "candidate_width": 64, "time_budget": 8.0},
    },
}


def minimax_config_for(difficulty: str = "normal", **overrides: Any) -> MinimaxConfig:
    base = DIFFICULTY_PRESETS.get(difficulty, DIFFICULTY_PRESETS["normal"])["minimax"]
    return replace(MinimaxConfig(), **{**base, **overrides})


def mcts_config_for(difficulty: str = "normal", **overrides: Any) -> MCTSConfig:
    base = DIFFICULTY_PRESETS.get(difficulty, DIFFICULTY_PRESETS["normal"])["mcts"]
    return replace(MCTSConfig(), **{**base, **overrides})


def load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


DEFAULT_SETTINGS: dict[str, Any] = {
    # gameplay defaults
    "arena": "classic",
    "difficulty": "normal",
    "marbles_per_player": 3,
    "use_fuzzy": True,
    "use_opponent_model": True,
    "rounds": 1,                    # 1 = single match; 2 = two rounds with the first shooter swapped (fairer)
    # AI visualisation and animation
    "show_analysis": True,          # AI decision-visualisation overlay (stats panel + candidate lines)
    "show_trajectories": True,      # trajectory previews (human aiming and AI candidates)
    "anim_speed": 1.0,              # shot playback speed in Human-vs-AI / Human-vs-Human
    "ai_playback_speed": 1.0,       # default playback speed for AI-vs-AI matches and replays
    # graphics
    "graphics_quality": "high",     # low | medium | high
    "effects": "full",              # off | reduced | full
    "show_fps": False,
    # audio
    "master_volume": 0.8,
    "music_volume": 0.6,
    "sfx_volume": 0.9,
    "muted": False,
}


def load_settings() -> dict[str, Any]:
    data = load_json(SETTINGS_FILE, {})
    out = dict(DEFAULT_SETTINGS)
    if isinstance(data, dict):
        # migrate keys from earlier versions
        if "sound" in data and "muted" not in data:
            data["muted"] = not bool(data["sound"])
        if "playback_speed" in data and "ai_playback_speed" not in data:
            data["ai_playback_speed"] = data["playback_speed"]
        out.update({k: v for k, v in data.items() if k in DEFAULT_SETTINGS})
    return out


def save_settings(settings: dict[str, Any]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)
