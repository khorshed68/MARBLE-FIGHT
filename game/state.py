"""Immutable game-state representation shared by rules, physics, AI and UI.

Between turns every marble is at rest, so a state only needs positions; the
velocities exist only inside one physics simulation. States are frozen
dataclasses: every transition returns a *new* state, which is what lets the
AI search explore futures without touching the live match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

PLAYER_NAMES = ("Player 1", "Player 2")
ABILITY_KEYS = ("precision", "power", "spin")
DRAW = -1


@dataclass(frozen=True)
class MarbleState:
    mid: int          # stable marble id (index into GameState.marbles)
    owner: int        # 0 or 1
    x: float
    y: float
    alive: bool = True

    def dist_center(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass(frozen=True)
class ShotRecord:
    """Summary of the most recent shot (used for explanations and fuzzy risk)."""

    player: int
    marble_id: int
    angle: float
    power: float
    ability: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    eliminated: tuple[int, ...] = ()
    first_hit: int = -1


@dataclass(frozen=True)
class GameState:
    marbles: tuple[MarbleState, ...]
    to_move: int
    turn: int                                         # number of completed shots
    abilities: tuple[tuple[int, ...], tuple[int, ...]]  # remaining uses per player, ordered as ABILITY_KEYS
    arena_id: str
    turn_limit: int
    clock: float = 0.0                                 # seconds of simulated motion so far (drives rotating bumpers)
    winner: int | None = None                          # 0/1, DRAW (-1) or None while running
    end_reason: str = ""
    last_shot: ShotRecord | None = field(default=None, compare=False)

    # ------------------------------------------------------------------ queries
    @property
    def is_terminal(self) -> bool:
        return self.winner is not None

    def alive(self, player: int) -> list[MarbleState]:
        return [m for m in self.marbles if m.alive and m.owner == player]

    def count(self, player: int) -> int:
        return sum(1 for m in self.marbles if m.alive and m.owner == player)

    def ability_count(self, player: int, key: str) -> int:
        return self.abilities[player][ABILITY_KEYS.index(key)]

    def marble(self, mid: int) -> MarbleState:
        return self.marbles[mid]

    def key(self, precision: int = 3) -> tuple:
        """Hashable, rounded identity of the state (for caches/tests)."""
        return (
            self.to_move, self.turn, self.abilities, round(self.clock, precision),
            tuple((m.alive, round(m.x, precision), round(m.y, precision)) for m in self.marbles),
        )

    def fingerprint(self, precision: int = 6) -> str:
        """Compact identity of the state used to verify replays (rounded to 1e-6)."""
        k = self.key(precision)
        return f"{k}|w={self.winner}"

    def with_changes(self, **kw) -> "GameState":
        return replace(self, **kw)

    # ------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        return {
            "marbles": [[m.mid, m.owner, m.x, m.y, m.alive] for m in self.marbles],
            "to_move": self.to_move,
            "turn": self.turn,
            "abilities": [list(a) for a in self.abilities],
            "arena_id": self.arena_id,
            "turn_limit": self.turn_limit,
            "clock": self.clock,
            "winner": self.winner,
            "end_reason": self.end_reason,
        }

    @staticmethod
    def from_dict(d: dict) -> "GameState":
        return GameState(
            marbles=tuple(MarbleState(int(a), int(b), float(c), float(e), bool(f)) for a, b, c, e, f in d["marbles"]),
            to_move=int(d["to_move"]),
            turn=int(d["turn"]),
            abilities=tuple(tuple(int(x) for x in a) for a in d["abilities"]),  # type: ignore[arg-type]
            arena_id=d["arena_id"],
            turn_limit=int(d["turn_limit"]),
            clock=float(d.get("clock", 0.0)),
            winner=d.get("winner"),
            end_reason=d.get("end_reason", ""),
        )
