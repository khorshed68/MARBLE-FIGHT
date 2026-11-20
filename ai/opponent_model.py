"""History-Based Adaptive AI: an interpretable model of the opponent's habits.

Only *executed* opponent shots are observed (never future or hidden actions).
From them the model derives a few counted statistics and turns them into small,
explainable adjustments of the evaluation function (:class:`EvalBias`):

* ``defense_mult`` – more weight on keeping our marbles away from the edge when
  the opponent plays aggressively (shots aimed at our marbles / high power near
  the boundary).
* ``threat_mult`` – opponents that shoot hard project their threats further.
* ``protect`` – extra protection for the marble the opponent targets most.

With an empty history every multiplier is exactly 1.0 (the default policy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from game.state import GameState

SECTORS = 8  # preferred-direction histogram resolution


@dataclass
class EvalBias:
    defense_mult: float = 1.0
    threat_mult: float = 1.0
    protect: dict[int, float] = field(default_factory=dict)   # marble id -> extra danger weight

    @staticmethod
    def neutral() -> "EvalBias":
        return EvalBias()


@dataclass
class OpponentModel:
    observer: int                                     # the player who owns this model
    arena_radius: float = 10.0
    shots: int = 0
    aggressive: int = 0
    defensive: int = 0
    edge_power_shots: int = 0                         # high power shots taken from near the boundary
    power_sum: float = 0.0
    knockouts: int = 0
    sector_counts: list[int] = field(default_factory=lambda: [0] * SECTORS)
    targeted: dict[int, int] = field(default_factory=dict)
    category_attempts: dict[str, int] = field(default_factory=dict)
    category_success: dict[str, int] = field(default_factory=dict)
    last_categories: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.__init__(self.observer, self.arena_radius)  # type: ignore[misc]

    # ------------------------------------------------------------------ observation
    def classify(self, before: GameState, marble_id: int, angle: float, power: float) -> tuple[str, int]:
        """Return (category, targeted marble id or -1) for an opponent shot."""
        s = before.marbles[marble_id]
        dx, dy = math.cos(angle), math.sin(angle)
        best, target = 0.0, -1
        for m in before.marbles:
            if not m.alive or m.owner != self.observer:
                continue
            vx, vy = m.x - s.x, m.y - s.y
            d = math.hypot(vx, vy)
            if d < 1e-6:
                continue
            cos = (vx * dx + vy * dy) / d
            # within ~ the angle a marble subtends (plus slack for cut shots)
            if cos > math.cos(math.radians(22)) and cos > best:
                best, target = cos, m.mid
        r0 = math.hypot(s.x, s.y)
        inward = -(dx * s.x + dy * s.y) / r0 if r0 > 1e-6 else 0.0
        if target >= 0:
            return ("aggressive_power" if power >= 0.8 else "aggressive_touch"), target
        if inward > 0.3:
            return "defensive", -1
        return "reposition", -1

    def observe(self, before: GameState, marble_id: int, angle: float, power: float, after: GameState) -> None:
        if before.to_move == self.observer:
            raise ValueError("OpponentModel only observes the opponent's shots")
        cat, target = self.classify(before, marble_id, angle, power)
        self.shots += 1
        self.power_sum += power
        if cat.startswith("aggressive"):
            self.aggressive += 1
            self.targeted[target] = self.targeted.get(target, 0) + 1
        elif cat == "defensive":
            self.defensive += 1
        s = before.marbles[marble_id]
        if self.arena_radius - math.hypot(s.x, s.y) < 3.0 and power >= 0.8:
            self.edge_power_shots += 1
        self.sector_counts[int(((angle % (2 * math.pi)) / (2 * math.pi)) * SECTORS) % SECTORS] += 1
        lost = sum(1 for a, b in zip(before.marbles, after.marbles) if a.alive and not b.alive and a.owner == self.observer)
        self.knockouts += lost
        self.category_attempts[cat] = self.category_attempts.get(cat, 0) + 1
        if lost:
            self.category_success[cat] = self.category_success.get(cat, 0) + 1
        self.last_categories = (self.last_categories + [cat])[-6:]

    # ------------------------------------------------------------------ profile
    @property
    def aggression(self) -> float:
        """Laplace-smoothed share of aggressive shots (0.5 with no data)."""
        return (self.aggressive + 1) / (self.shots + 2)

    @property
    def avg_power(self) -> float:
        return (self.power_sum + 0.65) / (self.shots + 1)

    def repeated_pattern(self) -> str | None:
        if len(self.last_categories) >= 3 and len(set(self.last_categories[-3:])) == 1:
            return self.last_categories[-1]
        return None

    def bias(self) -> EvalBias:
        if self.shots == 0:
            return EvalBias.neutral()
        a = self.aggression
        edge_rate = (self.edge_power_shots + 0.5) / (self.shots + 1)
        defense = 1.0 + 0.6 * (a - 0.5) * 2.0 + 0.3 * (edge_rate - 0.5)
        threat = 1.0 + 0.6 * (self.avg_power - 0.65)
        protect: dict[int, float] = {}
        total_t = sum(self.targeted.values())
        if total_t >= 2:
            for mid, c in self.targeted.items():
                protect[mid] = 0.6 * c / total_t
        return EvalBias(round(max(0.4, min(1.8, defense)), 4), round(max(0.6, min(1.5, threat)), 4), protect)

    def profile(self) -> dict:
        b = self.bias()
        return {
            "shots_observed": self.shots,
            "aggression": round(self.aggression, 3),
            "defensive_rate": round(self.defensive / self.shots, 3) if self.shots else None,
            "avg_power": round(self.avg_power, 3),
            "edge_power_shots": self.edge_power_shots,
            "knockouts_suffered": self.knockouts,
            "preferred_sector": (max(range(SECTORS), key=lambda i: self.sector_counts[i]) if self.shots else None),
            "most_targeted_marble": (max(self.targeted, key=self.targeted.get) if self.targeted else None),
            "category_success": {k: f"{self.category_success.get(k, 0)}/{v}" for k, v in self.category_attempts.items()},
            "repeated_pattern": self.repeated_pattern(),
            "defense_mult": b.defense_mult,
            "threat_mult": b.threat_mult,
        }

    # ------------------------------------------------------------------ transfer (worker processes)
    def snapshot(self) -> dict:
        return {
            "observer": self.observer, "arena_radius": self.arena_radius, "shots": self.shots, "aggressive": self.aggressive,
            "defensive": self.defensive, "edge_power_shots": self.edge_power_shots,
            "power_sum": self.power_sum, "knockouts": self.knockouts,
            "sector_counts": list(self.sector_counts), "targeted": dict(self.targeted),
            "category_attempts": dict(self.category_attempts), "category_success": dict(self.category_success),
            "last_categories": list(self.last_categories),
        }

    @staticmethod
    def from_snapshot(d: dict) -> "OpponentModel":
        m = OpponentModel(int(d["observer"]))
        for k, v in d.items():
            if k == "targeted":
                v = {int(a): int(b) for a, b in v.items()}
            setattr(m, k, v)
        return m
