"""Arena definitions: circular boundary, surface zones, bumpers and magnets.

Arenas are loaded from ``data/arenas.json``. Everything here is pure data plus a
few deterministic geometric queries, so the same arena object is used by the
live match, by the AI's look-ahead simulations and by the renderer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

from config import ARENAS_FILE, load_json


@dataclass(frozen=True)
class SurfaceZone:
    """Circular floor patch with its own friction multiplier."""

    kind: str
    x: float
    y: float
    r: float
    friction: float
    label: str = ""


@dataclass(frozen=True)
class Bumper:
    """Static circular obstacle."""

    x: float
    y: float
    r: float


@dataclass(frozen=True)
class RotatingBumper:
    """Circular obstacle orbiting the arena centre.

    Its position is a pure function of the arena clock ``t`` (seconds of
    simulated motion since the start of the match), so AI look-ahead can
    reproduce it exactly: angle = phase + omega * t.
    """

    orbit: float
    phase: float
    omega: float
    r: float

    def position(self, t: float) -> tuple[float, float]:
        a = self.phase + self.omega * t
        return self.orbit * math.cos(a), self.orbit * math.sin(a)

    def velocity(self, t: float) -> tuple[float, float]:
        a = self.phase + self.omega * t
        s = self.orbit * self.omega
        return -s * math.sin(a), s * math.cos(a)


@dataclass(frozen=True)
class Magnet:
    """Magnetic vortex: pulls *rolling* marbles toward its core.

    The pull scales with min(1, speed), so it fades as a marble slows down and
    can never keep a marble moving forever (rolling friction dominates once
    speed < rolling_decel / strength).
    """

    x: float
    y: float
    r: float
    strength: float


@dataclass(frozen=True)
class Arena:
    arena_id: str
    name: str
    theme: str
    radius: float
    base_friction: float
    zones: tuple[SurfaceZone, ...] = ()
    bumpers: tuple[Bumper, ...] = ()
    rotating_bumpers: tuple[RotatingBumper, ...] = ()
    magnets: tuple[Magnet, ...] = ()
    palette: dict = field(default_factory=dict, compare=False, hash=False)

    def friction_at(self, x: float, y: float) -> float:
        """Friction multiplier at a point (first matching zone wins)."""
        for z in self.zones:
            dx = x - z.x
            dy = y - z.y
            if dx * dx + dy * dy <= z.r * z.r:
                return self.base_friction * z.friction
        return self.base_friction

    def zone_at(self, x: float, y: float) -> SurfaceZone | None:
        for z in self.zones:
            if (x - z.x) ** 2 + (y - z.y) ** 2 <= z.r * z.r:
                return z
        return None

    def obstacle_positions(self, t: float) -> list[tuple[float, float, float]]:
        """All obstacles (x, y, r) at arena clock t."""
        out = [(b.x, b.y, b.r) for b in self.bumpers]
        for rb in self.rotating_bumpers:
            px, py = rb.position(t)
            out.append((px, py, rb.r))
        return out

    @property
    def has_dynamic_obstacles(self) -> bool:
        return bool(self.rotating_bumpers)


def _arena_from_dict(arena_id: str, d: dict) -> Arena:
    return Arena(
        arena_id=arena_id,
        name=d.get("name", arena_id),
        theme=d.get("theme", ""),
        radius=float(d.get("radius", 10.0)),
        base_friction=float(d.get("base_friction", 1.0)),
        zones=tuple(
            SurfaceZone(z.get("kind", "zone"), float(z["x"]), float(z["y"]), float(z["r"]),
                        float(z["friction"]), z.get("label", ""))
            for z in d.get("zones", [])
        ),
        bumpers=tuple(Bumper(float(b["x"]), float(b["y"]), float(b["r"])) for b in d.get("bumpers", [])),
        rotating_bumpers=tuple(
            RotatingBumper(float(b["orbit"]), float(b["phase"]), float(b["omega"]), float(b["r"]))
            for b in d.get("rotating_bumpers", [])
        ),
        magnets=tuple(
            Magnet(float(m["x"]), float(m["y"]), float(m["r"]), float(m["strength"]))
            for m in d.get("magnets", [])
        ),
        palette=dict(d.get("palette", {})),
    )


_FALLBACK = {"classic": {"name": "Classic Kancha Court", "radius": 10.0, "base_friction": 1.0}}


@lru_cache(maxsize=None)
def _all_arenas() -> dict[str, Arena]:
    raw = load_json(ARENAS_FILE, _FALLBACK)
    if not isinstance(raw, dict) or not raw:
        raw = _FALLBACK
    return {k: _arena_from_dict(k, v) for k, v in raw.items()}


def arena_ids() -> list[str]:
    return list(_all_arenas().keys())


def get_arena(arena_id: str) -> Arena:
    arenas = _all_arenas()
    if arena_id not in arenas:
        raise KeyError(f"Unknown arena '{arena_id}'. Available: {', '.join(arenas)}")
    return arenas[arena_id]
