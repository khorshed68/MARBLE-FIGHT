"""Match rules: initial placement, terminal conditions and turn-limit scoring."""

from __future__ import annotations

import math

from config import GameConfig, DEFAULT_CONFIG
from game.arena import Arena
from game.state import ABILITY_KEYS, DRAW, GameState, MarbleState


class SetupError(ValueError):
    pass


def starting_positions(arena: Arena, cfg: GameConfig, per_player: int) -> list[tuple[int, float, float]]:
    """Legal, non-overlapping start spots. Player 2's spots are Player 1's
    rotated by 180 degrees, so the layout is point-symmetric (fair)."""
    rules, phys = cfg.rules, cfg.physics
    if per_player < 1 or per_player > 7:
        raise SetupError("marbles_per_player must be between 1 and 7")
    spots: list[tuple[float, float]] = []
    ring = rules.start_radius
    spread = math.radians(rules.start_spread_deg)
    # first ring around the player's home angle (-90 deg), then an inner ring
    ring_slots = min(per_player, 4)
    for k in range(ring_slots):
        off = (k - (ring_slots - 1) / 2.0) * spread
        a = -math.pi / 2 + off
        spots.append((ring * math.cos(a), ring * math.sin(a)))
    inner = per_player - ring_slots
    for k in range(inner):
        off = (k - (inner - 1) / 2.0) * spread * 1.3
        a = -math.pi / 2 + off
        spots.append(((ring - 2.2) * math.cos(a), (ring - 2.2) * math.sin(a)))

    out: list[tuple[int, float, float]] = [(0, x, y) for x, y in spots] + [(1, -x, -y) for x, y in spots]
    _check_layout(out, arena, cfg)
    return out


def _check_layout(spots: list[tuple[int, float, float]], arena: Arena, cfg: GameConfig) -> None:
    r = cfg.physics.marble_radius
    for i, (_, x, y) in enumerate(spots):
        if math.hypot(x, y) + r >= arena.radius:
            raise SetupError("start spot outside the arena")
        for _, ox, oy in spots[i + 1:]:
            if math.hypot(x - ox, y - oy) < 2 * r + 1e-6:
                raise SetupError("start spots overlap")
        for ox, oy, orr in arena.obstacle_positions(0.0):
            if math.hypot(x - ox, y - oy) < r + orr + 1e-6:
                raise SetupError("start spot overlaps an obstacle")


def initial_state(
    arena: Arena,
    cfg: GameConfig = DEFAULT_CONFIG,
    per_player: int | None = None,
    first_player: int | None = None,
    turn_limit: int | None = None,
) -> GameState:
    per = cfg.rules.marbles_per_player if per_player is None else per_player
    spots = starting_positions(arena, cfg, per)
    marbles = tuple(MarbleState(i, owner, x, y, True) for i, (owner, x, y) in enumerate(spots))
    uses = dict(cfg.rules.ability_uses)
    ab = tuple(uses.get(k, 0) for k in ABILITY_KEYS)
    return GameState(
        marbles=marbles,
        to_move=cfg.rules.first_player if first_player is None else first_player,
        turn=0,
        abilities=(ab, ab),
        arena_id=arena.arena_id,
        turn_limit=cfg.rules.turn_limit if turn_limit is None else turn_limit,
    )


def turn_limit_score(state: GameState, arena: Arena) -> tuple[int, float]:
    """Deterministic score used when the turn limit is reached.

    Primary: surviving marbles. Secondary: total distance of surviving marbles
    from the boundary (higher = safer position), rounded to 1e-6 so float noise
    cannot decide a match.
    """
    out = []
    for p in (0, 1):
        alive = state.alive(p)
        safety = sum(max(0.0, arena.radius - m.dist_center()) for m in alive)
        out.append((len(alive), round(safety, 6)))
    return out[0], out[1]  # type: ignore[return-value]


def resolve_terminal(state: GameState, arena: Arena) -> tuple[int | None, str]:
    """Return (winner, reason) for a state reached after a resolved shot."""
    c0, c1 = state.count(0), state.count(1)
    if c0 == 0 and c1 == 0:
        return DRAW, "Both sides lost their last marble on the same shot"
    if c1 == 0:
        return 0, "All Player 2 marbles eliminated"
    if c0 == 0:
        return 1, "All Player 1 marbles eliminated"
    if state.turn >= state.turn_limit:
        s0, s1 = turn_limit_score(state, arena)
        if s0 > s1:
            return 0, f"Turn limit: Player 1 wins on score {s0[0]} marbles / {s0[1]:.2f} safety vs {s1[0]} / {s1[1]:.2f}"
        if s1 > s0:
            return 1, f"Turn limit: Player 2 wins on score {s1[0]} marbles / {s1[1]:.2f} safety vs {s0[0]} / {s0[1]:.2f}"
        return DRAW, "Turn limit reached with an exactly tied score"
    return None, ""
