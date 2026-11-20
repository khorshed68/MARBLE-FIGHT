"""The single authoritative transition function of the game.

``simulate_turn(state, action) -> TurnResult`` is used by the live match, by the
trajectory preview, by Minimax and by MCTS. Nothing else is allowed to move
marbles, so the AI's predictions and the real game can never disagree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from config import DEFAULT_CONFIG, GameConfig
from game.actions import ABILITY_TO_KEY, Action, validate_action
from game.arena import Arena, get_arena
from game.physics import Contact, run_physics
from game.rules import resolve_terminal
from game.state import ABILITY_KEYS, GameState, MarbleState, ShotRecord


@dataclass
class TurnResult:
    state: GameState                       # the new state (input state is untouched)
    eliminated: tuple[int, ...]            # marble ids eliminated by this shot, in order
    sim_time: float
    frames_simulated: int
    truncated: bool                        # True if the max-duration safety cap was hit
    marble_contacts: int
    obstacle_contacts: int
    first_hit: int                         # marble id the shooter touched first (-1 none)
    contacts: list[Contact] = field(default_factory=list)
    frames: list | None = None             # recorded positions for rendering playback
    elimination_times: tuple[float, ...] = ()


def launch_velocity(action: Action, cfg: GameConfig) -> tuple[float, float, float, bool]:
    """Physics effect of an action: (vx, vy, spin, precision)."""
    speed = action.power * cfg.physics.shot_speed_max
    if action.ability == "power":
        speed *= cfg.physics.power_shot_multiplier
    spin = 0.0
    if action.ability == "spin_left":
        spin = 1.0
    elif action.ability == "spin_right":
        spin = -1.0
    return speed * math.cos(action.angle), speed * math.sin(action.angle), spin, action.ability == "precision"


def simulate_turn(
    state: GameState,
    action: Action,
    arena: Arena | None = None,
    cfg: GameConfig = DEFAULT_CONFIG,
    record: bool = False,
    validate: bool = True,
    record_contacts: bool | None = None,
) -> TurnResult:
    """Apply ``action`` for ``state.to_move`` and resolve the shot completely."""
    if arena is None:
        arena = get_arena(state.arena_id)
    if validate:
        validate_action(state, action, cfg.rules)
    marbles = state.marbles
    xs = [m.x for m in marbles]
    ys = [m.y for m in marbles]
    alive = [m.alive for m in marbles]
    vx, vy, spin, precision = launch_velocity(action, cfg)
    res = run_physics(
        xs, ys, alive, action.marble_id, vx, vy, arena, cfg.physics,
        clock=state.clock, spin=spin, precision=precision, record=record,
        record_contacts=record if record_contacts is None else record_contacts,
    )

    # eliminated marbles keep the spot where they crossed the boundary (dead marbles never move)
    new_marbles = tuple(
        MarbleState(m.mid, m.owner, res.xs[i], res.ys[i], res.alive[i]) for i, m in enumerate(marbles)
    )
    abilities = state.abilities
    if action.ability != "none":
        p = state.to_move
        idx = ABILITY_KEYS.index(ABILITY_TO_KEY[action.ability])
        row = list(abilities[p])
        row[idx] -= 1
        abilities = (tuple(row), abilities[1]) if p == 0 else (abilities[0], tuple(row))  # type: ignore[assignment]

    eliminated = tuple(i for i, _ in res.eliminations)
    shooter = marbles[action.marble_id]
    shot = ShotRecord(
        player=state.to_move, marble_id=action.marble_id, angle=action.angle, power=action.power,
        ability=action.ability, start_x=shooter.x, start_y=shooter.y,
        end_x=res.xs[action.marble_id], end_y=res.ys[action.marble_id],
        eliminated=eliminated, first_hit=res.shooter_first_hit,
    )
    nxt = GameState(
        marbles=new_marbles,
        to_move=1 - state.to_move,
        turn=state.turn + 1,
        abilities=abilities,  # type: ignore[arg-type]
        arena_id=state.arena_id,
        turn_limit=state.turn_limit,
        clock=state.clock + res.sim_time,
        last_shot=shot,
    )
    winner, reason = resolve_terminal(nxt, arena)
    if winner is not None:
        nxt = nxt.with_changes(winner=winner, end_reason=reason)
    return TurnResult(
        state=nxt, eliminated=eliminated, sim_time=res.sim_time, frames_simulated=res.frames_simulated,
        truncated=res.truncated, marble_contacts=res.marble_contacts, obstacle_contacts=res.obstacle_contacts,
        first_hit=res.shooter_first_hit, contacts=res.contacts, frames=res.frames,
        elimination_times=tuple(t for _, t in res.eliminations),
    )
