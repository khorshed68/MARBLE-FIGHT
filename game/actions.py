"""Shot actions: representation, validation and candidate generation.

An :class:`Action` is the only way to change a game state. Every action — from
the human's mouse, from Minimax or from MCTS — goes through
:func:`validate_action` before it is simulated, so all players obey exactly the
same rules.

:func:`generate_candidate_actions` builds the discrete action set used by
*both* AI agents. It mixes

* uniform shots: ``direction_count`` directions x ``power_levels``,
* aimed shots: straight at each opponent marble, and "ghost-ball" pushes that
  aim the shooter so the target is driven radially outward toward the boundary,
* ability variants (Precision / Power / Spin) while uses remain.

Each candidate gets a cheap geometric ``priority`` (no physics) used for move
ordering, beam limits and MCTS progressive widening.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from config import RulesConfig
from game.state import ABILITY_KEYS, GameState

ABILITIES = ("none", "precision", "power", "spin_left", "spin_right")
ABILITY_TO_KEY = {"precision": "precision", "power": "power", "spin_left": "spin", "spin_right": "spin"}
ABILITY_LABELS = {
    "none": "Normal",
    "precision": "Precision (stop) shot",
    "power": "Power shot",
    "spin_left": "Spin shot (curve left)",
    "spin_right": "Spin shot (curve right)",
}


class IllegalActionError(ValueError):
    """Raised when an action violates the rules."""


@dataclass(frozen=True)
class Action:
    marble_id: int
    angle: float                    # radians, world frame (0 = +x, counter-clockwise)
    power: float                    # normalised launch power in [power_min, power_max]
    ability: str = "none"
    tag: str = field(default="", compare=False)          # generator label for explanations
    priority: float = field(default=0.0, compare=False)  # cheap ordering score

    def describe(self) -> str:
        deg = math.degrees(self.angle) % 360.0
        return f"marble #{self.marble_id} @ {deg:5.1f}° power {self.power:.2f} [{ABILITY_LABELS.get(self.ability, self.ability)}]"

    def to_dict(self) -> dict:
        return {"marble_id": self.marble_id, "angle": self.angle, "power": self.power,
                "ability": self.ability, "tag": self.tag}

    @staticmethod
    def from_dict(d: dict) -> "Action":
        return Action(int(d["marble_id"]), float(d["angle"]), float(d["power"]), d.get("ability", "none"), d.get("tag", ""))


def validate_action(state: GameState, action: Action, rules: RulesConfig, player: int | None = None) -> None:
    """Raise :class:`IllegalActionError` if ``action`` is not legal now."""
    if state.is_terminal:
        raise IllegalActionError("The match is already over.")
    mover = state.to_move
    if player is not None and player != mover:
        raise IllegalActionError(f"It is not player {player + 1}'s turn.")
    if not isinstance(action.marble_id, int) or not (0 <= action.marble_id < len(state.marbles)):
        raise IllegalActionError(f"Marble index {action.marble_id!r} does not exist.")
    m = state.marbles[action.marble_id]
    if m.owner != mover:
        raise IllegalActionError("You can only shoot your own marble.")
    if not m.alive:
        raise IllegalActionError("That marble has been eliminated.")
    if not (isinstance(action.angle, (int, float)) and math.isfinite(action.angle)):
        raise IllegalActionError("Shot direction must be a finite angle.")
    if not (isinstance(action.power, (int, float)) and math.isfinite(action.power)):
        raise IllegalActionError("Shot power must be a finite number.")
    if not (rules.power_min - 1e-9 <= action.power <= rules.power_max + 1e-9):
        raise IllegalActionError(f"Power {action.power:.3f} outside [{rules.power_min}, {rules.power_max}].")
    if action.ability not in ABILITIES:
        raise IllegalActionError(f"Unknown ability '{action.ability}'.")
    if action.ability != "none":
        key = ABILITY_TO_KEY[action.ability]
        if state.ability_count(mover, key) <= 0:
            raise IllegalActionError(f"No {key} shots left.")


def is_legal(state: GameState, action: Action, rules: RulesConfig) -> bool:
    try:
        validate_action(state, action, rules)
        return True
    except IllegalActionError:
        return False


def _norm_angle(a: float) -> float:
    return a % (2.0 * math.pi)


def _ang_diff(a: float, b: float) -> float:
    d = (a - b) % (2.0 * math.pi)
    return min(d, 2.0 * math.pi - d)


def _blocked(state: GameState, sx: float, sy: float, tx: float, ty: float, ignore: tuple[int, ...], radius: float) -> bool:
    """True if another live marble sits on the straight path S->T."""
    dx, dy = tx - sx, ty - sy
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return False
    for m in state.marbles:
        if not m.alive or m.mid in ignore:
            continue
        u = ((m.x - sx) * dx + (m.y - sy) * dy) / L2
        if 0.0 < u < 1.0:
            px, py = sx + u * dx, sy + u * dy
            if (m.x - px) ** 2 + (m.y - py) ** 2 < (2 * radius) ** 2:
                return True
    return False


def generate_candidate_actions(
    state: GameState,
    rules: RulesConfig,
    arena_radius: float = 10.0,
    marble_radius: float = 0.45,
    include_abilities: bool = True,
    player: int | None = None,
) -> list[Action]:
    """Return legal candidate shots for the player to move, best-priority first.

    The list is deterministic for a given state. Priorities are geometric
    estimates only; the agents always confirm them with the physics engine.
    """
    if state.is_terminal:
        return []
    p = state.to_move if player is None else player
    own = state.alive(p)
    opp = state.alive(1 - p)
    powers = rules.power_levels
    lo, mid, hi = powers[0], powers[len(powers) // 2], powers[-1]
    can = {k: state.ability_count(p, k) > 0 for k in ABILITY_KEYS} if include_abilities else {k: False for k in ABILITY_KEYS}
    out: list[Action] = []
    seen: set[tuple[int, int, int, str]] = set()

    def add(mid_: int, ang: float, pw: float, ab: str, tag: str, pri: float) -> None:
        ang = _norm_angle(ang)
        k = (mid_, int(round(math.degrees(ang) * 2)), int(round(pw * 100)), ab)  # 0.5 degree dedupe
        if k in seen:
            return
        seen.add(k)
        out.append(Action(mid_, ang, pw, ab, tag, pri))

    for s in own:
        s_edge = arena_radius - s.dist_center()
        for t in opp:
            dx, dy = t.x - s.x, t.y - s.y
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                continue
            t_r = t.dist_center()
            t_edge = arena_radius - t_r
            danger = max(0.0, 1.0 - t_edge / arena_radius)  # 0 centre .. 1 boundary
            blocked = _blocked(state, s.x, s.y, t.x, t.y, (s.mid, t.mid), marble_radius)
            direct = math.atan2(dy, dx)
            base = 40.0 + 40.0 * danger - 2.0 * dist - (25.0 if blocked else 0.0)
            # straight hit: drives the target along the line of centres
            outward_dot = (dx * t.x + dy * t.y) / (dist * max(t_r, 1e-6)) if t_r > 1e-6 else 0.0
            add(s.mid, direct, hi, "none", f"hit#{t.mid}", base + 10.0 * outward_dot)
            add(s.mid, direct, mid, "none", f"hit#{t.mid}", base + 6.0 * outward_dot - 4.0)
            # ghost-ball push toward the nearest boundary point
            if t_r > 1e-6:
                ux, uy = t.x / t_r, t.y / t_r
                gx, gy = t.x - ux * 2 * marble_radius, t.y - uy * 2 * marble_radius
                gdx, gdy = gx - s.x, gy - s.y
                gdist = math.hypot(gdx, gdy)
                if gdist > 1e-6:
                    gang = math.atan2(gdy, gdx)
                    cut = _ang_diff(gang, math.atan2(uy, ux))
                    if cut < math.radians(75):
                        gblocked = _blocked(state, s.x, s.y, gx, gy, (s.mid, t.mid), marble_radius)
                        pri = base + 18.0 * math.cos(cut) + 8.0 - (20.0 if gblocked else 0.0)
                        add(s.mid, gang, hi, "none", f"push#{t.mid}", pri)
                        add(s.mid, gang, mid, "none", f"push#{t.mid}", pri - 3.0)
                        if can["power"]:
                            add(s.mid, gang, hi, "power", f"push#{t.mid}", pri - 6.0 + 12.0 * danger)
                        if can["precision"]:
                            add(s.mid, gang, hi, "precision", f"push#{t.mid}", pri - 2.0 + 10.0 * max(0.0, 1.0 - s_edge / 4.0))
            if can["power"]:
                add(s.mid, direct, hi, "power", f"hit#{t.mid}", base - 8.0 + 12.0 * danger)
            if can["precision"]:
                add(s.mid, direct, hi, "precision", f"hit#{t.mid}", base - 5.0)
            if blocked and can["spin"]:
                # curve around a blocker: start offset to one side and let spin bend it back
                for sign, ab in ((1.0, "spin_right"), (-1.0, "spin_left")):
                    add(s.mid, direct + sign * math.radians(14), hi, ab, f"curve#{t.mid}", base - 6.0)

        # uniform directions: repositioning and exploratory shots
        n = rules.direction_count
        s_r = s.dist_center()
        for k in range(n):
            ang = 2.0 * math.pi * k / n
            # prefer directions heading toward the centre when the marble is exposed
            inward = 0.0
            if s_r > 1e-6:
                inward = -(math.cos(ang) * s.x + math.sin(ang) * s.y) / s_r
            for pw in powers:
                risk = (1.0 if inward < 0 else 0.4) * pw * max(0.0, 1.0 - s_edge / arena_radius)
                pri = 6.0 * inward * (1.0 - s_edge / arena_radius) - 10.0 * risk - 3.0 * pw
                add(s.mid, ang, pw, "none", f"dir{k}", pri)
            if can["spin"] and k % 2 == 0:
                add(s.mid, ang, hi, "spin_left", f"dir{k}", -12.0)
                add(s.mid, ang, hi, "spin_right", f"dir{k}", -12.0)
        if s_r > 1e-6:
            # gentle retreat toward the centre (defensive)
            ang = math.atan2(-s.y, -s.x)
            add(s.mid, ang, lo, "none", "retreat", 4.0 + 10.0 * (1.0 - s_edge / arena_radius))

    # stable ordering: priority desc, then deterministic tiebreak
    out.sort(key=lambda a: (-a.priority, a.marble_id, a.ability, round(a.angle, 6), a.power))
    return out
