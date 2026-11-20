"""Heuristic evaluation shared by Minimax and MCTS.

``evaluate(state, perspective)`` returns a score where positive is good for
``perspective``. Terminal results use ``WIN_SCORE`` (10 000), which is far
larger than any possible non-terminal score (bounded well under 2 000 here),
so a forced win always dominates and a forced loss is always avoided.

Non-terminal features (all documented in AI_ALGORITHMS.md, section 13):

==================  ==========================================================
material            120 per marble advantage
edge danger         own marbles near the rim: -40 * danger^2 (x defense bias)
                    enemy marbles near the rim: +30 * danger^2
attack lines        unblocked lines along which a marble can drive an enemy
                    marble outward (alignment x enemy danger x distance);
                    weighted by who moves next
safety margin       +1.5 per unit of average distance from the rim
abilities           +4 per remaining ability charge (own minus enemy)
==================  ==========================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ai.opponent_model import EvalBias
from game.state import DRAW, GameState

WIN_SCORE = 10_000.0


@dataclass(frozen=True)
class EvalWeights:
    material: float = 120.0
    own_edge: float = 40.0
    opp_edge: float = 30.0
    danger_band: float = 4.0
    attack_to_move: float = 26.0     # threats of the side that shoots next
    attack_waiting: float = 10.0     # threats of the side that must wait a turn
    safety: float = 1.5
    ability: float = 4.0


DEFAULT_WEIGHTS = EvalWeights()


def _edge_danger(m, radius: float, band: float) -> float:
    d_edge = radius - math.hypot(m.x, m.y)
    x = (band - d_edge) / band
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x


def _attack_potential(state: GameState, attacker: int, radius: float, marble_r: float) -> float:
    """Sum over enemy marbles of the best unblocked outward-push opportunity."""
    total = 0.0
    attackers = [m for m in state.marbles if m.alive and m.owner == attacker]
    targets = [m for m in state.marbles if m.alive and m.owner != attacker]
    others = [m for m in state.marbles if m.alive]
    for t in targets:
        tr = math.hypot(t.x, t.y)
        if tr < 1e-6:
            continue
        t_danger = max(0.0, min(1.0, (6.0 - (radius - tr)) / 6.0))
        if t_danger <= 0.0:
            continue
        ux, uy = t.x / tr, t.y / tr
        best = 0.0
        for a in attackers:
            dx, dy = t.x - a.x, t.y - a.y
            d = math.hypot(dx, dy)
            if d < 1e-6:
                continue
            align = (dx * ux + dy * uy) / d
            if align <= 0.2:
                continue
            # blocked by any third marble?
            L2 = d * d
            blocked = False
            for o in others:
                if o.mid == a.mid or o.mid == t.mid:
                    continue
                u = ((o.x - a.x) * dx + (o.y - a.y) * dy) / L2
                if 0.0 < u < 1.0:
                    px, py = a.x + u * dx, a.y + u * dy
                    if (o.x - px) ** 2 + (o.y - py) ** 2 < (2 * marble_r) ** 2:
                        blocked = True
                        break
            if blocked:
                continue
            v = align * t_danger * max(0.0, 1.0 - d / (2.2 * radius))
            if v > best:
                best = v
        total += best
    return total


def evaluate(
    state: GameState,
    perspective: int,
    radius: float = 10.0,
    marble_r: float = 0.45,
    weights: EvalWeights = DEFAULT_WEIGHTS,
    bias: EvalBias | None = None,
) -> float:
    if state.winner is not None:
        if state.winner == DRAW:
            return 0.0
        # faster wins / slower losses are slightly preferred; survivors count too
        margin = 50.0 * state.count(state.winner) - state.turn
        return WIN_SCORE + margin if state.winner == perspective else -(WIN_SCORE + margin)
    return evaluate_features(state, perspective, radius, marble_r, weights, bias)["total"]


def evaluate_features(
    state: GameState,
    perspective: int,
    radius: float = 10.0,
    marble_r: float = 0.45,
    weights: EvalWeights = DEFAULT_WEIGHTS,
    bias: EvalBias | None = None,
) -> dict[str, float]:
    """Non-terminal feature breakdown (used by the analysis overlay)."""
    w = weights
    b = bias or EvalBias.neutral()
    me, opp = perspective, 1 - perspective
    own = [m for m in state.marbles if m.alive and m.owner == me]
    their = [m for m in state.marbles if m.alive and m.owner == opp]

    material = w.material * (len(own) - len(their))
    own_danger = sum(_edge_danger(m, radius, w.danger_band) * (1.0 + b.protect.get(m.mid, 0.0)) for m in own)
    opp_danger = sum(_edge_danger(m, radius, w.danger_band) for m in their)
    edge = -w.own_edge * b.defense_mult * own_danger + w.opp_edge * opp_danger

    my_attack = _attack_potential(state, me, radius, marble_r)
    their_attack = _attack_potential(state, opp, radius, marble_r) * b.threat_mult
    if state.to_move == me:
        attack = w.attack_to_move * my_attack - w.attack_waiting * their_attack
    else:
        attack = w.attack_waiting * my_attack - w.attack_to_move * their_attack

    def avg_margin(ms):
        return sum(radius - math.hypot(m.x, m.y) for m in ms) / len(ms) if ms else 0.0

    safety = w.safety * (avg_margin(own) - avg_margin(their))
    abil = w.ability * (sum(state.abilities[me]) - sum(state.abilities[opp]))
    total = material + edge + attack + safety + abil
    return {"material": material, "edge": edge, "attack": attack, "safety": safety, "abilities": abil, "total": total}
