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
    """Non-terminal feature breakdown; :func:`evaluation_terms` shows the same numbers with their inputs."""
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


def evaluation_terms(
    state: GameState,
    perspective: int,
    radius: float = 10.0,
    marble_r: float = 0.45,
    weights: EvalWeights = DEFAULT_WEIGHTS,
    bias: EvalBias | None = None,
) -> list[tuple[str, str, str, float]]:
    """(feature, general formula, formula with the numbers, contribution) rows whose sum is ``evaluate``.

    Every row is computed with the same helpers and weights as :func:`evaluate_features`
    (used by the SHOW HEURISTIC view of the match popup and the Algorithm Lab).
    """
    if state.winner is not None:
        v = evaluate(state, perspective, radius, marble_r, weights, bias)
        if state.winner == DRAW:
            return [("terminal", "draw = 0", "0", v)]
        sign = "" if state.winner == perspective else "-"
        return [("terminal", "win: +(10000 + 50 x survivors - turn), loss: the negative",
                 f"{sign}(10000 + 50 x {state.count(state.winner)} - {state.turn})", v)]
    w = weights
    b = bias or EvalBias.neutral()
    me, opp = perspective, 1 - perspective
    own = [m for m in state.marbles if m.alive and m.owner == me]
    their = [m for m in state.marbles if m.alive and m.owner == opp]
    rows: list[tuple[str, str, str, float]] = []
    rows.append(("material", f"{w.material:g} x (own marbles - enemy marbles)",
                 f"{w.material:g} x ({len(own)} - {len(their)})", w.material * (len(own) - len(their))))
    od = sum(_edge_danger(m, radius, w.danger_band) * (1.0 + b.protect.get(m.mid, 0.0)) for m in own)
    td = sum(_edge_danger(m, radius, w.danger_band) for m in their)
    rows.append(("edge danger", f"-{w.own_edge:g} x defense x SUM own danger + {w.opp_edge:g} x SUM enemy danger",
                 f"-{w.own_edge:g} x {b.defense_mult:g} x {od:.3f} + {w.opp_edge:g} x {td:.3f}",
                 -w.own_edge * b.defense_mult * od + w.opp_edge * td))
    my = _attack_potential(state, me, radius, marble_r)
    raw_th = _attack_potential(state, opp, radius, marble_r)
    th = raw_th * b.threat_mult
    if state.to_move == me:
        gen = f"{w.attack_to_move:g} x own lines - {w.attack_waiting:g} x threat x enemy lines   (I shoot next)"
        num = f"{w.attack_to_move:g} x {my:.3f} - {w.attack_waiting:g} x {b.threat_mult:g} x {raw_th:.3f}"
        val = w.attack_to_move * my - w.attack_waiting * th
    else:
        gen = f"{w.attack_waiting:g} x own lines - {w.attack_to_move:g} x threat x enemy lines   (enemy shoots next)"
        num = f"{w.attack_waiting:g} x {my:.3f} - {w.attack_to_move:g} x {b.threat_mult:g} x {raw_th:.3f}"
        val = w.attack_waiting * my - w.attack_to_move * th
    rows.append(("attack lines", gen, num, val))

    def avg_margin(ms):
        return sum(radius - math.hypot(m.x, m.y) for m in ms) / len(ms) if ms else 0.0

    mo, mt = avg_margin(own), avg_margin(their)
    rows.append(("safety margin", f"{w.safety:g} x (own avg distance to rim - enemy avg distance to rim)",
                 f"{w.safety:g} x ({mo:.3f} - {mt:.3f})", w.safety * (mo - mt)))
    ao, at = sum(state.abilities[me]), sum(state.abilities[opp])
    rows.append(("abilities", f"{w.ability:g} x (own charges left - enemy charges left)",
                 f"{w.ability:g} x ({ao} - {at})", w.ability * (ao - at)))
    return rows
