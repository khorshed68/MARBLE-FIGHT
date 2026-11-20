"""Mamdani fuzzy-logic risk evaluator for a proposed shot.

Pipeline (all four stages are real and individually tested):

1. **Fuzzification** – six crisp inputs are mapped to degrees of membership in
   overlapping trapezoid/triangle fuzzy sets.
2. **Rule evaluation** – 14 IF-THEN rules; AND = min, OR = max, NOT = 1 - x.
   Each rule's firing strength clips (min-implication) its output set.
3. **Aggregation** – clipped output sets are combined with max.
4. **Defuzzification** – centroid of the aggregated set over the risk universe
   [0, 1] sampled at 101 points.

Output: risk in [0, 1]; larger = more dangerous for the *shooter's own side*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------- membership functions
def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoid: 0 below a, rises to 1 at b, flat to c, falls to 0 at d.
    a == b (or c == d) gives a vertical shoulder."""
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (d - x) / (d - c) if d > c else 1.0


def trimf(x: float, a: float, b: float, c: float) -> float:
    return trapmf(x, a, b, b, c)


@dataclass(frozen=True)
class FuzzySet:
    name: str
    kind: str          # "trap" or "tri"
    params: tuple[float, ...]

    def mu(self, x: float) -> float:
        return trapmf(x, *self.params) if self.kind == "trap" else trimf(x, *self.params)


def _T(name: str, *p: float) -> FuzzySet:
    return FuzzySet(name, "trap", p)


def _R(name: str, *p: float) -> FuzzySet:
    return FuzzySet(name, "tri", p)


# Linguistic variables ------------------------------------------------------
# edge_distance: distance (world units) from the shooter's start to the boundary (arena radius 10)
EDGE = {"near": _T("near", 0, 0, 1.5, 3.5), "medium": _R("medium", 2.0, 4.5, 7.0), "far": _T("far", 5.5, 7.5, 20, 20)}
# power: effective launch power (Power Shot can exceed 1.0)
POWER = {"low": _T("low", 0, 0, 0.3, 0.5), "medium": _R("medium", 0.35, 0.65, 0.9), "high": _T("high", 0.75, 1.0, 2.0, 2.0)}
# alignment: cos(angle between shot direction and the outward radial at the shooter)
ALIGN = {"inward": _T("inward", -1, -1, -0.6, -0.2), "sideways": _R("sideways", -0.5, 0.0, 0.5),
         "toward_edge": _T("toward_edge", 0.3, 0.7, 1, 1)}
# stop_margin: predicted distance from the shooter's stopping point to the boundary (0 = eliminated)
STOP = {"unsafe": _T("unsafe", 0, 0, 0.8, 2.0), "marginal": _R("marginal", 1.0, 2.5, 4.0), "safe": _T("safe", 3.0, 5.0, 20, 20)}
# threat: how well an opponent marble can knock the stopped shooter outward, in [0, 1]
THREAT = {"low": _T("low", 0, 0, 0.2, 0.45), "high": _T("high", 0.35, 0.65, 1, 1)}
# friction: surface friction multiplier under the shooter
FRICTION = {"slippery": _T("slippery", 0, 0, 0.5, 0.8), "normal": _R("normal", 0.6, 1.0, 1.5), "grippy": _T("grippy", 1.3, 1.8, 5, 5)}
# output: risk
RISK = {"low": _T("low", 0, 0, 0.15, 0.35), "moderate": _R("moderate", 0.25, 0.5, 0.75),
        "high": _R("high", 0.6, 0.78, 0.92), "very_high": _T("very_high", 0.85, 0.95, 1, 1)}

VARIABLES = {"edge": EDGE, "power": POWER, "align": ALIGN, "stop": STOP, "threat": THREAT, "friction": FRICTION}

UNIVERSE = [i / 100.0 for i in range(101)]


@dataclass(frozen=True)
class Rule:
    text: str
    antecedent: tuple                 # nested ("and"/"or"/"not", ...) or (var, set)
    consequent: str                   # RISK set name


def _A(*xs):
    return ("and",) + xs


def _O(*xs):
    return ("or",) + xs


RULES: tuple[Rule, ...] = (
    Rule("IF edge is near AND power is high THEN risk is very_high", _A(("edge", "near"), ("power", "high")), "very_high"),
    Rule("IF edge is far AND power is medium AND stop is NOT unsafe THEN risk is low",
         _A(("edge", "far"), ("power", "medium"), ("not", ("stop", "unsafe"))), "low"),
    Rule("IF align is toward_edge AND power is high THEN risk is high", _A(("align", "toward_edge"), ("power", "high")), "high"),
    Rule("IF stop is safe THEN risk is low", ("stop", "safe"), "low"),
    Rule("IF stop is unsafe THEN risk is very_high", ("stop", "unsafe"), "very_high"),
    Rule("IF stop is marginal THEN risk is moderate", ("stop", "marginal"), "moderate"),
    Rule("IF threat is high AND stop is marginal THEN risk is high", _A(("threat", "high"), ("stop", "marginal")), "high"),
    Rule("IF threat is high AND stop is safe THEN risk is moderate", _A(("threat", "high"), ("stop", "safe")), "moderate"),
    Rule("IF friction is slippery AND power is high THEN risk is high", _A(("friction", "slippery"), ("power", "high")), "high"),
    Rule("IF align is inward AND (power is low OR power is medium) AND stop is NOT unsafe THEN risk is low",
         _A(("align", "inward"), _O(("power", "low"), ("power", "medium")), ("not", ("stop", "unsafe"))), "low"),
    Rule("IF edge is near AND align is toward_edge THEN risk is high", _A(("edge", "near"), ("align", "toward_edge")), "high"),
    Rule("IF edge is medium AND power is medium AND align is sideways AND stop is NOT unsafe THEN risk is moderate",
         _A(("edge", "medium"), ("power", "medium"), ("align", "sideways"), ("not", ("stop", "unsafe"))), "moderate"),
    Rule("IF threat is low AND stop is safe THEN risk is low", _A(("threat", "low"), ("stop", "safe")), "low"),
    Rule("IF friction is grippy AND stop is safe THEN risk is low", _A(("friction", "grippy"), ("stop", "safe")), "low"),
)


@dataclass
class FuzzyInputs:
    edge: float          # world units to the boundary at shot start
    power: float         # effective power
    align: float         # cosine in [-1, 1]
    stop: float          # predicted stop margin (world units)
    threat: float        # [0, 1]
    friction: float      # surface multiplier


@dataclass
class FuzzyResult:
    risk: float
    memberships: dict[str, dict[str, float]] = field(default_factory=dict)
    activations: list[tuple[str, float]] = field(default_factory=list)


class FuzzyRiskEvaluator:
    """Computes the shot risk with the rule base above."""

    def __init__(self, rules: tuple[Rule, ...] = RULES, resolution: int = 101) -> None:
        self.rules = rules
        self.universe = [i / (resolution - 1) for i in range(resolution)]
        # precompute output set samples for speed
        self._out = {name: [s.mu(u) for u in self.universe] for name, s in RISK.items()}

    # 1. fuzzification
    @staticmethod
    def fuzzify(inp: FuzzyInputs) -> dict[str, dict[str, float]]:
        vals = {"edge": inp.edge, "power": inp.power, "align": inp.align, "stop": inp.stop,
                "threat": inp.threat, "friction": inp.friction}
        return {var: {name: s.mu(vals[var]) for name, s in sets.items()} for var, sets in VARIABLES.items()}

    # 2. rule evaluation
    def _strength(self, ante, fz: dict[str, dict[str, float]]) -> float:
        op = ante[0]
        if op == "and":
            return min(self._strength(a, fz) for a in ante[1:])
        if op == "or":
            return max(self._strength(a, fz) for a in ante[1:])
        if op == "not":
            return 1.0 - self._strength(ante[1], fz)
        var, name = ante
        return fz[var][name]

    def evaluate_rules(self, fz: dict[str, dict[str, float]]) -> list[tuple[Rule, float]]:
        return [(r, self._strength(r.antecedent, fz)) for r in self.rules]

    # 3. aggregation (max of min-clipped consequents)
    def aggregate(self, fired: list[tuple[Rule, float]]) -> list[float]:
        agg = [0.0] * len(self.universe)
        for rule, w in fired:
            if w <= 0.0:
                continue
            curve = self._out[rule.consequent]
            for i, mu in enumerate(curve):
                v = mu if mu < w else w
                if v > agg[i]:
                    agg[i] = v
        return agg

    # 4. defuzzification (centroid)
    def defuzzify(self, agg: list[float]) -> float:
        num = sum(u * m for u, m in zip(self.universe, agg))
        den = sum(agg)
        if den <= 1e-12:
            return 0.5   # no rule fired: neutral risk (cannot happen with this rule base, see tests)
        return min(1.0, max(0.0, num / den))

    def evaluate(self, inp: FuzzyInputs, explain: bool = False) -> FuzzyResult:
        fz = self.fuzzify(inp)
        fired = self.evaluate_rules(fz)
        risk = self.defuzzify(self.aggregate(fired))
        if not explain:
            return FuzzyResult(risk)
        return FuzzyResult(risk, fz, [(r.text, w) for r, w in fired if w > 0])

    # ------------------------------------------------------------------ helpers for the game
    def shot_inputs(self, state_before, action, state_after, arena, cfg) -> FuzzyInputs:
        """Extract crisp fuzzy inputs from a simulated shot."""
        s = state_before.marbles[action.marble_id]
        R = arena.radius
        r0 = math.hypot(s.x, s.y)
        edge = max(0.0, R - r0)
        eff_power = action.power * (cfg.physics.power_shot_multiplier if action.ability == "power" else 1.0)
        if r0 > 1e-6:
            align = (math.cos(action.angle) * s.x + math.sin(action.angle) * s.y) / r0
        else:
            align = 0.0
        after = state_after.marbles[action.marble_id]
        if not after.alive:
            stop, threat = 0.0, 1.0
        else:
            rs = math.hypot(after.x, after.y)
            stop = max(0.0, R - rs)
            threat = opponent_threat(state_after, after, rs)
        return FuzzyInputs(edge, eff_power, align, stop, threat, arena.friction_at(s.x, s.y))

    def shot_risk(self, state_before, action, state_after, arena, cfg) -> float:
        return self.evaluate(self.shot_inputs(state_before, action, state_after, arena, cfg)).risk


def opponent_threat(state, marble, r_m: float | None = None) -> float:
    """[0,1]: how well-placed the best enemy marble is to knock ``marble`` outward."""
    if r_m is None:
        r_m = math.hypot(marble.x, marble.y)
    if r_m < 1e-6:
        return 0.0
    ux, uy = marble.x / r_m, marble.y / r_m
    best = 0.0
    for o in state.marbles:
        if not o.alive or o.owner == marble.owner:
            continue
        dx, dy = marble.x - o.x, marble.y - o.y
        d = math.hypot(dx, dy)
        if d < 1e-6:
            continue
        closeness = min(1.0, max(0.0, (7.0 - d) / 6.0))
        outward = (dx * ux + dy * uy) / d
        v = closeness * (0.5 + 0.5 * outward)
        if v > best:
            best = v
    return best
