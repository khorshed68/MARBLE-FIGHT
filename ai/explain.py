"""Plain-language explanation of an AI decision, built only from measured numbers.

Given the position and the agent's :class:`Decision`, this module re-runs the
chosen shot through the shared physics (the same ``simulate_turn`` the agent
used) and compares the position before and after it with the evaluation
features (marbles, edge danger, attack lines). Together with the decision's own
statistics (fuzzy risk, candidate values or visits, search size) this gives a
short list of reasons such as "Knocks out enemy marble M2" or "Next best: ...
(value 27.4 vs 35.8)". Nothing is invented: every reason is a measured change
or a number the search reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai.evaluation import _attack_potential
from config import DEFAULT_CONFIG, GameConfig
from game.actions import Action
from game.arena import Arena, get_arena
from game.simulation import simulate_turn
from game.state import DRAW, GameState

GOOD, BAD, NEUTRAL = "good", "bad", "neutral"
ALGO_SHORT = {"Minimax + Alpha-Beta": "Minimax", "MCTS (UCT)": "MCTS"}
DANGER_BAND = 4.0          # same band as the evaluation function's edge danger
STEP = 0.15                # smallest change worth mentioning


@dataclass
class Reason:
    kind: str              # GOOD, BAD or NEUTRAL
    text: str


@dataclass
class RankRow:
    rank: int
    label: str
    score: str             # "35.8" (Minimax value) or "13 visits" (MCTS)
    risk: float | None
    selected: bool
    action: Action | None = None
    mean: float | None = None          # MCTS mean reward (Q) of the shot


@dataclass
class Explanation:
    player: int
    algorithm: str
    headline: str
    reasons: list[Reason] = field(default_factory=list)
    ranking: list[RankRow] = field(default_factory=list)
    search: str = ""


def _ranking(decision) -> list[RankRow]:
    from ai.search_trace import action_label
    rows = []
    for i, c in enumerate(decision.candidates[:4]):
        try:
            a = Action.from_dict(c["action"])
        except (KeyError, TypeError, ValueError):
            continue
        if "visits" in c:
            score = f"{c['visits']} visits"
        else:
            v = c.get("value")
            score = "-" if v is None else f"{v:.1f}"
        row = RankRow(i + 1, action_label(a), score, c.get("risk"), bool(c.get("selected")), a)
        if "visits" in c and c.get("value") is not None:
            row.mean = float(c["value"])
        rows.append(row)
    return rows


def explain(state: GameState, decision, arena: Arena | None = None, cfg: GameConfig = DEFAULT_CONFIG) -> Explanation:
    """Explain ``decision`` (taken in ``state``) with reasons measured from the real shot."""
    from ai.search_trace import action_label
    arena = arena or get_arena(state.arena_id)
    p = decision.player
    a = decision.action
    algo = ALGO_SHORT.get(decision.algorithm, decision.algorithm)
    ex = Explanation(p, algo, f"P{p + 1} {algo} chose: {action_label(a)}")
    ex.ranking = _ranking(decision)
    s = decision.stats or {}
    if "nodes_visited" in s:
        ex.search = (f"Compared {s.get('root_candidates', '?')} shots, {s.get('depth_reached', '?')} turn(s) ahead, "
                     f"{s.get('branches_pruned', 0)} branches pruned")
    elif "iterations" in s:
        ex.search = f"{s.get('iterations', '?')} simulated futures; chosen shot visited {s.get('selected_visits', '?')} times"
    if decision.fallback:
        ex.reasons.append(Reason(BAD, "Ran out of time: used a safe fallback shot"))
        return ex

    res = simulate_turn(state, a, arena, cfg, record=False, validate=False, record_contacts=False)
    after = res.state
    R = arena.radius
    opp = 1 - p
    reasons: list[Reason] = []
    # 1. result of the shot
    if after.winner == p:
        reasons.append(Reason(GOOD, "Wins the match with this shot"))
    elif after.winner == DRAW:
        reasons.append(Reason(NEUTRAL, "Ends the match in a draw"))
    lost_enemy = [m for m in res.eliminated if state.marbles[m].owner == opp]
    lost_own = [m for m in res.eliminated if state.marbles[m].owner == p]
    if lost_enemy:
        reasons.append(Reason(GOOD, "Knocks out enemy marble " + ", ".join(f"M{m}" for m in lost_enemy)))
    if lost_own:
        reasons.append(Reason(BAD, "Loses own marble " + ", ".join(f"M{m}" for m in lost_own) +
                              " (still the best option found)"))
    # 2. distance to the rim of the marbles still in play (measured before and after the shot)
    def rim(st, mid):
        m = st.marbles[mid]
        return R - (m.x * m.x + m.y * m.y) ** 0.5

    enemy = [(rim(state, m.mid), rim(after, m.mid), m.mid) for m in after.marbles
             if m.alive and m.owner == opp and state.marbles[m.mid].alive]
    pushed = [e for e in enemy if e[0] - e[1] >= 0.4 and e[1] < DANGER_BAND]
    if pushed:
        d0, d1, mid = min(pushed, key=lambda e: e[1])
        reasons.append(Reason(GOOD, f"Pushes enemy M{mid} toward the edge ({d0:.1f} -> {d1:.1f} from the rim)"))
    own = [(rim(state, m.mid), rim(after, m.mid), m.mid) for m in after.marbles
           if m.alive and m.owner == p and state.marbles[m.mid].alive]
    exposed = [o for o in own if o[1] < 2.5 and o[0] - o[1] >= 0.3]
    safer = [o for o in own if o[0] < DANGER_BAND and o[1] - o[0] >= 0.4]
    if exposed:
        d0, d1, mid = min(exposed, key=lambda o: o[1])
        reasons.append(Reason(BAD, f"Leaves own M{mid} only {d1:.1f} from the edge"))
    elif safer:
        d0, d1, mid = max(safer, key=lambda o: o[1] - o[0])
        reasons.append(Reason(GOOD, f"Moves own M{mid} away from the edge ({d0:.1f} -> {d1:.1f})"))
    # 3. attack lines for the next turns
    mr = cfg.physics.marble_radius
    if after.winner is None:
        my0, my1 = _attack_potential(state, p, R, mr), _attack_potential(after, p, R, mr)
        th0, th1 = _attack_potential(state, opp, R, mr), _attack_potential(after, opp, R, mr)
        if my1 - my0 > STEP:
            reasons.append(Reason(GOOD, "Sets up an attack line for the next turn"))
        if th1 - th0 > STEP:
            reasons.append(Reason(BAD, "Gives the opponent an attack chance"))
        elif th0 - th1 > STEP:
            reasons.append(Reason(GOOD, "Blocks or removes an enemy attack line"))
    if not reasons:
        reasons.append(Reason(NEUTRAL, "Quiet positioning shot (no big change on the board)"))
    # 4. fuzzy risk of the shot, as computed by the agent
    r = s.get("fuzzy_risk")
    if r is not None:
        if r < 0.35:
            reasons.append(Reason(GOOD, f"Low fuzzy risk for its own marbles ({r:.2f})"))
        elif r < 0.6:
            reasons.append(Reason(NEUTRAL, f"Medium fuzzy risk ({r:.2f})"))
        else:
            reasons.append(Reason(BAD, f"High fuzzy risk ({r:.2f}), but nothing safer scored better"))
    # 5. why not the runner-up
    chosen = next((row for row in ex.ranking if row.selected), None)
    other = next((row for row in ex.ranking if not row.selected), None)
    def num(row):
        try:
            return float(row.score)
        except (TypeError, ValueError):
            return None

    cv = num(chosen) if chosen is not None and algo != "MCTS" else None
    if cv is not None and cv >= 9000:
        reasons.insert(0, Reason(GOOD, "Search proved a forced win from here"))
    elif cv is not None and cv <= -9000:
        reasons.insert(0, Reason(BAD, "Every searched line loses: picks the slowest loss"))
    if chosen is not None and other is not None:
        risky = " (risky)" if (other.risk or 0) >= 0.6 else ""
        ov = num(other) if algo != "MCTS" else None
        if cv is not None and ov is not None and cv >= 9000 and ov >= 9000:
            reasons.append(Reason(NEUTRAL, f"{other.label} also wins, but later or with fewer marbles left"))
        elif chosen.score == other.score:
            how = "it has the higher mean reward" if algo == "MCTS" else "it came first in the search order"
            reasons.append(Reason(NEUTRAL, f"Tied with {other.label}{risky} ({other.score}); picked because {how}"))
        elif algo == "MCTS":
            reasons.append(Reason(BAD, f"Next best: {other.label}{risky}, {other.score} vs {chosen.score}"))
        else:
            ahead = " after the reply" if (s.get("depth_reached") or 0) >= 2 else ""
            reasons.append(Reason(BAD, f"Next best: {other.label}{risky}, value {other.score} vs {chosen.score}{ahead}"))
    ex.reasons = reasons[:6]
    return ex


def explanation_dict(ex: Explanation) -> dict[str, Any]:
    return {"headline": ex.headline, "search": ex.search,
            "reasons": [{"kind": r.kind, "text": r.text} for r in ex.reasons],
            "ranking": [{"rank": r.rank, "label": r.label, "score": r.score, "risk": r.risk, "selected": r.selected}
                        for r in ex.ranking]}
