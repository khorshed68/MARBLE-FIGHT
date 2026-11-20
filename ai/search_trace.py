"""Record the real Alpha-Beta search as a tree, for the Algorithm Lab's tree view.

:func:`traced_alphabeta` follows :func:`ai.minimax.alphabeta` step by step (same
move order, same shifted window, same cutoff test) and additionally stores
every visited node, every pruned sibling and the alpha/beta window at each node.
The tests check that it returns the same value and the same statistics as the
real function, so the picture is the search the agent really performs, not an
illustration.

Only small trees are meant to be drawn: :func:`trace_decision` uses a narrow
beam (default 4 root shots, 3 replies) and depth 1 or 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ai.base_agent import Clock
from ai.minimax import MarbleSearchProblem, MinimaxAgent, SearchStats, alphabeta, minimax
from config import DEFAULT_CONFIG, GameConfig, MinimaxConfig
from game.actions import Action
from game.state import GameState


@dataclass
class TraceNode:
    nid: int
    parent: int | None
    ply: int
    action: Action | None            # move leading into this node (None at the root)
    edge: float = 0.0                # fuzzy edge value of that move (root-player view)
    is_max: bool | None = None       # None for leaves
    alpha: float = -math.inf         # window when the node was entered (its own frame)
    beta: float = math.inf
    value: float | None = None       # value returned by the search for this node
    leaf: bool = False
    pruned: bool = False             # never searched: skipped by a cutoff
    cutoff: bool = False             # a cutoff happened inside this node
    pruned_count: int = 0
    children: list[int] = field(default_factory=list)
    state: GameState | None = None

    @property
    def backed_up(self) -> float | None:
        """Value as seen by the parent: edge + value."""
        return None if self.value is None else self.edge + self.value


@dataclass
class SearchTrace:
    nodes: list[TraceNode]
    value: float
    best_child: int | None           # root child chosen by the search
    stats: SearchStats
    minimax_value: float
    minimax_nodes: int
    depth: int
    root_player: int

    @property
    def root(self) -> TraceNode:
        return self.nodes[0]

    def principal_variation(self) -> list[int]:
        """Node ids along the line both players are assumed to play (root first)."""
        path = [0]
        n = self.nodes[0]
        while n.children:
            want = n.value
            nxt = None
            for c in n.children:
                ch = self.nodes[c]
                if not ch.pruned and ch.backed_up is not None and want is not None and ch.backed_up == want:
                    nxt = ch
                    break
            if nxt is None:
                break
            path.append(nxt.nid)
            n = nxt
        return path


def traced_alphabeta(problem, node, depth: int, alpha: float, beta: float, stats: SearchStats,
                     nodes: list[TraceNode], parent: int | None = None, ply: int = 0,
                     action: Action | None = None, edge: float = 0.0) -> float:
    """Exactly the logic of ai.minimax.alphabeta, plus recording."""
    me = TraceNode(len(nodes), parent, ply, action, edge, alpha=alpha, beta=beta, state=node)
    nodes.append(me)
    if parent is not None:
        nodes[parent].children.append(me.nid)
    stats.nodes_visited += 1
    stats.max_depth_reached = max(stats.max_depth_reached, ply)
    if depth == 0 or problem.is_terminal(node):
        stats.nodes_evaluated += 1
        me.leaf = True
        me.value = problem.evaluate(node)
        return me.value
    acts = problem.actions(node, depth)
    if not acts:
        stats.nodes_evaluated += 1
        me.leaf = True
        me.value = problem.evaluate(node)
        return me.value
    maximize = problem.is_max(node)
    me.is_max = maximize
    value = -math.inf if maximize else math.inf
    for i, a in enumerate(acts):
        e, child = problem.result(node, a)
        v = e + traced_alphabeta(problem, child, depth - 1, alpha - e, beta - e, stats, nodes, me.nid, ply + 1, a, e)
        if maximize:
            if v > value:
                value = v
            if value > alpha:
                alpha = value
        else:
            if v < value:
                value = v
            if value < beta:
                beta = value
        if alpha >= beta:
            stats.cutoffs += 1
            skipped = acts[i + 1:]
            stats.branches_pruned += len(skipped)
            me.cutoff = True
            me.pruned_count = len(skipped)
            for b in skipped:          # recorded for the picture only: never simulated
                pn = TraceNode(len(nodes), me.nid, ply + 1, b, pruned=True)
                nodes.append(pn)
                me.children.append(pn.nid)
            break
    me.value = value
    return value


def trace_decision(state: GameState, player: int | None = None, depth: int = 2, root_width: int = 4,
                   inner_width: int = 3, use_fuzzy: bool = True, game_cfg: GameConfig = DEFAULT_CONFIG) -> SearchTrace:
    """One Alpha-Beta pass of the real Minimax agent at a fixed depth, recorded node by node."""
    player = state.to_move if player is None else player
    cfg = MinimaxConfig(max_depth=depth, root_width=root_width, inner_width=inner_width, node_budget=10**6,
                        time_budget=None, use_fuzzy=use_fuzzy, use_opponent_model=False)
    agent = MinimaxAgent(player, cfg, game_cfg)
    agent._prepare(state)
    problem = MarbleSearchProblem(agent, player, Clock(None), cfg)
    problem.root_state = state
    nodes: list[TraceNode] = []
    stats = SearchStats()
    value = traced_alphabeta(problem, state, depth, -math.inf, math.inf, stats, nodes)
    best = None
    for c in nodes[0].children:          # first root child reaching the value (same rule as the agent)
        ch = nodes[c]
        if not ch.pruned and ch.backed_up == value:
            best = c
            break
    mm_stats = SearchStats()             # plain minimax on the same tree, for the node comparison
    mm_value = minimax(problem, state, depth, mm_stats)
    return SearchTrace(nodes, value, best, stats, mm_value, mm_stats.nodes_visited, depth, player)


SHORT_ABILITY = {"power": "Power", "precision": "Precision", "spin_left": "Spin L", "spin_right": "Spin R"}


def action_label(a: Action | None) -> str:
    """Short readable name of a shot, e.g. 'M1 push M2 | Power 100%'."""
    if a is None:
        return "root"
    tag = a.tag or ""
    if tag.startswith("push#"):
        what = f"push M{tag[5:]}"
    elif tag.startswith("hit#"):
        what = f"hit M{tag[4:]}"
    elif tag.startswith("curve#"):
        what = f"curve to M{tag[6:]}"
    elif tag == "retreat":
        what = "retreat"
    else:
        what = f"{math.degrees(a.angle) % 360:.0f} deg"
    ab = f" | {SHORT_ABILITY.get(a.ability, a.ability)}" if a.ability != "none" else " | Normal"
    return f"M{a.marble_id} {what}{ab} {a.power * 100:.0f}%"


# ----------------------------------------------------------------------------- demo positions
def demo_positions() -> list[tuple[str, str, Any]]:
    """(key, title, state builder) for the Lab's scenario selector."""
    from game.arena import get_arena
    from game.rules import initial_state
    from game.state import MarbleState

    def trap():
        ms = ((0, 0.0, -3.0), (0, 4.0, -4.0), (1, 2.0, 7.0), (1, -3.0, 2.0))
        marbles = tuple(MarbleState(i, o, x, y, True) for i, (o, x, y) in enumerate(ms))
        return GameState(marbles, 0, 0, ((2, 2, 2), (2, 2, 2)), "classic", 60)

    return [
        ("trap", "Knockout trap (Classic): enemy marble near the rim", trap),
        ("open_classic", "Opening position (Classic Kancha Court)", lambda: initial_state(get_arena("classic"))),
        ("open_neon", "Opening position (Neon Crystal Arena)", lambda: initial_state(get_arena("neon"))),
    ]


# ============================================================================ MCTS, one iteration at a time
@dataclass
class IterationEvent:
    """What one MCTS iteration did (for the Lab's step-by-step view)."""

    index: int                        # 1-based iteration number
    path: list[Any]                   # MCTSNode objects from the root to the evaluated node
    expanded: Any | None              # node added by the expansion phase (None if none)
    reward: float                     # reward for the root player
    rollout_turns: int
    terminal: bool


class MCTSStepper:
    """Runs the real MCTSAgent's four phases one iteration at a time.

    ``step`` performs exactly one pass of the loop in :meth:`MCTSAgent.decide`
    (select -> expand -> rollout -> backpropagate) with the agent's own methods,
    seed and configuration, so after N steps the tree is the one ``decide``
    builds with N iterations (checked by the tests).
    """

    def __init__(self, state: GameState, iterations: int = 40, candidate_width: int = 4, rollout_depth: int = 1,
                 seed: int = 7, use_fuzzy: bool = True, game_cfg: GameConfig = DEFAULT_CONFIG) -> None:
        from ai.mcts import MCTSAgent, MCTSNode
        from config import MCTSConfig
        self.cfg = MCTSConfig(iterations=iterations, rollout_depth=rollout_depth, candidate_width=candidate_width,
                              time_budget=None, use_fuzzy=use_fuzzy, use_opponent_model=False)
        self.state = state
        self.player = state.to_move
        self.agent = MCTSAgent(self.player, self.cfg, game_cfg, seed=seed)
        self.agent._prepare(state)
        self.root = MCTSNode(state, None, None, 1 - state.to_move)
        self.agent._untried(self.root)          # same lazy list the first selection would build
        self.events: list[IterationEvent] = []

    @property
    def iterations(self) -> int:
        return len(self.events)

    @property
    def done(self) -> bool:
        return self.iterations >= self.cfg.iterations

    def step(self) -> IterationEvent | None:
        if self.done:
            return None
        ag = self.agent
        leaf, acc = ag.select(self.root)
        expanded = None
        if not leaf.is_terminal and ag._untried(leaf):
            leaf = ag.expand(leaf)
            acc += leaf.edge
            expanded = leaf
        if leaf.is_terminal:
            reward, d = ag.terminal_reward(leaf.state), 0
        else:
            reward, d = ag.rollout(leaf.state, acc)
        ag.backpropagate(leaf, reward, self.player)
        path = []
        n = leaf
        while n is not None:
            path.append(n)
            n = n.parent
        ev = IterationEvent(self.iterations + 1, path[::-1], expanded, reward, d, leaf.is_terminal)
        self.events.append(ev)
        return ev

    def run(self, n: int) -> None:
        for _ in range(n):
            if self.step() is None:
                break

    def best(self):
        """The child the agent would play now: most visits, then mean reward (as in decide)."""
        if not self.root.children:
            return None
        return max(self.root.children, key=lambda ch: (ch.visits, ch.mean))

    def uct(self, node) -> float:
        from ai.mcts import uct_value
        p = node.parent
        return uct_value(node.value_sum, node.visits, p.visits if p is not None else 0, self.cfg.exploration)

    def widening_limit(self, node) -> int:
        return self.agent._widening_limit(node)


# ============================================================================ positions from recorded matches
@dataclass
class MatchPositions:
    """Every position of a recorded match, rebuilt by re-simulating its replay."""

    path: str
    record: dict[str, Any]
    states: list[GameState]          # states[i] = position before turn i (only verified turns)
    error: str | None = None         # why rebuilding stopped early (e.g. old physics settings)

    @property
    def turns(self) -> list[dict[str, Any]]:
        return self.record["turns"][:len(self.states)]

    def players(self) -> str:
        from ai.factory import AGENT_LABELS
        names = []
        for s in self.record["specs"]:
            names.append("Human" if s["kind"] == "human" else AGENT_LABELS.get(s["kind"], s["kind"]).split(" ")[0])
        return " vs ".join(names)

    def title(self) -> str:
        from game.arena import get_arena
        r = self.record
        try:
            arena = get_arena(r["arena"]).name
        except KeyError:
            arena = r["arena"]
        res = r.get("result", {})
        score = res.get("score")
        sc = f" | final {score[0]}:{score[1]}" if score else ""
        return f"{r.get('created', '?')} | {self.players()}\n{arena}{sc}"

    def controller(self, turn: int) -> str:
        p = self.turns[turn]["player"]
        s = self.record["specs"][p]
        if s["kind"] == "human":
            return "Human"
        from ai.factory import AGENT_LABELS
        return f"{AGENT_LABELS.get(s['kind'], s['kind'])} ({s.get('difficulty', '?')})"


def load_match_positions(path: str, game_cfg: GameConfig = DEFAULT_CONFIG) -> MatchPositions:
    """Re-simulate a replay file and keep the position before every turn that reproduces exactly."""
    from game.replay import ReplayError, ReplayPlayer, load_replay
    rec = load_replay(path)
    p = ReplayPlayer(rec, game_cfg)
    states: list[GameState] = []
    error = None
    if not p.config_matches:
        error = "recorded with different game settings; positions cannot be rebuilt exactly"
        return MatchPositions(path, rec, states, error)
    while not p.done:
        before = p.state
        try:
            p.step()
        except ReplayError as exc:
            error = str(exc)
            break
        states.append(before)
    return MatchPositions(path, rec, states, error)


def recent_matches(limit: int = 5, directory: str | None = None,
                   game_cfg: GameConfig = DEFAULT_CONFIG) -> list[MatchPositions]:
    """The most recent saved matches whose positions can be rebuilt (newest first)."""
    from game.replay import REPLAY_DIR, ReplayError, list_replays
    out = []
    for f in list_replays(directory or REPLAY_DIR, limit=limit * 3):
        try:
            mp = load_match_positions(f, game_cfg)
        except ReplayError:
            continue
        if mp.states:
            out.append(mp)
        if len(out) >= limit:
            break
    return out


def same_shot(a: Action | None, b: Action | None) -> bool:
    """Same marble, ability, power and (to 0.01 rad) direction."""
    if a is None or b is None:
        return False
    d = abs((a.angle - b.angle + math.pi) % (2 * math.pi) - math.pi)
    return a.marble_id == b.marble_id and a.ability == b.ability and abs(a.power - b.power) < 1e-6 and d < 0.01
