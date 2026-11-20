"""The Lab's search-tree view must show the real Alpha-Beta search, not a drawing of it."""

import math
import random

import pytest

from ai.minimax import MarbleSearchProblem, MinimaxAgent, SearchStats, alphabeta
from ai.search_trace import action_label, demo_positions, trace_decision, traced_alphabeta
from ai.base_agent import Clock
from config import MinimaxConfig
from tests.test_minimax import TreeProblem, random_tree, wrap


@pytest.mark.parametrize("edges", [False, True])
def test_trace_matches_real_alphabeta_on_random_trees(edges):
    rng = random.Random(77 + edges)
    for _ in range(200):
        depth = rng.randint(1, 4)
        tree = random_tree(rng, depth, 4, edges)
        p = TreeProblem(root_is_max=rng.random() < 0.5)
        real, traced = SearchStats(), SearchStats()
        v = alphabeta(p, wrap(tree), depth, -math.inf, math.inf, real)
        nodes = []
        vt = traced_alphabeta(p, wrap(tree), depth, -math.inf, math.inf, traced, nodes)
        assert vt == pytest.approx(v)
        assert traced.as_dict() == real.as_dict()
        visited = [n for n in nodes if not n.pruned]
        assert len(visited) == real.nodes_visited
        assert sum(1 for n in nodes if n.pruned) == real.branches_pruned


def test_textbook_tree_records_the_two_pruned_leaves():
    tree = [[3, 12, 8], [2, 4, 6], [14, 5, 2]]
    nodes = []
    st = SearchStats()
    assert traced_alphabeta(TreeProblem(), wrap(tree), 2, -math.inf, math.inf, st, nodes) == 3
    pruned = [n for n in nodes if n.pruned]
    assert len(pruned) == 2 and all(nodes[n.parent].cutoff for n in pruned)
    second_min = nodes[nodes[0].children[1]]
    assert second_min.cutoff and second_min.value == 2 and second_min.pruned_count == 2


@pytest.mark.parametrize("key", [k for k, _, _ in demo_positions()])
def test_trace_matches_the_real_agent_search_on_game_positions(key):
    state = dict((k, mk) for k, _, mk in demo_positions())[key]()
    tr = trace_decision(state, depth=2, root_width=4, inner_width=3)
    # the same search through the untraced function
    cfg = MinimaxConfig(max_depth=2, root_width=4, inner_width=3, node_budget=10**6, time_budget=None,
                        use_opponent_model=False)
    agent = MinimaxAgent(state.to_move, cfg)
    agent._prepare(state)
    p = MarbleSearchProblem(agent, state.to_move, Clock(None), cfg)
    p.root_state = state
    st = SearchStats()
    v = alphabeta(p, state, 2, -math.inf, math.inf, st)
    assert tr.value == pytest.approx(v) == pytest.approx(tr.minimax_value)
    assert tr.stats.as_dict() == st.as_dict()
    assert tr.stats.nodes_visited <= tr.minimax_nodes
    best = tr.nodes[tr.best_child]
    assert best.backed_up == pytest.approx(tr.value)
    pv = tr.principal_variation()
    assert pv[0] == 0 and pv[1] == tr.best_child and len(pv) == 3
    for nid in pv[1:]:
        assert not tr.nodes[nid].pruned


def test_depth_one_and_labels():
    state = demo_positions()[0][2]()
    tr = trace_decision(state, depth=1, root_width=4)
    assert len(tr.root.children) == 4 and all(tr.nodes[c].leaf for c in tr.root.children)
    assert tr.stats.branches_pruned == 0                    # MAX root alone never cuts
    assert action_label(None) == "root"
    lab = action_label(tr.nodes[tr.best_child].action)
    assert lab.startswith("M") and "%" in lab


# ----------------------------------------------------------------------------- MCTS stepper
def _tree_signature(node, depth=3):
    from ai.search_trace import action_label
    if depth == 0:
        return ()
    return tuple((action_label(c.action), c.visits, round(c.value_sum, 9), _tree_signature(c, depth - 1))
                 for c in node.children)


@pytest.mark.parametrize("key", [k for k, _, _ in demo_positions()])
def test_mcts_stepper_builds_the_same_tree_as_the_real_agent(key):
    from ai.mcts import MCTSAgent
    from ai.search_trace import MCTSStepper
    from config import MCTSConfig
    state = dict((k, mk) for k, _, mk in demo_positions())[key]()
    st = MCTSStepper(state, iterations=30, candidate_width=4, rollout_depth=1, seed=7)
    st.run(1000)                                         # stops at the budget
    assert st.done and st.iterations == 30 and st.root.visits == 30
    agent = MCTSAgent(state.to_move, MCTSConfig(iterations=30, rollout_depth=1, candidate_width=4, time_budget=None,
                                                use_opponent_model=False), seed=7)
    d = agent.decide(state)
    assert _tree_signature(st.root) == _tree_signature(agent.last_root)
    assert st.best().action == d.action
    assert st.step() is None                             # no iteration beyond the budget


def test_mcts_events_describe_each_iteration():
    from ai.search_trace import MCTSStepper
    st = MCTSStepper(demo_positions()[0][2](), iterations=12, candidate_width=4)
    for i in range(12):
        ev = st.step()
        assert ev.index == i + 1 and ev.path[0] is st.root
        assert 0.0 <= ev.reward <= 1.0
        if ev.expanded is not None:
            assert ev.expanded is ev.path[-1] and ev.expanded.visits == 1
    # progressive widening: the first iterations open the root's shots one by one
    assert [len(e.path) for e in st.events[:4]] == [2, 2, 2, 2]
    assert sum(c.visits for c in st.root.children) == st.root.visits == 12
    assert all(st.uct(c) > 0 for c in st.root.children)


# ----------------------------------------------------------------------------- positions from recorded matches
def test_match_positions_are_rebuilt_exactly_from_a_replay(tmp_path):
    from ai.search_trace import load_match_positions, recent_matches, same_shot
    from game.actions import Action
    from game.match import Match, PlayerSpec
    from game.replay import record_from_match, save_replay
    fast = {"max_depth": 1, "root_width": 6, "inner_width": 3}
    m = Match("neon", PlayerSpec("minimax", "easy", overrides=fast), PlayerSpec("human"), seed=3, turn_limit=6)
    states = []
    while not m.is_over:
        states.append(m.state)
        if m.is_ai_turn():
            m.play_ai_turn()
        else:
            own = m.state.alive(m.state.to_move)[0]
            m.apply(Action(own.mid, math.atan2(-own.y, -own.x), 0.4))
    path = save_replay(record_from_match(m, "human"), "unit", str(tmp_path))
    mp = load_match_positions(path)
    assert mp.error is None and len(mp.states) == len(m.history)
    assert [s.fingerprint() for s in mp.states] == [s.fingerprint() for s in states]
    assert mp.controller(0).startswith("Minimax") and mp.controller(1) == "Human"
    assert "vs" in mp.title() and "\n" in mp.title()
    assert [x.path for x in recent_matches(5, str(tmp_path))] == [path]
    a = Action.from_dict(mp.turns[0]["action"])
    assert same_shot(a, Action(a.marble_id, a.angle + 2 * math.pi, a.power, a.ability))
    assert not same_shot(a, Action(a.marble_id, a.angle + 0.2, a.power, a.ability))
    # a tree can be drawn for any rebuilt position
    tr = trace_decision(mp.states[1], depth=1)
    assert tr.root_player == mp.states[1].to_move and tr.best_child is not None


def test_replay_from_other_settings_is_not_rebuilt(tmp_path):
    import json
    from ai.search_trace import load_match_positions, recent_matches
    from game.match import Match, PlayerSpec
    from game.replay import record_from_match, save_replay
    m = Match("classic", PlayerSpec("human"), PlayerSpec("human"), seed=1, turn_limit=2)
    rec = record_from_match(m)
    rec["config_fingerprint"] = "0000000000000000"          # recorded with different physics
    path = save_replay(rec, "old", str(tmp_path))
    mp = load_match_positions(path)
    assert mp.states == [] and "different game settings" in mp.error
    assert recent_matches(5, str(tmp_path)) == []          # never offered in the Lab
