"""Algorithm Lab: measured statistics and explanations of the AI techniques.

All figures are aggregated from data/match_history.jsonl and data/tournaments/,
which are written only by real matches. With no data an empty state is shown.
"""

from __future__ import annotations

import glob
import math
import json
import os

from direct.gui.DirectGui import DirectFrame
from panda3d.core import LineSegs

from ai.fuzzy_risk import EDGE, POWER, RISK, STOP, FuzzyInputs, FuzzyRiskEvaluator
from analytics.match_logger import clear_history, load_history
from analytics.metrics import aggregate, head_to_head
from config import TOURNAMENT_DIR
from ui import theme as T
from ui.main_menu import Screen
from ui.widgets import BarChart, Table

ALGO_COL = {"minimax": T.P1, "mcts": T.P2, "human": (0.7, 0.7, 0.8, 1)}
ALGO_NAME = {"minimax": "Minimax+AB", "mcts": "MCTS (UCT)", "human": "Human"}
TABS = [("overview", "OVERVIEW"), ("search", "SEARCH METRICS"), ("tree", "SEARCH TREE"), ("arenas", "ARENAS"),
        ("tournaments", "TOURNAMENTS"), ("last", "LAST MATCH"), ("concepts", "HOW THE AI WORKS")]
TREE_DEPTHS = [(2, "2 plies (my shot + reply)"), (1, "1 ply (my shot only)")]
TREE_ALGOS = [("minimax", "Minimax + Alpha-Beta"), ("mcts", "MCTS (UCT)")]
MCTS_ITERS = 40


class AlgorithmLab(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.bg = DirectFrame(parent=self.root, frameColor=(0.02, 0.03, 0.06, 0.9), frameSize=(-3, 3, -1, 1))
        # the Lab is laid out for 16:9 (x = -1.6 .. 1.6); shrink it on narrower windows so nothing is cut off
        try:
            aspect = float(app.getAspectRatio())
        except Exception:
            aspect = 16 / 9
        self.root.setScale(min(1.0, max(0.6, (aspect - 0.03) / 1.62)))
        T.text(self.root, "ALGORITHM LAB", (0, 0.88), 0.07, T.TEXT, "center", "title")
        T.text(self.root, "Measured results only — every number comes from a match that was actually played.",
               (0, 0.83), 0.03, T.MUTED, "center")
        self.tab_btns = {}
        for i, (key, lab) in enumerate(TABS):
            x = (i - (len(TABS) - 1) / 2) * 0.37
            self.tab_btns[key] = T.button(self.root, lab, self.show, extra_args=[key], pos=(x, 0, 0.74),
                                          size=(0.178, 0.035), text_scale=0.026)
        self.tree_opts = {"algo": "minimax", "scene": "trap", "turn": 0, "depth": 2, "fuzzy": True}
        self._recent_matches = None
        self._trace_cache: dict = {}
        self._mcts_cache: dict = {}
        T.button(self.root, "BACK", lambda: app.goto("menu"), pos=(-1.4, 0, -0.92), size=(0.2, 0.04))
        T.button(self.root, "CLEAR HISTORY", self.clear, pos=(1.35, 0, -0.92), size=(0.25, 0.04), accent=T.DANGER)
        self.body = None
        self.reload()
        self.show("overview")

    def reload(self) -> None:
        self.records = load_history()
        self.agg = aggregate(self.records)

    def clear(self) -> None:
        clear_history()
        self.reload()
        self.show(self.current)

    def show(self, key: str) -> None:
        self.current = key
        for k, b in self.tab_btns.items():
            T.set_button_active(b, k == key)
        if self.body is not None:
            self.body.destroy()
        self.body = DirectFrame(parent=self.root, frameColor=(0, 0, 0, 0))
        getattr(self, f"_tab_{key}")(self.body)

    def _empty(self, parent, msg: str) -> None:
        p = T.panel(parent, (-0.9, 0.9, -0.2, 0.2), (0, 0, 0.1))
        T.text(p, "NO DATA YET", (0, 0.06), 0.06, T.WARN, "center", "title")
        T.text(p, msg, (0, -0.04), 0.032, T.MUTED, "center", wordwrap=50)

    # ------------------------------------------------------------------ tabs
    def _tab_overview(self, parent) -> None:
        algos = self.agg["algorithms"]
        if not self.agg["total_matches"]:
            self._empty(parent, "Play an AI-vs-AI match, a Human-vs-AI match or run a tournament; results appear here.")
            return
        h = head_to_head(self.records)
        T.text(parent, f"Matches recorded: {self.agg['total_matches']}     Minimax vs MCTS head-to-head: "
                       f"Minimax {h['minimax']} | MCTS {h['mcts']} | draws {h['draw']} ({h['matches']} matches)",
               (0, 0.63), 0.032, T.TEXT, "center")
        keys = list(algos)
        rows = []
        for k in keys:
            a = algos[k]
            rows.append([ALGO_NAME.get(k, k), a["matches"], a["wins"], a["losses"], a["draws"],
                         T.fmt(a["win_rate"] * 100 if a["win_rate"] is not None else None, 1, "%"),
                         T.fmt(a["avg_decision_time"], 3, " s"), T.fmt(a["max_decision_time"], 3, " s"),
                         T.fmt(a["avg_fuzzy_risk"], 3), T.fmt(a["fallbacks"])])
        Table(parent, ["Agent", "Matches", "Wins", "Losses", "Draws", "Win rate", "Avg time", "Max time",
                       "Avg fuzzy risk", "Fallbacks"], rows, (-1.45, 0.57),
              [0.38, 0.24, 0.2, 0.24, 0.2, 0.26, 0.28, 0.28, 0.32, 0.24], title="Per-agent results",
              colors=[ALGO_COL.get(k, T.TEXT) for k in keys])
        BarChart(parent, "Win rate", [(ALGO_NAME.get(k, k), algos[k]["win_rate"], ALGO_COL.get(k, T.TEXT)) for k in keys],
                 (-1.45, 0.1), 0.92, fmt=lambda v: f"{v * 100:.1f}%", max_value=1.0)
        BarChart(parent, "Average decision time (s)",
                 [(ALGO_NAME.get(k, k), algos[k]["avg_decision_time"], ALGO_COL.get(k, T.TEXT)) for k in keys if k != "human"],
                 (-0.46, 0.1), 0.92, fmt=lambda v: f"{v:.3f}")
        BarChart(parent, "Physics simulations per decision",
                 [(ALGO_NAME.get(k, k), (algos[k]["total_simulations"] / algos[k]["decisions"])
                   if algos[k]["decisions"] and algos[k]["total_simulations"] is not None else None,
                   ALGO_COL.get(k, T.TEXT)) for k in keys if k != "human"],
                 (0.53, 0.1), 0.92, fmt=lambda v: f"{v:.0f}")

    def _tab_search(self, parent) -> None:
        a = self.agg["algorithms"]
        if not ({"minimax", "mcts"} & set(a)):
            self._empty(parent, "No AI decisions recorded yet.")
            return
        mm, mc = a.get("minimax"), a.get("mcts")
        T.text(parent, "n/a = not measured for this agent (different from a measured 0)", (0, 0.66), 0.027,
               T.MUTED, "center")
        p = T.panel(parent, (0, 1.4, -1.22, 0), (-1.45, 0, 0.6), color=T.PANEL_LIGHT)
        T.text(p, "MINIMAX + ALPHA-BETA", (0.03, -0.06), 0.042, T.P1, kind="title")
        if mm:
            tot = (mm["total_nodes"] or 0) + (mm["total_prunes"] or 0)
            rows = [("Decisions", T.fmt(mm["decisions"])),
                    ("Average decision time", T.fmt(mm["avg_decision_time"], 3, " s")),
                    ("Maximum decision time", T.fmt(mm["max_decision_time"], 3, " s")),
                    ("Nodes visited (total)", T.fmt(mm["total_nodes"])),
                    ("Leaves evaluated (total)", T.fmt(mm["total_leaves"])),
                    ("Alpha-beta cutoffs (total)", T.fmt(mm["total_cutoffs"])),
                    ("Branches pruned (total)", T.fmt(mm["total_prunes"])),
                    ("Pruned share of branches", T.fmt(mm["total_prunes"] / tot * 100 if tot and mm["total_prunes"] is not None else None, 1, "%")),
                    ("Average / maximum depth reached", f"{T.fmt(mm['avg_depth'], 2)} / {T.fmt(mm['max_depth'])} plies"),
                    ("Average backed-up evaluation", T.fmt(mm["avg_value"], 2)),
                    ("Physics simulations (total)", T.fmt(mm["total_simulations"])),
                    ("Avg fuzzy risk of chosen shots", T.fmt(mm["avg_fuzzy_risk"], 3)),
                    ("Fallback shots", T.fmt(mm["fallbacks"]))]
            for i, (k, v) in enumerate(rows):
                T.text(p, k, (0.03, -0.14 - i * 0.078), 0.031, T.MUTED)
                T.text(p, v, (1.37, -0.14 - i * 0.078), 0.031, T.TEXT, "right")
        else:
            T.text(p, "No Minimax decisions recorded.", (0.03, -0.15), 0.032, T.MUTED)
        q = T.panel(parent, (0, 1.4, -1.22, 0), (0.05, 0, 0.6), color=T.PANEL_LIGHT)
        T.text(q, "MONTE CARLO TREE SEARCH (UCT)", (0.03, -0.06), 0.042, T.P2, kind="title")
        if mc:
            rows = [("Decisions", T.fmt(mc["decisions"])),
                    ("Average decision time", T.fmt(mc["avg_decision_time"], 3, " s")),
                    ("Maximum decision time", T.fmt(mc["max_decision_time"], 3, " s")),
                    ("Tree iterations (total)", T.fmt(mc["total_iterations"])),
                    ("Physics simulations completed (total)", T.fmt(mc["total_simulations"])),
                    ("Average rollout depth (turns)", T.fmt(mc["avg_rollout_depth"], 2)),
                    ("Avg visit count of selected action", T.fmt(mc["avg_selected_visits"], 1)),
                    ("Avg reward estimate of selected action", T.fmt(mc["avg_value"], 3)),
                    ("Maximum tree depth", T.fmt(mc["max_tree_depth"])),
                    ("Avg fuzzy risk of chosen shots", T.fmt(mc["avg_fuzzy_risk"], 3)),
                    ("Fallback shots", T.fmt(mc["fallbacks"]))]
            for i, (k, v) in enumerate(rows):
                T.text(q, k, (0.03, -0.14 - i * 0.078), 0.031, T.MUTED)
                T.text(q, v, (1.37, -0.14 - i * 0.078), 0.031, T.TEXT, "right")
        else:
            T.text(q, "No MCTS decisions recorded.", (0.03, -0.15), 0.032, T.MUTED)

    def _tab_arenas(self, parent) -> None:
        ar = self.agg["arenas"]
        if not ar:
            self._empty(parent, "No matches recorded yet.")
            return
        rows = []
        for k, v in ar.items():
            res = v["results"]
            rows.append([k, v["matches"], T.fmt(v["avg_turns"], 1), res.get("minimax", 0), res.get("mcts", 0),
                         res.get("human", 0), res.get("draw", 0)])
        t = Table(parent, ["Arena", "Matches", "Avg turns", "Minimax wins", "MCTS wins", "Human wins", "Draws"], rows,
                  (-1.45, 0.66), [0.4, 0.28, 0.3, 0.36, 0.32, 0.32, 0.22], title="Results by arena", row_h=0.05,
                  scale=0.029)
        prow = []
        for v in self.agg["pairings"][:12]:
            wins = ", ".join(f"{k} {n}" for k, n in sorted(v["wins"].items())) or "-"
            prow.append([v["pairing"], v["arena"], v["matches"], T.fmt(v["avg_turns"], 1), wins, v["draws"]])
        Table(parent, ["Pairing", "Arena", "Matches", "Avg turns", "Wins (mirror matches: by seat)", "Draws"], prow,
              (-1.45, 0.66 - t.height - 0.04), [0.62, 0.34, 0.26, 0.28, 0.9, 0.2],
              title="Results by algorithm pairing and arena", row_h=0.047, scale=0.027)

    def _tab_tournaments(self, parent) -> None:
        files = sorted(glob.glob(os.path.join(TOURNAMENT_DIR, "*.json")), reverse=True)[:9]
        if not files:
            self._empty(parent, "Run Tournament Mode (or python -m analytics.tournament) to compare the algorithms "
                                "over many matches.")
            return
        rows = []
        for f in files:
            try:
                d = json.load(open(f, encoding="utf-8"))
            except (OSError, ValueError):
                continue
            h = d.get("head_to_head", {})
            al = d.get("aggregate", {}).get("algorithms", {})
            wr = lambda k: T.fmt(al[k]["win_rate"] * 100, 1, "%") if k in al and al[k]["win_rate"] is not None else "—"
            c = d.get("config", {})
            rows.append([d["tournament_id"], f"{d['completed']}/{d['scheduled']}", c.get("difficulty", ""),
                         ",".join(x[:3] for x in c.get("arenas", [])), f"{h.get('minimax', 0)}-{h.get('mcts', 0)}-{h.get('draw', 0)}",
                         wr("minimax"), wr("mcts"),
                         T.fmt(al.get("minimax", {}).get("avg_decision_time"), 3),
                         T.fmt(al.get("mcts", {}).get("avg_decision_time"), 3)])
        Table(parent, ["Tournament", "Matches", "Budget", "Arenas", "MM-MCTS-Draw", "MM win%", "MCTS win%",
                       "MM time", "MCTS time"], rows, (-1.45, 0.6),
              [0.48, 0.24, 0.24, 0.36, 0.4, 0.28, 0.3, 0.26, 0.26], title="Recent tournaments (data/tournaments)",
              row_h=0.055, scale=0.028)
        T.text(parent, "Markdown reports are written to the reports/ folder.", (0, -0.75), 0.03, T.MUTED, "center")

    def _tab_last(self, parent) -> None:
        dec = self.app.last_match_decisions
        if not dec:
            self._empty(parent, "Finish a match in this session to see its per-turn decision data.")
            return
        p = T.panel(parent, (-1.45, 1.45, -1.3, 0), (0, 0, 0.62), color=T.PANEL_LIGHT)
        T.text(p, "Decision time per turn (bars) and fuzzy risk of the chosen shot (dots)", (-1.4, -0.06), 0.034,
               T.EDGE, kind="title")
        n = len(dec)
        tmax = max((d.get("decision_time") or 0) for d in dec) or 1.0
        x0, x1, y0, y1 = -1.35, 1.35, -1.15, -0.15
        bw = (x1 - x0) / max(n, 1)
        for i, d in enumerate(dec):
            t = d.get("decision_time") or 0.0
            h = (y1 - y0) * 0.85 * t / tmax
            col = T.PLAYER_COLORS[d["player"]]
            DirectFrame(parent=p, frameColor=(col[0], col[1], col[2], 0.8),
                        frameSize=(x0 + i * bw + bw * 0.15, x0 + (i + 1) * bw - bw * 0.15, y0, y0 + h))
            r = d.get("fuzzy_risk")
            if r is not None:
                DirectFrame(parent=p, frameColor=T.WARN, frameSize=(-0.008, 0.008, -0.008, 0.008),
                            pos=(x0 + (i + 0.5) * bw, 0, y0 + (y1 - y0) * 0.85 * r))
        T.text(p, f"max {tmax:.2f} s", (x0, y1 - 0.02), 0.026, T.MUTED)
        T.text(p, f"{n} decisions | cyan = Player 1 | orange = Player 2 | yellow dot = fuzzy risk (0..1 scale)",
               (0, y0 - 0.06), 0.027, T.MUTED, "center")

    # ------------------------------------------------------------------ search tree (live Alpha-Beta trace)
    def _tree_set(self, key, value) -> None:
        self.tree_opts[key] = value
        if key == "scene":
            self.tree_opts["turn"] = 0
        self.show("tree")

    # positions: three fixed demos + the most recent saved matches (rebuilt from their replays)
    def _recent(self):
        if self._recent_matches is None:
            from ai.search_trace import recent_matches
            try:
                self._recent_matches = recent_matches(5, game_cfg=self.app.game_cfg)
            except Exception:                    # a damaged replay folder must never break the Lab
                self._recent_matches = []
        return self._recent_matches

    def _match(self):
        """The selected recorded match (or None for a demo position)."""
        sc = self.tree_opts["scene"]
        if not sc.startswith("match"):
            return None
        ms = self._recent()
        i = int(sc[5:])
        return ms[i] if i < len(ms) else None

    def _scene_state(self):
        from ai.search_trace import demo_positions
        o = self.tree_opts
        m = self._match()
        if m is not None:
            o["turn"] = max(0, min(o["turn"], len(m.states) - 1))
            return m.states[o["turn"]]
        if o["scene"].startswith("match"):        # match no longer available
            o["scene"] = "trap"
        return {key: mk for key, _, mk in demo_positions()}[o["scene"]]()

    def _scene_key(self):
        o = self.tree_opts
        m = self._match()
        return (m.path, o["turn"]) if m is not None else (o["scene"],)

    def _trace(self):
        from ai.search_trace import trace_decision
        o = self.tree_opts
        state = self._scene_state()
        k = self._scene_key() + (o["depth"], o["fuzzy"])
        if k not in self._trace_cache:
            self._trace_cache[k] = trace_decision(state, depth=o["depth"], root_width=4, inner_width=3,
                                                  use_fuzzy=o["fuzzy"], game_cfg=self.app.game_cfg)
        return self._trace_cache[k]

    def _stepper(self):
        from ai.search_trace import MCTSStepper
        o = self.tree_opts
        state = self._scene_state()
        k = self._scene_key() + (o["fuzzy"],)
        if k not in self._mcts_cache:
            self._mcts_cache[k] = MCTSStepper(state, iterations=MCTS_ITERS, candidate_width=4, rollout_depth=1,
                                              seed=7, use_fuzzy=o["fuzzy"], game_cfg=self.app.game_cfg)
        return self._mcts_cache[k]

    def _mcts_step(self, n: int) -> None:
        self._stepper().run(n)
        self.show("tree")

    def _mcts_reset(self) -> None:
        self._scene_state()
        self._mcts_cache.pop(self._scene_key() + (self.tree_opts["fuzzy"],), None)
        self.show("tree")

    def _scene_banner(self, parent, mt, demo_title: str, to_move: int) -> None:
        """Large panel above the tree that says exactly which position the tree belongs to."""
        x, z = -0.72, 0.67
        w, h = 0.76, 0.24
        p = T.panel(parent, (0, w, -h, 0), (x, 0, z), color=(0.05, 0.08, 0.15, 0.96))
        if mt is None:
            DirectFrame(parent=p, frameColor=T.MUTED, frameSize=(0, 0.008, -h, 0))
            T.text(p, "DEMO POSITION", (0.03, -0.05), 0.036, T.MUTED, kind="title")
            T.text(p, demo_title, (0.03, -0.1), 0.027, T.TEXT, wordwrap=27)
            T.text(p, "a fixed teaching example, not a match you played", (0.03, -0.205), 0.022, T.MUTED)
            return
        i = int(self.tree_opts["scene"][5:])
        r = mt.record
        when = str(r.get("created", "?"))
        day, _, clock = when.partition(" ")
        mode = {"arena": "AI vs AI", "ai_vs_ai": "AI vs AI", "human": "Human vs AI", "pvp": "Human vs Human",
                "tournament": "Tournament"}.get(r.get("source", ""), "Match")
        res = r.get("result", {})
        sc = res.get("score") or ["?", "?"]
        try:
            from game.arena import get_arena
            arena = get_arena(r["arena"]).name
        except KeyError:
            arena = r["arena"]
        turn = self.tree_opts["turn"]
        DirectFrame(parent=p, frameColor=T.GOLD, frameSize=(0, 0.008, -h, 0))
        T.text(p, "YOUR LAST MATCH" if i == 0 else f"RECENT MATCH {i + 1}", (0.03, -0.05), 0.036, T.GOLD,
               kind="title")
        T.text(p, f"{day}  {clock}", (w - 0.03, -0.05), 0.032, (1, 1, 1, 1), "right")
        players = mt.players() + ("  (tournament)" if mode == "Tournament" else "")
        T.text(p, f"{players}  |  {arena}", (0.03, -0.1), 0.027, T.TEXT)
        T.text(p, f"Final score  P1 {sc[0]} : {sc[1]} P2   |   {len(mt.states)} shots", (0.03, -0.147), 0.027,
               T.TEXT)
        spec = r["specs"][to_move]
        who = "Human" if spec["kind"] == "human" else             f"{spec['kind'].replace('minimax', 'Minimax').replace('mcts', 'MCTS')}, {spec.get('difficulty', '?')}"
        T.text(p, f"Turn {turn + 1} of {len(mt.states)}:  Player {to_move + 1} ({who}) to move",
               (0.03, -0.2), 0.027, T.PLAYER_COLORS[to_move])

    @staticmethod
    def _played_detail(turn: dict) -> str:
        d = turn.get("decision")
        if not d:
            return "a human shot (no search)"
        if d.get("fallback"):
            return "a fallback shot (the AI ran out of time)"
        s = d.get("stats", {})
        if "iterations" in s:
            return f"{s.get('iterations')} iterations, chosen shot visited {s.get('selected_visits')} times"
        return (f"value {s.get('evaluation')}, depth {s.get('depth_reached')}, "
                f"{s.get('nodes_visited')} nodes, {s.get('branches_pruned')} pruned")

    def _tab_tree(self, parent) -> None:
        from ai.search_trace import Action, action_label, demo_positions, same_shot
        from game.arena import get_arena
        from ui.widgets import Cycler
        o = self.tree_opts
        scenes = [(k, title.split(" (")[0] if k == "trap" else title.replace("Opening position", "Opening"))
                  for k, title, _ in demo_positions()]
        titles = {k: title for k, title, _ in demo_positions()}
        for i, m in enumerate(self._recent()):
            key = f"match{i}"
            scenes.append((key, "Last match" if i == 0 else f"Recent match {i + 1}"))
            titles[key] = m.title()
        state = self._scene_state()               # also falls back to a demo if a match disappeared
        mt = self._match()
        mcts = o["algo"] == "mcts"
        rows = [("Algorithm", TREE_ALGOS, o["algo"], "algo"),
                ("Position", scenes, o["scene"], "scene")]
        if mt is not None:
            turns = [(i, f"Turn {i + 1} / {len(mt.states)} | P{tt['player'] + 1} " +
                      ("Human" if mt.record["specs"][tt["player"]]["kind"] == "human" else
                       mt.record["specs"][tt["player"]]["kind"].replace("minimax", "Minimax").replace("mcts", "MCTS")))
                     for i, tt in enumerate(mt.turns)]
            rows.append(("Turn", turns, o["turn"], "turn"))
        rows.append(("Fuzzy", [(True, "risk penalty on"), (False, "risk penalty off")], o["fuzzy"], "fuzzy"))
        if not mcts:
            rows.append(("Depth", TREE_DEPTHS, o["depth"], "depth"))
        for i, (lab, opts, val, key) in enumerate(rows):
            Cycler(parent, lab, opts, [v for v, _ in opts].index(val), (-1.5, 0.64 - i * 0.092),
                   lambda v, key=key: self._tree_set(key, v), width=0.7, label_w=0.2, scale=0.028)
        if mcts:
            T.text(parent, f"Budget: {MCTS_ITERS} iterations, 1-turn rollouts, seed 7",
                   (-1.5, 0.64 - len(rows) * 0.092 + 0.01), 0.024, T.MUTED)
        tr = self._trace()
        if mcts:
            st = self._stepper()
            bn = st.best()
            best = bn.action if bn is not None else None
        else:
            best = tr.nodes[tr.best_child].action if tr.best_child is not None else None
        played = Action.from_dict(mt.turns[o["turn"]]["action"]) if mt is not None else None
        # ---- mini map of the position (top view, same coordinates as the physics)
        arena = get_arena(state.arena_id)
        cx, cz, s = -1.15, -0.07, 0.25 / arena.radius
        mp = T.panel(parent, (-0.28, 0.28, -0.28, 0.28), (cx, 0, cz), color=T.PANEL_LIGHT)
        ls = LineSegs()
        ls.setThickness(2)
        ls.setColor(T.EDGE[0], T.EDGE[1], T.EDGE[2], 0.9)
        for k in range(73):
            a = 2 * math.pi * k / 72
            (ls.moveTo if k == 0 else ls.drawTo)(arena.radius * s * math.cos(a), 0, arena.radius * s * math.sin(a))
        for b in arena.obstacle_positions(state.clock):
            bx, by, br = b
            ls.setColor(1.0, 0.75, 0.3, 0.8)
            for k in range(17):
                a = 2 * math.pi * k / 16
                (ls.moveTo if k == 0 else ls.drawTo)((bx + br * math.cos(a)) * s, 0, (by + br * math.sin(a)) * s)
        for act, col in ((played, (1, 1, 1, 0.95)), (best, T.GOLD)):
            if act is not None:
                m = state.marbles[act.marble_id]
                ls.setColor(col[0], col[1], col[2], col[3])
                ls.moveTo(m.x * s, 0, m.y * s)
                ls.drawTo((m.x + 4.0 * math.cos(act.angle)) * s, 0, (m.y + 4.0 * math.sin(act.angle)) * s)
        mp.attachNewNode(ls.create())
        for m in state.marbles:
            if m.alive:
                r = max(0.012, 0.45 * s * 1.3)
                DirectFrame(parent=mp, frameColor=T.PLAYER_COLORS[m.owner], frameSize=(-r, r, -r, r),
                            pos=(m.x * s, 0, m.y * s))
                T.text(mp, f"M{m.mid}", (m.x * s + 0.02, m.y * s + 0.015), 0.022, T.TEXT)
        if mt is not None:
            T.text(parent, "white line = shot played in the match", (cx, -0.385), 0.024, (1, 1, 1, 1), "center")
            T.text(parent, "gold line = this tree's choice", (cx, -0.425), 0.024, T.GOLD, "center")
        else:
            T.text(parent, f"Player {tr.root_player + 1} to move | gold line = chosen shot", (cx, -0.405), 0.024,
                   T.MUTED, "center")
        self._scene_banner(parent, mt, titles[o["scene"]], tr.root_player)
        # ---- what the search did (all measured from this run)
        if mcts:
            lines = [(f"Iterations: {st.iterations} / {MCTS_ITERS}", T.TEXT)]
            if bn is not None:
                lines += [(f"Choice now (most visits): {self._short_label(best)}", T.GOLD),
                          (f"   {bn.visits} visits, mean reward {bn.mean:.3f}", T.GOLD)]
            else:
                lines.append(("No shot explored yet", T.MUTED))
            mm = tr.nodes[tr.best_child].action if tr.best_child is not None else None
            lines += [(f"Physics simulations so far: {st.agent.simulations}", T.TEXT),
                      (f"Minimax depth {tr.depth} chooses: {self._short_label(mm)}", T.P1)]
            note = ("Same agent code as in a match (select, expand, rollout, backpropagate); a small budget so "
                    "each iteration can be followed. Q = mean reward of the player who made that shot.")
        else:
            stt = tr.stats
            same = abs(tr.value - tr.minimax_value) < 1e-9
            lines = [
                (f"Chosen shot: {self._short_label(best)}", T.GOLD),
                (f"Value {tr.value:.2f} | plain Minimax {tr.minimax_value:.2f}" +
                 (" (identical)" if same else " (DIFFERENT)"), T.GOOD if same else T.DANGER),
                (f"Nodes {stt.nodes_visited} (without pruning {tr.minimax_nodes}) | pruned {stt.branches_pruned}"
                 f" | cutoffs {stt.cutoffs}", T.TEXT),
            ]
            note = ("Narrow beam for display: 4 shots at the root, 3 replies each. The real agent uses wider "
                    "beams and iterative deepening; the algorithm is the same function.")
        if mt is not None:
            tt = mt.turns[o["turn"]]
            lines = (lines[1:3] if mcts else lines[:2]) + [
                (f"In the match: P{tt['player'] + 1} {mt.controller(o['turn'])}", T.TEXT),
                (f"   played {action_label(played)}", (1, 1, 1, 1)),
                (f"   {self._played_detail(tt)}", T.MUTED),
                (("Same shot as this tree" if same_shot(best, played) else
                  "Different from this tree" if best is not None else "Run iterations to compare"),
                 T.GOOD if same_shot(best, played) else T.WARN),
            ]
            if not tt.get("decision"):
                note = ("A human played this turn. The tree shows what the AI would consider in the same "
                        "position, rebuilt exactly from the replay.")
            else:
                note = ("The match searched with its full budget (wider beam, deeper search); this tree uses a small "
                        "one so it can be drawn, so its choice may differ from the shot that was played.")
            if mt.error:
                note = f"Only the first {len(mt.states)} turns could be rebuilt: {mt.error}"
        y0 = -0.49
        for i, (txt, col) in enumerate(lines):
            T.text(parent, txt, (-1.5, y0 - i * 0.046), 0.025, col)
        T.text(parent, note, (-1.5, y0 - len(lines) * 0.046 - 0.01), 0.022, T.MUTED, wordwrap=31)
        if mcts:
            self._draw_mcts(parent, st)
            return
        self._draw_tree(parent, tr)

    @staticmethod
    def _short_label(a) -> str:
        from ai.search_trace import action_label
        return action_label(a)

    @staticmethod
    def _two_lines(a) -> tuple[str, str]:
        from ai.search_trace import action_label
        lab = action_label(a)
        if " | " in lab:
            head, tail = lab.split(" | ", 1)
            return head, tail
        head, _, power = lab.rpartition(" ")
        return head, f"Normal {power}"

    def _draw_mcts(self, parent, st) -> None:
        """Top two levels of the real MCTS tree, with N, Q, UCT and the path of the last iteration."""
        x0, x1 = -0.7, 1.5
        root = st.root
        rp = st.player
        rows = {0: 0.56, 1: 0.13, 2: -0.27}
        heights = {0: 0.17, 1: 0.29, 2: 0.22}
        untried_root = list(root.untried or [])
        level1 = [(c, False) for c in root.children] + [(a, True) for a in untried_root]
        units = [max(1, len(c.children)) if not ghost else 1 for c, ghost in level1]
        slot = (x1 - x0) / max(1, sum(units))
        last = st.events[-1] if st.events else None
        on_path = set(id(n) for n in last.path) if last else set()
        new = id(last.expanded) if (last and last.expanded is not None) else None
        best = st.best()
        pos = {}
        acc = x0
        for (c, ghost), u in zip(level1, units):
            pos[id(c)] = acc + u * slot / 2
            if not ghost:
                for j, g in enumerate(c.children):
                    pos[id(g)] = acc + (j + 0.5) * slot
            acc += u * slot
        pos[id(root)] = (x0 + x1) / 2
        # edges: brightness and line count grow with the share of visits
        ls = LineSegs()
        ls.setThickness(2)

        def edge(parent_node, child, ply, ghost=False):
            px, pz = pos[id(parent_node)], rows[ply - 1] - heights[ply - 1] / 2
            cx, cz = pos[id(child)], rows[ply] + heights[ply] / 2
            if ghost:
                ls.setColor(0.5, 0.5, 0.55, 0.45)
                for k in range(0, 11, 2):
                    f0, f1 = k / 11, (k + 1) / 11
                    ls.moveTo(px + (cx - px) * f0, 0, pz + (cz - pz) * f0)
                    ls.drawTo(px + (cx - px) * f1, 0, pz + (cz - pz) * f1)
                return
            share = child.visits / max(1, parent_node.visits)
            if id(child) in on_path and id(parent_node) in on_path:
                col = T.GOOD
            elif parent_node is root and child is best:
                col = T.GOLD
            else:
                col = (0.55, 0.62, 0.78, 1.0)
            ls.setColor(col[0], col[1], col[2], 0.3 + 0.7 * share)
            for k in range(1 + int(round(3 * share))):
                off = (k - 0.5 * int(round(3 * share))) * 0.004
                ls.moveTo(px + off, 0, pz)
                ls.drawTo(cx + off, 0, cz)

        for c, ghost in level1:
            edge(root, c, 1, ghost)
            if not ghost:
                for g in c.children:
                    edge(c, g, 2)
        parent.attachNewNode(ls.create())

        def box(node, ply, w, ghost=False):
            x, z, h = pos[id(node)], rows[ply], heights[ply]
            if ply == 1 and not ghost and node is best:
                DirectFrame(parent=parent, frameColor=T.GOLD, frameSize=(-w / 2 - 0.008, w / 2 + 0.008,
                                                                       -h / 2 - 0.008, h / 2 + 0.008), pos=(x, 0, z))
            elif not ghost and id(node) in on_path:
                DirectFrame(parent=parent, frameColor=T.GOOD, frameSize=(-w / 2 - 0.006, w / 2 + 0.006,
                                                                       -h / 2 - 0.006, h / 2 + 0.006), pos=(x, 0, z))
            colr = (0.07, 0.07, 0.09, 0.95) if ghost else T.PANEL_LIGHT
            b = T.panel(parent, (-w / 2, w / 2, -h / 2, h / 2), (x, 0, z), color=colr)
            if not ghost:
                mover = rp if node is root else node.mover
                bar = T.PLAYER_COLORS[mover]
                DirectFrame(parent=b, frameColor=bar, frameSize=(-w / 2, w / 2, h / 2 - 0.008, h / 2))
            return b

        def write(b, h, rows_txt):
            y = h / 2 - 0.034
            for txt, c, sc in rows_txt:
                T.text(b, txt, (0, y), sc, c, "center")
                y -= sc + 0.011

        # root
        lim = st.widening_limit(root)
        b = box(root, 0, 0.66)
        write(b, heights[0], [(f"ROOT | Player {rp + 1} to move", T.PLAYER_COLORS[rp], 0.028),
                              (f"N = {root.visits} iterations", T.GOLD, 0.032),
                              (f"children {len(root.children)} / widening limit {lim} | untried {len(untried_root)}",
                               T.MUTED, 0.021),
                              ("plays the MOST VISITED child", T.MUTED, 0.021)])
        for (c, ghost), u in zip(level1, units):
            w = min(0.5, u * slot - 0.02)
            if ghost:
                l1, l2 = self._two_lines(c)
                bb = box(c, 1, w, ghost=True)
                write(bb, heights[1], [("NOT EXPANDED YET", T.MUTED, 0.022), (l1, T.MUTED, 0.021),
                                       (l2, T.MUTED, 0.02), ("progressive widening", T.MUTED, 0.019),
                                       ("opens it when N grows", T.MUTED, 0.019)])
                continue
            l1, l2 = self._two_lines(c.action)
            bb = box(c, 1, w)
            head = ("NEW  " if id(c) == new else "") + f"P{c.mover + 1}: {l1}"
            write(bb, heights[1], [
                (head, T.GOOD if id(c) == new else T.PLAYER_COLORS[c.mover], 0.025), (l2, T.TEXT, 0.023),
                (f"N {c.visits}   Q {c.mean:.3f}", T.GOLD if c is best else T.TEXT, 0.027),
                (f"UCT {st.uct(c):.3f}" if c.visits else "UCT +inf", T.TEXT, 0.022),
                (f"risk edge {c.edge:+.1f}", T.MUTED, 0.021),
                (f"{len(c.children)} replies | {len(c.untried or [])} untried", T.MUTED, 0.021)])
            for g in c.children:
                gw = slot - 0.012
                gb = box(g, 2, gw)
                gl1, gl2 = self._two_lines(g.action)
                deeper = sum(1 for _ in g.children)
                big = 1.0 if slot < 0.2 else 1.25          # few replies -> wide boxes -> larger text
                rows_g = [("NEW" if id(g) == new else f"P{g.mover + 1} reply",
                           T.GOOD if id(g) == new else T.PLAYER_COLORS[g.mover], 0.018 * big),
                          (gl1.split(" ", 1)[-1] if slot < 0.2 else gl1, T.TEXT, 0.018 * big),
                          (gl2.replace(" 100%", "").replace("Normal", "Norm") if slot < 0.2 else gl2, T.TEXT,
                           0.017 * big),
                          (f"N {g.visits}", T.GOLD, 0.024 * big), (f"Q {g.mean:.2f}", T.TEXT, 0.024 * big)]
                if deeper:
                    rows_g.append((f"+{deeper} deeper", T.MUTED, 0.017 * big))
                write(gb, heights[2], rows_g)
        # ---- the last iteration, phase by phase
        y = -0.47
        if last is None:
            T.text(parent, "Press NEXT ITERATION: each press runs one full MCTS iteration and shows its 4 phases.",
                   ((x0 + x1) / 2, y), 0.027, T.WARN, "center")
        else:
            names = []
            for n in last.path[1:]:
                l1, l2 = self._two_lines(n.action)
                names.append(f"P{n.mover + 1} {l1} ({l2})")
            sel = " > ".join(["root"] + names[:-1] if last.expanded is not None else ["root"] + names)
            if len(last.path) > 3:
                sel += f"   [reached depth {len(last.path) - 1}; deeper levels are not drawn]"
            exp = names[-1] if last.expanded is not None else "nothing (terminal or fully expanded)"
            own = last.reward
            texts = [
                (f"ITERATION {last.index}", T.GOOD, 0.03),
                (f"1 Selection (highest UCT at each level): {sel}", T.TEXT, 0.023),
                (f"2 Expansion: {exp}", T.GOOD, 0.023),
                (("3 Terminal position: exact result " if last.terminal else
                  f"3 Rollout: {last.rollout_turns} quick turn(s), then evaluation -> ") +
                 f"reward {own:.3f} for Player {rp + 1}", T.TEXT, 0.023),
                (f"4 Backpropagation: every node on the path gets N + 1 and W + {own:.3f} (Player {rp + 1}'s shots) "
                 f"or W + {1 - own:.3f} (Player {2 - rp}'s shots)", T.TEXT, 0.023),
            ]
            for txt, c, sc in texts:
                T.text(parent, txt, (x0, y), sc, c, "left", wordwrap=95)
                y -= sc + 0.017
        bx = [(-0.45, "NEXT ITERATION", lambda: self._mcts_step(1)), (0.05, "+10", lambda: self._mcts_step(10)),
              (0.5, "RUN ALL", lambda: self._mcts_step(MCTS_ITERS)), (0.95, "RESET", self._mcts_reset)]
        for x, lab, cmd in bx:
            btn = T.button(parent, lab, cmd, pos=(x, 0, -0.8), size=(0.2, 0.035), text_scale=0.026,
                           enabled=not (st.done and lab != "RESET"))
        T.text(parent, "green = path of the last iteration   gold = current choice   grey dashed = not expanded yet"
                       "   brighter / more lines = more visits", ((x0 + x1) / 2, -0.875), 0.021, T.MUTED, "center")

    def _draw_tree(self, parent, tr) -> None:
        nodes = tr.nodes
        x0, x1 = -0.7, 1.5
        rows = {0: 0.56, 1: 0.12 if tr.depth == 2 else -0.05, 2: -0.36}
        units: dict[int, int] = {}

        def count(n):
            ch = nodes[n].children
            units[n] = max(1, sum(count(c) for c in ch)) if ch else 1
            return units[n]

        count(0)
        slot = (x1 - x0) / units[0]
        pos: dict[int, float] = {}

        def place(n, left):
            pos[n] = left + units[n] * slot / 2
            acc = left
            for c in nodes[n].children:
                place(c, acc)
                acc += units[c] * slot

        place(0, x0)
        pv = set(tr.principal_variation())
        rp = tr.root_player
        heights = {0: 0.15, 1: 0.27 if tr.depth == 2 else 0.25, 2: 0.25}
        # edges first (drawn under the boxes)
        ls = LineSegs()
        ls.setThickness(2)
        for n in nodes:
            if n.parent is None:
                continue
            px, pz = pos[n.parent], rows[nodes[n.parent].ply] - heights[nodes[n.parent].ply] / 2
            cx, cz = pos[n.nid], rows[n.ply] + heights[n.ply] / 2
            if n.pruned:
                ls.setColor(T.DANGER[0], T.DANGER[1], T.DANGER[2], 0.55)
                for k in range(0, 11, 2):                      # dashed
                    f0, f1 = k / 11, (k + 1) / 11
                    ls.moveTo(px + (cx - px) * f0, 0, pz + (cz - pz) * f0)
                    ls.drawTo(px + (cx - px) * f1, 0, pz + (cz - pz) * f1)
                continue
            col = T.GOLD if (n.nid in pv and n.parent in pv) else (0.55, 0.62, 0.78, 0.7)
            ls.setColor(col[0], col[1], col[2], col[3])
            ls.moveTo(px, 0, pz)
            ls.drawTo(cx, 0, cz)
        parent.attachNewNode(ls.create())

        def fv(v):
            if v is None:
                return "-"
            if v == float("inf"):
                return "+inf"
            if v == float("-inf"):
                return "-inf"
            return f"{v:+.1f}"

        for n in nodes:
            x, z, h = pos[n.nid], rows[n.ply], heights[n.ply]
            w = min(0.62 if n.ply == 0 else 0.5, units[n.nid] * slot - 0.02)
            if n.nid in pv:
                DirectFrame(parent=parent, frameColor=T.GOLD, frameSize=(-w / 2 - 0.008, w / 2 + 0.008,
                                                                       -h / 2 - 0.008, h / 2 + 0.008), pos=(x, 0, z))
            mover = nodes[n.parent].state.to_move if n.parent is not None else None
            if n.pruned:
                box = T.panel(parent, (-w / 2, w / 2, -h / 2, h / 2), (x, 0, z), color=(0.07, 0.07, 0.09, 0.95))
                l1, l2 = self._two_lines(n.action)
                rows_txt = [("PRUNED", T.DANGER, 0.027), (l1, T.MUTED, 0.021), (l2, T.MUTED, 0.021),
                            ("never simulated", T.MUTED, 0.02)]
            else:
                col = T.PLAYER_COLORS[rp if mover is None else mover]      # bar = who took this shot
                box = T.panel(parent, (-w / 2, w / 2, -h / 2, h / 2), (x, 0, z), color=T.PANEL_LIGHT)
                DirectFrame(parent=box, frameColor=col, frameSize=(-w / 2, w / 2, h / 2 - 0.008, h / 2))
                if n.parent is None:
                    rows_txt = [(f"ROOT | MAX | Player {rp + 1} to move", T.PLAYER_COLORS[rp], 0.03),
                                (f"value {fv(n.value)}", T.GOLD, 0.038),
                                ("takes the LARGEST child value", T.MUTED, 0.024)]
                elif n.leaf:
                    l1, l2 = self._two_lines(n.action)
                    rows_txt = [(f"P{mover + 1}: {l1}", T.PLAYER_COLORS[mover], 0.022), (l2, T.TEXT, 0.021),
                                (f"eval {fv(n.value)}", T.TEXT, 0.022), (f"risk edge {fv(n.edge)}", T.MUTED, 0.02),
                                (f"= {fv(n.backed_up)}", T.GOLD if n.nid in pv else T.TEXT, 0.028)]
                    if n.state.winner is not None:
                        rows_txt[2] = ("WIN" if n.state.winner == rp else ("DRAW" if n.state.winner < 0 else "LOSS"),
                                       T.GOOD if n.state.winner == rp else T.DANGER, 0.02)
                else:
                    l1, l2 = self._two_lines(n.action)
                    kind = "MAX" if n.is_max else "MIN"
                    rows_txt = [(f"P{mover + 1}: {l1}", T.PLAYER_COLORS[mover], 0.026), (l2, T.TEXT, 0.024),
                                (f"risk edge {fv(n.edge)}", T.MUTED, 0.022),
                                (f"{kind} node, window ({fv(n.alpha)}, {fv(n.beta)})", T.MUTED, 0.021),
                                (f"best reply {fv(n.value)}  ->  {fv(n.backed_up)}", T.GOLD if n.nid in pv else T.TEXT,
                                 0.026)]
                    if n.cutoff:
                        rows_txt.append((f"CUTOFF: {n.pruned_count} skipped", T.DANGER, 0.024))
            y = h / 2 - 0.035
            grow = 1.3 if (tr.depth == 1 and n.ply == 1) else 1.0     # wide boxes when only one ply is drawn
            for txt, c, sc in rows_txt:
                sc *= grow
                T.text(box, txt, (0, y), sc, c, "center")
                y -= sc + 0.012
        # legend
        T.text(parent, "gold = principal variation (best line for both players)    red dashed = pruned by alpha-beta"
                       "    value shown = risk edge + child value", ((x0 + x1) / 2, -0.58), 0.025, T.MUTED, "center")
        T.text(parent, "A cutoff happens when alpha >= beta: the opponent already has a reply that makes this shot "
                       "worse than one we found earlier, so its other replies cannot change the decision.",
               ((x0 + x1) / 2, -0.635), 0.025, T.MUTED, "center", wordwrap=80)

    def _tab_concepts(self, parent) -> None:
        txt = [
            ("Minimax + Alpha-Beta", T.P1,
             "Searches my shot, the opponent's best reply, my answer ... (one ply = one full turn with physics). "
             "Leaves are scored by the heuristic; alpha-beta skips replies that cannot change the decision. "
             "Iterative deepening keeps the last complete depth if the budget runs out."),
            ("MCTS (UCT)", T.P2,
             "Repeats select -> expand -> rollout -> backpropagate. UCT = W/N + c*sqrt(ln Np / N) balances good and "
             "unexplored shots; progressive widening adds candidate shots as visits grow. The most-visited shot wins."),
            ("Fuzzy risk", T.WARN,
             "Inputs (edge distance, power, aim vs boundary, predicted stop margin, enemy threat, friction) are "
             "fuzzified, 14 rules fire (min/max), outputs are aggregated and defuzzified by centroid into risk 0..1."),
            ("History-Based Adaptive AI", T.GOOD,
             "Counts the opponent's aggressive/defensive shots, power and targeted marbles; aggressive opponents make "
             "the agent weigh edge safety more and protect the marble that is targeted most."),
        ]
        y = 0.62
        for head, col, body in txt:
            T.text(parent, head, (-1.45, y), 0.036, col, kind="title")
            T.text(parent, body, (-1.45, y - 0.055), 0.031, T.TEXT, wordwrap=44)
            y -= 0.33
        # membership function plots (actual sets from ai/fuzzy_risk.py)
        p = T.panel(parent, (0, 1.35, -1.25, 0), (0.1, 0, 0.66), color=T.PANEL_LIGHT)
        T.text(p, "Fuzzy membership functions (from ai/fuzzy_risk.py)", (0.03, -0.05), 0.03, T.EDGE, kind="title")
        plots = [("edge distance (0-10)", EDGE, 0, 10), ("power (0-1.4)", POWER, 0, 1.4),
                 ("stop margin (0-8)", STOP, 0, 8), ("output risk (0-1)", RISK, 0, 1)]
        cols = [T.P1, T.WARN, T.GOOD, T.DANGER]
        for i, (name, sets, lo, hi) in enumerate(plots):
            oy = -0.32 - i * 0.24
            T.text(p, name, (0.03, oy + 0.17), 0.025, T.MUTED)
            ls = LineSegs()
            ls.setThickness(2)
            for j, (sn, s) in enumerate(sets.items()):
                c = cols[j % 4]
                ls.setColor(c[0], c[1], c[2], 1)
                for k in range(61):
                    x = lo + (hi - lo) * k / 60
                    px, py = 0.05 + 0.8 * k / 60, oy + 0.13 * s.mu(x)
                    (ls.moveTo if k == 0 else ls.drawTo)(px, 0, py)
                T.text(p, sn, (0.9, oy + 0.11 - j * 0.033), 0.022, c)
            p.attachNewNode(ls.create())
        ev = FuzzyRiskEvaluator().evaluate(FuzzyInputs(1.2, 1.0, 0.8, 1.5, 0.6, 1.0), explain=True)
        T.text(p, f"Example input (edge 1.2, power 1.0, aim 0.8 outward, stop 1.5, threat 0.6): risk = {ev.risk:.3f}",
               (0.03, -1.2), 0.022, T.TEXT)
