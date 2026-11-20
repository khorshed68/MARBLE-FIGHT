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
from ui.tree_view import MCTS_ITERS, TreeDrawing  # noqa: F401  (MCTS_ITERS re-exported)
from ui.widgets import BarChart, Table

ALGO_COL = {"minimax": T.P1, "mcts": T.P2, "human": (0.7, 0.7, 0.8, 1)}
ALGO_NAME = {"minimax": "Minimax+AB", "mcts": "MCTS (UCT)", "human": "Human"}
TABS = [("overview", "OVERVIEW"), ("search", "SEARCH METRICS"), ("tree", "SEARCH TREE"), ("arenas", "ARENAS"),
        ("tournaments", "TOURNAMENTS"), ("last", "LAST MATCH"), ("concepts", "HOW THE AI WORKS")]
TREE_DEPTHS = [(2, "2 plies (my shot + reply)"), (1, "1 ply (my shot only)")]
TREE_ALGOS = [("minimax", "Minimax + Alpha-Beta"), ("mcts", "MCTS (UCT)")]


class AlgorithmLab(TreeDrawing, Screen):
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
        cx, cz = -1.15, -0.07
        self._draw_minimap(parent, state, played, best, cx, cz, 0.28)
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
