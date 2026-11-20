"""Tournament Mode UI. Matches run in worker processes via analytics.tournament;
this screen only shows real progress and the measured outcome."""

from __future__ import annotations

import threading

from direct.gui.DirectGui import DirectFrame

from analytics.tournament import TournamentConfig, run_tournament
from ui import theme as T
from ui.main_menu import DIFFS, ON_OFF, Screen, arena_options
from ui.widgets import Cycler, Table, index_of

PAIRINGS = [("mm_mcts", "Minimax vs MCTS"), ("all", "All pairings (MM-MCTS, MM-MM, MCTS-MCTS)"),
            ("mm_mm", "Minimax vs Minimax"), ("mcts_mcts", "MCTS vs MCTS")]
PAIR_MAP = {"mm_mcts": [("minimax", "mcts")], "mm_mm": [("minimax", "minimax")], "mcts_mcts": [("mcts", "mcts")],
            "all": [("minimax", "mcts"), ("minimax", "minimax"), ("mcts", "mcts")]}


class TournamentScreen(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.bg = DirectFrame(parent=self.root, frameColor=(0.02, 0.03, 0.06, 0.82), frameSize=(-3, 3, -1, 1))
        self.header("TOURNAMENT MODE", "Plays many complete AI-vs-AI matches automatically and records real results.")
        p = T.panel(self.root, (0, 1.3, -1.3, 0), (-1.45, 0, 0.66))
        self.p = p
        st = app.settings
        arenas = [("all", "Rotate all arenas")] + arena_options()
        self.c = {}
        rows = [("matches", "Number of matches", [(n, str(n)) for n in (2, 4, 6, 10, 12, 20, 30)], 6),
                ("pairing", "Algorithm pairings", PAIRINGS, "mm_mcts"),
                ("arenas", "Arena rotation", arenas, "all"),
                ("difficulty", "Computation budget", DIFFS, "easy"),
                ("rotate", "Alternate starting player", ON_OFF, True),
                ("seed", "Base random seed", [(s, str(s)) for s in (1, 7, 42, 2024)], 42),
                ("fuzzy", "Fuzzy risk", ON_OFF, st["use_fuzzy"]),
                ("adapt", "History-Based Adaptive AI", ON_OFF, st["use_opponent_model"])]
        for i, (key, lab, opts, val) in enumerate(rows):
            self.c[key] = Cycler(p, lab, opts, index_of(opts, val), (0.05, -0.08 - i * 0.095), None, width=1.2,
                                 label_w=0.5, scale=0.036)
        T.text(p, "Budget presets set Minimax depth/width and MCTS iterations.", (0.05, -0.86), 0.026, T.MUTED)
        T.text(p, "Matches run in parallel worker processes.", (0.05, -0.9), 0.026, T.MUTED)
        self.run_btn = T.button(p, "RUN TOURNAMENT", self.run, pos=(0.65, 0, -1.0), size=(0.4, 0.05), accent=T.GOLD)
        T.set_button_active(self.run_btn, True)
        self.cancel_btn = T.button(p, "CANCEL", self.cancel, pos=(0.65, 0, -1.11), size=(0.4, 0.04), accent=T.DANGER)
        T.button(p, "BACK", self.back, pos=(0.65, 0, -1.21), size=(0.4, 0.04))
        self.out = DirectFrame(parent=self.root, frameColor=(0, 0, 0, 0))
        self.status = T.text(self.root, "Configure and press RUN TOURNAMENT.", (-0.05, 0.6), 0.034, T.TEXT)
        self.bar_bg = DirectFrame(parent=self.root, frameColor=(0.1, 0.12, 0.18, 1), frameSize=(-0.05, 1.45, 0.52, 0.55))
        self.bar = DirectFrame(parent=self.root, frameColor=T.P1, frameSize=(-0.05, -0.05, 0.52, 0.55))
        self.lines: list[str] = []
        self.log_txt = T.text(self.root, "", (-0.05, 0.46), 0.028, T.TEXT)
        self.thread = None
        self.cancelled = False
        self.progress = (0, 0)
        self.result = None
        self.error = None
        self.task = app.taskMgr.add(self._update, "tournament-ui")

    def run(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        a = self.c["arenas"].get()
        arenas = ["classic", "neon", "magnetic"] if a == "all" else [a]
        cfg = TournamentConfig(matches=self.c["matches"].get(), pairings=PAIR_MAP[self.c["pairing"].get()],
                               arenas=arenas, difficulty=self.c["difficulty"].get(), rotate_start=self.c["rotate"].get(),
                               base_seed=self.c["seed"].get(), use_fuzzy=self.c["fuzzy"].get(),
                               use_opponent_model=self.c["adapt"].get())
        self.cancelled = False
        self.result = None
        self.error = None
        self.lines = []
        self.progress = (0, cfg.matches)
        self.out.destroy()
        self.out = DirectFrame(parent=self.root, frameColor=(0, 0, 0, 0))

        def prog(i, n, rec):
            s = rec["summary"]
            self.progress = (i, n)
            self.lines.append(f"#{rec['job']['index']:>2} {rec['job']['p1']:>7} vs {rec['job']['p2']:<7} "
                              f"{s['arena']:<9} -> {s['winner_label']} ({s['score'][0]}-{s['score'][1]}, {s['turns']} turns)")

        def work():
            try:
                self.result = run_tournament(cfg, prog, cancel=lambda: self.cancelled)
            except Exception as exc:  # shown to the user instead of crashing the UI
                self.error = repr(exc)

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def cancel(self) -> None:
        self.cancelled = True

    def back(self) -> None:
        self.cancelled = True
        self.app.goto("menu")

    def _update(self, task):
        i, n = self.progress
        running = self.thread is not None and self.thread.is_alive()
        if running:
            self.status.setText(f"Running... {i}/{n} matches finished" + ("  (cancelling)" if self.cancelled else ""))
        frac = i / n if n else 0
        self.bar["frameSize"] = (-0.05, -0.05 + 1.5 * frac, 0.52, 0.55)
        self.log_txt.setText("\n".join(self.lines[-9:]))
        if not running and self.thread is not None:
            self.thread = None
            if self.error:
                self.status.setText(f"Tournament failed: {self.error}")
            elif self.result:
                self._show_result(self.result)
        return task.cont

    def _show_result(self, res: dict) -> None:
        h = res["head_to_head"]
        self.status.setText(f"Done: {res['completed']} matches in {res['wall_time_s']} s. "
                            f"Minimax {h['minimax']} | MCTS {h['mcts']} | draws {h['draw']}")
        algos = res["aggregate"]["algorithms"]
        rows = []
        for k, a in algos.items():
            rows.append([k, a["matches"], f"{a['wins']}/{a['losses']}/{a['draws']}",
                         T.fmt(a["win_rate"] * 100 if a["win_rate"] is not None else None, 1, "%"),
                         T.fmt(a["avg_decision_time"], 3, " s"), T.fmt(a["total_nodes"]), T.fmt(a["total_prunes"]),
                         T.fmt(a["total_iterations"])])
        Table(self.out, ["Agent", "Seats", "W/L/D", "Win rate", "Avg time", "MM nodes", "AB prunes", "MCTS iters"],
              rows, (-0.05, -0.08), [0.22, 0.14, 0.2, 0.18, 0.18, 0.2, 0.2, 0.2], title="Measured results",
              colors=[T.P1 if k == "minimax" else T.P2 for k in algos], scale=0.027)
        f = res.get("files", {})
        T.text(self.out, f"Saved {f.get('json', '')}", (-0.05, -0.5), 0.022, T.MUTED, wordwrap=70)
        T.text(self.out, f"Report {f.get('report', '')}", (-0.05, -0.55), 0.022, T.MUTED, wordwrap=70)
        T.button(self.out, "OPEN ALGORITHM LAB", lambda: self.app.goto("lab"), pos=(0.7, 0, -0.7), size=(0.35, 0.045))

    def destroy(self) -> None:
        self.cancelled = True
        self.app.taskMgr.remove(self.task)
        super().destroy()
