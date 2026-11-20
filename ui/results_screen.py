"""Match results overlay. Every number comes from Match.summary()."""

from __future__ import annotations

from ui import theme as T


class ResultsScreen:
    def __init__(self, app, summary: dict, on_rematch, on_menu, on_lab, on_replay=None,
                 replay_mode: bool = False, series: dict | None = None) -> None:
        """``series`` (from Series.to_dict) switches the header to the combined two-round result;
        the per-player statistics below then describe the last round."""
        self.app = app
        s = summary
        fr = T.panel(app.aspect2d, (-1.05, 1.05, -0.78, 0.72), (0, 0, 0), color=(0.04, 0.055, 0.1, 0.97))
        self.frame = fr
        w = series["winner"] if series else s["winner"]
        what = " THE SERIES" if series else ""
        if w == -1:
            title, col = ("SERIES DRAW" if series else "DRAW"), T.GOLD
        else:
            title, col = f"PLAYER {w + 1} WINS{what}", T.PLAYER_COLORS[w]
        T.text(fr, title, (0, 0.58), 0.1 if not series else 0.085, col, "center", "title")
        if series:
            T.text(fr, series.get("winner_label") or "", (0, 0.5), 0.038, T.TEXT, "center")
            T.text(fr, series["reason"], (0, 0.445), 0.032, T.MUTED, "center", wordwrap=50)
            rw, tm = series["round_wins"], series["total_marbles"]
            T.text(fr, f"{rw[0]}  :  {rw[1]}", (0, 0.31), 0.11, T.TEXT, "center", "title")
            T.text(fr, f"rounds won   |   marbles remaining over both rounds {tm[0]} : {tm[1]}", (0, 0.255), 0.028,
                   T.MUTED, "center")
            parts = []
            for r in series["rounds"]:
                res = "draw" if r["winner"] == -1 else f"P{r['winner'] + 1} won"
                parts.append(f"Round {r['round']} (P{r['first_player'] + 1} shot first): {res} "
                             f"{r['score'][0]}:{r['score'][1]}")
            T.text(fr, "     ".join(parts), (0, 0.18), 0.03, T.TEXT, "center")
        else:
            T.text(fr, s["winner_label"] or "", (0, 0.5), 0.038, T.TEXT, "center")
            T.text(fr, s["end_reason"], (0, 0.445), 0.032, T.MUTED, "center", wordwrap=50)
            T.text(fr, f"{s['score'][0]}  :  {s['score'][1]}", (0, 0.31), 0.11, T.TEXT, "center", "title")
            T.text(fr, "final score (marbles remaining)", (0, 0.255), 0.028, T.MUTED, "center")
            info = (f"Arena: {s['arena_name']}    Turns: {s['turns']}    Duration: {s['duration_s']:.1f} s    "
                    f"Marbles eliminated: {s['eliminated_total']}")
            T.text(fr, info, (0, 0.18), 0.032, T.TEXT, "center")
        for p, pl in enumerate(s["players"]):
            x = -0.98 + p * 1.0
            pc = T.panel(fr, (0, 0.96, -0.5, 0), (x, 0, 0.12), color=T.PANEL_LIGHT)
            head = f"PLAYER {p + 1} | {pl['controller']}" + ("  (last round)" if series else "")
            T.text(pc, head, (0.03, -0.06), 0.036, T.PLAYER_COLORS[p], kind="title")
            rows = [("Difficulty", pl["difficulty"] if pl["kind"] != "human" else "—"),
                    ("Remaining marbles", pl["remaining"]),
                    ("Knocked out by opponent", pl["eliminated_by_opponent"]),
                    ("Self-eliminations", pl["self_eliminations"]),
                    ("Decisions made", pl["decisions"] if pl["kind"] != "human" else "—"),
                    ("Avg decision time", T.fmt(pl["avg_decision_time"], 3, " s")),
                    ("Max decision time", T.fmt(pl["max_decision_time"], 3, " s"))]
            if pl["kind"] == "minimax":
                rows += [("Minimax nodes", T.fmt(pl.get("minimax_nodes"))),
                         ("Alpha-Beta prunes", T.fmt(pl.get("alphabeta_prunes")))]
            elif pl["kind"] == "mcts":
                rows += [("MCTS iterations", T.fmt(pl.get("mcts_iterations"))),
                         ("Physics simulations", T.fmt(pl.get("physics_simulations")))]
            if pl["kind"] != "human":
                rows.append(("Fallback shots", pl["fallbacks"]))
            for i, (k, v) in enumerate(rows):
                y = -0.115 - i * 0.037
                T.text(pc, k, (0.03, y), 0.029, T.MUTED)
                T.text(pc, str(v), (0.93, y), 0.029, T.TEXT, "right")
        if replay_mode:
            T.text(fr, "REPLAY — re-simulated from the recorded actions with the real rules", (0, 0.39), 0.03,
                   T.WARN, "center")
            buttons = [("WATCH AGAIN", on_rematch), ("ALGORITHM LAB", on_lab), ("MAIN MENU", on_menu)]
        else:
            buttons = [("REMATCH", on_rematch)]
            if on_replay:
                buttons.append(("REPLAY LAST ROUND" if series else "WATCH REPLAY", on_replay))
            buttons += [("ALGORITHM LAB", on_lab), ("MAIN MENU", on_menu)]
        step = 1.9 / len(buttons)
        for i, (lab, cmd) in enumerate(buttons):
            T.button(fr, lab, cmd, pos=(-0.95 + step * (i + 0.5), 0, -0.68), size=(step / 2 - 0.03, 0.05),
                     text_scale=0.04)

    def destroy(self) -> None:
        self.frame.destroy()
