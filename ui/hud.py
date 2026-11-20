"""In-match heads-up display: scoreboard, turn indicator, event feed, power meter
and the optional AI-analysis overlay. All values come from the match/decisions."""

from __future__ import annotations

import math

from direct.gui.DirectGui import DGG, DirectFrame
from panda3d.core import TransparencyAttrib

from game.state import ABILITY_KEYS, GameState
from ui import theme as T
from ui.geometry import radial_gradient

ABILITY_SHORT = {"precision": "Precision", "power": "Power", "spin": "Spin"}


class HUD:
    # Screen-space edges of the HUD (aspect2d units). The game camera frames the arena
    # inside the free area between them, so the HUD never covers the playground.
    TOP_PANEL_H = 0.20            # player / turn panels
    MARGIN = 0.03
    POWER_TOP = 0.34              # top of the power panel above the bottom edge
    ANALYSIS_W = 0.88
    ANALYSIS_BOTTOM = -0.44       # leaves room for the 6 control buttons below it

    def obstacles(self, power: bool, analysis: bool, buttons: int) -> list[tuple[float, float, float, float]]:
        """Screen rectangles (x0, x1, y0, y1) in normalised coords occupied by HUD panels.

        The game camera frames the arena so that its rim never enters any of them.
        ``buttons`` is the number of control buttons stacked in the bottom-right corner.
        """
        a, m = self.aspect, self.MARGIN

        def r(x0, x1, y0, y1):
            return (x0 / a, x1 / a, y0, y1)

        top = 1.0 - m
        rects = [
            r(-a + m, -a + m + self.player_w, top - self.TOP_PANEL_H, top),              # player 1
            r(a - m - self.player_w, a - m, top - self.TOP_PANEL_H, top),                # player 2
            r(-self.turn_hw, self.turn_hw, top - self.TOP_PANEL_H * 0.75, top),          # turn panel
            r(-a + m, -a + m + self.feed_w, -1.0 + m, -1.0 + m + 0.34),                  # event feed
        ]
        if analysis:
            rects.append(r(a - m - self.ANALYSIS_W, a - m, self.ANALYSIS_BOTTOM, 0.56))
        if power:
            rects.append(r(-0.6, 0.6, -1.0 + 0.13, -1.0 + self.POWER_TOP))
        if buttons:
            rects.append(r(a - 0.59, a - 0.05, -1.0 + 0.034, -1.0 + 0.07 + (buttons - 1) * 0.088 + 0.036))
        return rects

    def __init__(self, app, match, show_analysis: bool = True, on_tree=None) -> None:
        self.app = app
        self.match = match
        self.on_tree = on_tree          # SHOW TREE button in the analysis panel (None = no button)
        self.why_items: list = []
        self.root = DirectFrame(parent=app.aspect2d, frameColor=(0, 0, 0, 0))
        self.dot_tex = radial_gradient(32, (1, 1, 1, 1), (1, 1, 1, 0), 6.0)
        # panel widths follow the window shape so nothing overlaps on narrow (4:3, 5:4) screens
        self.aspect = app.getAspectRatio()
        self.turn_hw = 0.47
        self.player_w = max(0.6, min(0.82, self.aspect - self.MARGIN - self.turn_hw - 0.04))
        self.feed_w = max(0.5, min(0.95, self.aspect - 0.6 - self.MARGIN - 0.04))
        self.banner_t = 0.0
        self.banner_col = T.TEXT
        self._build_player_panels()
        self._build_turn_panel()
        self._build_feed()
        self._build_analysis()
        self._build_power()
        self.banner = T.text(self.app.aspect2d, "", (0, 0.25), 0.1, T.TEXT, "center", "title")
        self.banner.hide()
        self.set_analysis_visible(show_analysis)

    # ------------------------------------------------------------------ construction
    def _build_player_panels(self) -> None:
        self.pp = []
        w, h = self.player_w, self.TOP_PANEL_H
        for p in (0, 1):
            anchor = self.app.a2dTopLeft if p == 0 else self.app.a2dTopRight
            x0 = self.MARGIN if p == 0 else -(w + self.MARGIN)
            fr = T.panel(anchor, (0, w, -h, 0), (x0, 0, -self.MARGIN))
            col = T.PLAYER_COLORS[p]
            DirectFrame(parent=fr, frameColor=col, frameSize=(0, w, -0.01, 0))
            name = T.text(fr, f"PLAYER {p + 1}", (0.03, -0.062), 0.045, col, kind="title")
            count = T.text(fr, "", (w - 0.03, -0.066), 0.05, T.TEXT, "right", "title")
            ctrl = T.text(fr, "", (0.03, -0.107), 0.03, T.MUTED)
            dots = []
            # every marble the player started with (eliminated ones stay as grey dots after a rebuild)
            n = sum(1 for m in self.match.state.marbles if m.owner == p)
            for i in range(max(n, 1)):
                d = DirectFrame(parent=fr, frameTexture=self.dot_tex, frameColor=col,
                                frameSize=(-0.022, 0.022, -0.022, 0.022), pos=(0.05 + i * 0.055, 0, -0.155))
                d.setTransparency(TransparencyAttrib.MAlpha)
                dots.append(d)
            abil = T.text(fr, "", (w - 0.03, -0.164), 0.026, T.MUTED, "right")
            self.pp.append({"frame": fr, "ctrl": ctrl, "dots": dots, "count": count, "abil": abil, "name": name})

    def _build_turn_panel(self) -> None:
        h = self.TOP_PANEL_H * 0.75
        hw = self.turn_hw
        fr = T.panel(self.app.a2dTopCenter, (-hw, hw, -h, 0), (0, 0, -self.MARGIN))
        self.turn_txt = T.text(fr, "", (0, -0.058), 0.048, T.TEXT, "center", "title")
        self.status_txt = T.text(fr, "", (0, -0.112), 0.031, T.TEXT, "center")
        self.turn_frame = fr
        self.turn_bar = DirectFrame(parent=fr, frameColor=T.P1, frameSize=(-hw, hw, -h, -h + 0.01))

    def _build_feed(self) -> None:
        fr = T.panel(self.app.a2dBottomLeft, (0, self.feed_w, 0, 0.34), (self.MARGIN, 0, self.MARGIN))
        self.feed_chars = int((self.feed_w - 0.06) / 0.0155)
        T.text(fr, "EVENT FEED", (0.03, 0.3), 0.029, T.EDGE, kind="title")
        self.feed_lines = [T.text(fr, "", (0.03, 0.255 - i * 0.04), 0.026, T.TEXT) for i in range(6)]
        self.feed_frame = fr

    def _build_analysis(self) -> None:
        w = self.ANALYSIS_W
        fr = T.panel(self.app.a2dRightCenter, (-w, 0, self.ANALYSIS_BOTTOM, 0.56), (-self.MARGIN, 0, 0.0))
        T.text(fr, "AI ANALYSIS", (-w + 0.03, 0.505), 0.04, T.EDGE, kind="title")
        if self.on_tree is not None:
            self.tree_btn = T.button(fr, "SHOW TREE  (T)", self.on_tree, pos=(-0.2, 0, 0.517), size=(0.17, 0.032),
                                     text_scale=0.027, accent=T.GOLD)
        # key for the marble names used below and on the board tags
        for p, x in ((0, -w + 0.03), (1, -w + 0.33)):
            ids = [m.mid for m in self.match.state.marbles if m.owner == p]
            rng = f"M{ids[0]}-M{ids[-1]}" if ids and ids == list(range(ids[0], ids[-1] + 1)) and len(ids) > 1                 else ", ".join(f"M{i}" for i in ids)
            T.text(fr, f"{rng} = Player {p + 1}", (x, 0.466), 0.026, T.PLAYER_COLORS[p], kind="title")
        self.an_blocks = []
        for p in (0, 1):
            y = 0.425 - p * 0.118
            head = T.text(fr, "", (-w + 0.03, y), 0.03, T.PLAYER_COLORS[p], kind="title")
            body = T.text(fr, "", (-w + 0.03, y - 0.038), 0.026, T.TEXT)
            self.an_blocks.append((head, body))
        DirectFrame(parent=fr, frameColor=(T.EDGE[0], T.EDGE[1], T.EDGE[2], 0.45),
                    frameSize=(-w + 0.02, -0.02, 0.212, 0.217))
        T.text(fr, "WHY THIS SHOT", (-w + 0.03, 0.17), 0.036, T.GOLD, kind="title")
        self.why_head = T.text(fr, "the reasons appear after the first AI decision", (-w + 0.03, 0.122), 0.031,
                               T.MUTED, wordwrap=27)
        self.an_frame = fr

    def _build_power(self) -> None:
        h = self.POWER_TOP - 0.13
        fr = T.panel(self.app.a2dBottomCenter, (-0.6, 0.6, 0, h), (0, 0, 0.13))
        T.text(fr, "POWER", (-0.57, h - 0.055), 0.03, T.EDGE, kind="title")
        self.power_bg = DirectFrame(parent=fr, frameColor=(0.1, 0.12, 0.18, 1), frameSize=(-0.57, 0.57, 0.1, 0.135))
        self.power_fill = DirectFrame(parent=fr, frameColor=T.P1, frameSize=(-0.57, -0.57, 0.1, 0.135))
        self.power_txt = T.text(fr, "", (0.57, h - 0.055), 0.03, T.TEXT, "right")
        self.ability_txt = T.text(fr, "", (-0.57, 0.055), 0.028, T.TEXT)
        self.hint_txt = T.text(fr, "", (-0.57, 0.017), 0.023, T.MUTED)
        self.power_frame = fr
        fr.hide()

    # ------------------------------------------------------------------ updates
    def refresh(self, state: GameState, status: str, status_color=T.TEXT) -> None:
        m = self.match
        for p in (0, 1):
            d = self.pp[p]
            spec = m.specs[p]
            ctrl = "Human player" if spec.is_human else f"{spec.label} | {spec.difficulty}"
            d["ctrl"].setText(ctrl)
            alive = state.count(p)
            d["count"].setText(str(alive))
            for i, dot in enumerate(d["dots"]):
                dot["frameColor"] = T.PLAYER_COLORS[p] if i < alive else (0.25, 0.27, 0.33, 0.6)
            d["abil"].setText("   ".join(f"{ABILITY_SHORT[k]} {state.ability_count(p, k)}" for k in ABILITY_KEYS))
            active = (state.to_move == p and not state.is_terminal)
            d["name"]["fg"] = T.PLAYER_COLORS[p] if active else T.MUTED
        self.turn_txt.setText(f"TURN {min(state.turn + 1, state.turn_limit)} / {state.turn_limit}"
                              if not state.is_terminal else "MATCH OVER")
        self.status_txt.setText(status)
        self.status_txt["fg"] = status_color
        self.turn_bar["frameColor"] = T.PLAYER_COLORS[state.to_move] if not state.is_terminal else T.GOLD
        lines = m.events[-6:]
        for i, t in enumerate(self.feed_lines):
            txt = lines[i] if i < len(lines) else ""
            t.setText(txt[:self.feed_chars])
            t["fg"] = T.WARN if "KNOCKOUT" in txt or "OVER" in txt else (T.DANGER if "own marble" in txt else T.TEXT)

    def set_analysis(self, player: int, decision) -> None:
        head, body = self.an_blocks[player]
        if decision is None:
            head.setText(f"P{player + 1} | {self.match.specs[player].label}")
            body.setText("waiting for first decision..." if not self.match.specs[player].is_human
                         else "human player (no search)")
            return
        s = decision.stats
        a = decision.action
        deg = math.degrees(a.angle) % 360
        ab = {"none": "Normal", "precision": "Precision", "power": "Power", "spin_left": "Spin L",
              "spin_right": "Spin R"}.get(a.ability, a.ability)
        lines = [f"M{a.marble_id}  dir {deg:3.0f}°  power {a.power:.2f}  {ab}" + ("  FALLBACK" if decision.fallback else "")]
        tm = T.fmt(s.get("decision_time"), 2, " s")
        if "nodes_visited" in s:
            lines.append(f"depth {T.fmt(s.get('depth_reached'))}  |  nodes {T.fmt(s.get('nodes_visited'))}  |  "
                         f"eval {T.fmt(s.get('evaluation'), 1)}  |  {tm}")
        elif "iterations" in s:
            lines.append(f"iterations {T.fmt(s.get('iterations'))}  |  visits {T.fmt(s.get('selected_visits'))}  |  "
                         f"reward {T.fmt(s.get('selected_reward'), 2)}  |  {tm}")
        head.setText(f"P{player + 1} | {decision.algorithm}")
        body.setText("\n".join(lines[:2]))

    def set_explanation(self, ex) -> None:
        """WHY THIS SHOT: plain-language reasons and the candidate ranking (ai.explain.Explanation)."""
        for it in self.why_items:
            it.destroy()
        self.why_items = []
        if ex is None:
            self.why_head.setText("the reasons appear after the first AI decision")
            self.why_head["fg"] = T.MUTED
            return
        w = self.ANALYSIS_W
        fr = self.an_frame
        self.why_head.setText(ex.headline)
        self.why_head["fg"] = T.PLAYER_COLORS[ex.player]
        y = 0.122 - 0.04 * self._rows(self.why_head) - 0.008
        mark = {"good": ("+", T.GOOD), "bad": ("-", T.DANGER), "neutral": ("~", T.WARN)}
        for r in ex.reasons:
            if y < -0.2:                      # keep room for the candidates
                break
            sym, col = mark.get(r.kind, ("~", T.TEXT))
            o = T.text(fr, f"{sym} {r.text}", (-w + 0.03, y), 0.03, col, wordwrap=27)
            self.why_items.append(o)
            y -= 0.039 * self._rows(o)
        if ex.search:
            o = T.text(fr, ex.search, (-w + 0.03, y - 0.004), 0.025, T.MUTED, wordwrap=33)
            self.why_items.append(o)
            y -= 0.033 * self._rows(o) + 0.012
        if ex.ranking:
            o = T.text(fr, "CANDIDATES  (# = tag on the board)", (-w + 0.03, y - 0.006), 0.029, T.EDGE, kind="title")
            self.why_items.append(o)
            y -= 0.048
            for row in ex.ranking:
                if y < self.ANALYSIS_BOTTOM + 0.03:
                    break
                col = T.GOLD if row.selected else T.TEXT
                score = row.score + (f"  Q {row.mean:.2f}" if row.mean is not None else "")
                left = T.text(fr, f"#{row.rank}  {row.label[:27]}", (-w + 0.03, y), 0.028, col)
                right = T.text(fr, score, (-0.03, y), 0.028, col, "right")
                self.why_items += [left, right]
                y -= 0.038

    @staticmethod
    def _rows(o) -> int:
        try:
            return max(1, int(o.textNode.getNumRows()))
        except Exception:
            return 1

    def set_analysis_visible(self, v: bool) -> None:
        (self.an_frame.show if v else self.an_frame.hide)()

    def set_power(self, visible: bool, power: float = 0.0, ability: str = "none", player: int = 0,
                  abilities_left: dict | None = None, hint: str = "") -> None:
        if not visible:
            self.power_frame.hide()
            return
        self.power_frame.show()
        w = 1.14 * max(0.0, min(1.0, power))
        self.power_fill["frameSize"] = (-0.57, -0.57 + w, 0.1, 0.135)
        hot = (T.PLAYER_COLORS[player][0] * (1 - power) + 1.0 * power,
               T.PLAYER_COLORS[player][1] * (1 - power) + 0.35 * power,
               T.PLAYER_COLORS[player][2] * (1 - power) + 0.2 * power, 1)
        self.power_fill["frameColor"] = hot
        if ability == "power":
            from config import DEFAULT_CONFIG
            boost = DEFAULT_CONFIG.physics.power_shot_multiplier
            self.power_txt.setText(f"{power * 100:.0f}% x{boost:g} POWER SHOT = {power * boost * 100:.0f}%")
        else:
            self.power_txt.setText(f"{power * 100:.0f}%")
        left = abilities_left or {}
        opts = [("1", "none", "Normal")] + [(k, ab, ABILITY_SHORT[ab.split('_')[0]] + (" L" if ab == "spin_left" else " R" if ab == "spin_right" else ""))
                                             for k, ab in (("2", "precision"), ("3", "power"), ("4", "spin_left"), ("5", "spin_right"))]
        parts = []
        for key, ab, name in opts:
            n = "" if ab == "none" else f"({left.get(ab, 0)})"
            mark = ">" if ab == ability else " "
            parts.append(f"{mark}[{key}] {name}{n}")
        self.ability_txt.setText("  ".join(parts))
        self.hint_txt.setText(hint)

    def show_banner(self, txt: str, color=T.TEXT, duration: float = 1.4) -> None:
        self.banner.setText(txt)
        self.banner_col = color
        self.banner["fg"] = color
        self.banner.show()
        self.banner_t = duration

    def update(self, dt: float) -> None:
        if self.banner_t > 0:
            self.banner_t -= dt
            a = min(1.0, self.banner_t / 0.4)
            c = self.banner_col
            self.banner["fg"] = (c[0], c[1], c[2], a)
            if self.banner_t <= 0:
                self.banner.hide()

    def destroy(self) -> None:
        self.banner.destroy()
        self.root.destroy()
        for d in self.pp:
            d["frame"].destroy()
        for fr in (self.turn_frame, self.feed_frame, self.an_frame, self.power_frame):
            fr.destroy()
