"""The match screen: connects Match (rules/physics/AI), ArenaView (3-D), HUD and audio.

Modes::

    ai_vs_ai   both seats are AI agents; START / PAUSE / STEP / SPEED controls
    human      Human vs AI
    pvp        Human vs Human on one computer (alternating turns)
    replay     re-simulates a recorded match through the real rules and physics

Rounds: with rounds=2 (ai_vs_ai / human / pvp) the same pairing plays a second
round in which the other player shoots first (see game/series.py). Every round
is logged and saved as its own replay; the results screen shows the combined
series result.

Turn phases::

    waiting_start -> turn_start -> (human | thinking -> reveal) -> animating -> post -> turn_start ... -> over

AI searches run in the AIWorker process; this screen only polls for results on
the main thread, so the window keeps rendering and the controls stay live.
Input is accepted only in the ``human`` phase, i.e. never while a shot resolves.
"""

from __future__ import annotations

import math
import time

from direct.gui.DirectGui import DirectFrame

from ai.base_agent import Decision
from analytics.match_logger import log_match
from game.actions import Action, ABILITY_TO_KEY, IllegalActionError, is_legal
from game.match import Match, PlayerSpec
from game.replay import config_fingerprint, record_from_match, save_replay
from game.series import Series
from game.simulation import simulate_turn
from game.state import DRAW, PLAYER_NAMES
from ui import theme as T
from ui.arena_view import ArenaView
from ui.hud import HUD
from ui.playback import Playback

SPEEDS = [0.5, 1.0, 2.0, 4.0]
ABILITY_SFX = {"precision": "ability_precision", "power": "ability_power",
               "spin_left": "ability_spin", "spin_right": "ability_spin"}


def preview_paths(state, action, arena, cfg, color, style, follow_target=True, max_t=None):
    """Shooter path up to its first contact (plus the struck marble's initial path),
    taken from a real simulate_turn run."""
    res = simulate_turn(state, action, arena, cfg, record=True, validate=False)
    sid = action.marble_id
    hits = [c.t for c in res.contacts if c.a == sid or c.b == sid]
    t_hit = min(hits) if hits else None
    limit = t_hit if t_hit is not None else (max_t if max_t is not None else 1e9)
    pts = []
    for k, f in enumerate(res.frames):
        if f[0] > limit + 1e-9:
            break
        if k % 2 == 0 or f[0] >= limit - 1e-9:
            pts.append((f[1][sid], f[2][sid]))
        if not f[3][sid]:
            break
    out = [(pts, color, style)]
    tgt = res.first_hit
    if follow_target and t_hit is not None and tgt >= 0:
        tp = [(f[1][tgt], f[2][tgt]) for f in res.frames if t_hit <= f[0] <= t_hit + 0.7 and f[3][tgt]][::2]
        if len(tp) >= 2:
            out.append((tp, (1.0, 1.0, 1.0, color[3] * 0.8), "dashed"))
    return out, res


class GameScreen:
    def __init__(self, app, mode: str, specs: tuple[PlayerSpec, PlayerSpec] | None = None, arena_id: str | None = None,
                 marbles: int = 3, seed: int = 0, autostart: bool = False, replay: dict | None = None,
                 rounds: int = 1) -> None:
        self.app = app
        self.mode = mode
        st = app.settings
        self.replay = replay
        self.replay_index = 0
        self.replay_error: str | None = None
        first_player, turn_limit = 0, None
        if mode == "replay":
            specs = tuple(PlayerSpec(**s) for s in replay["specs"])  # type: ignore[assignment]
            arena_id, marbles, seed = replay["arena"], int(replay["marbles"]), int(replay["seed"])
            first_player, turn_limit = int(replay["first_player"]), int(replay["turn_limit"])
        self.match = Match(arena_id, specs[0], specs[1], marbles=marbles, seed=seed, first_player=first_player,
                           turn_limit=turn_limit)
        self.base_seed = seed
        # two-round series (first shooter swapped); a replay always shows one recorded round
        self.series: Series | None = Series(int(rounds), first_player) if mode != "replay" and int(rounds) > 1 else None
        self.view = ArenaView(app, self.match.arena, self.match.cfg, quality=st.get("graphics_quality", "high"),
                              effects=st.get("effects", "full"))
        self.show_analysis = bool(st.get("show_analysis", True))
        self.view.show_tags = self.show_analysis          # M0, M1 ... tags above the marbles
        self.view.set_state(self.match.state)
        self.show_traj = bool(st.get("show_trajectories", True))
        auto = mode in ("ai_vs_ai", "replay")
        self.speed = float(st.get("ai_playback_speed" if auto else "anim_speed", 1.0))
        self.has_ai = any(not sp.is_human for sp in self.match.specs)
        # the AI-analysis panel only makes sense when at least one seat is an AI
        self.tree_popup = None
        self.last_ai = None              # (state, decision) of the most recent AI decision, for SHOW TREE
        self.last_expl = None            # its plain-language explanation (WHY THIS SHOT)
        self.hud = HUD(app, self.match, self.show_analysis and self.has_ai,
                       on_tree=self.toggle_tree if self.has_ai else None)
        self.last_decisions = [None, None]
        self.last_ai = None
        self.last_expl = None
        for p in (0, 1):
            self.hud.set_analysis(p, None)
        self.hud.set_explanation(None)
        app.cam_rig.auto_orbit = 0.0
        app.cam_rig.aspect_listeners.append(self._on_aspect)
        humans = [p for p in (0, 1) if self.match.specs[p].is_human]
        app.cam_rig.set_view(heading=180.0 if humans == [1] else 0.0, snap=True)
        self.fit_camera()
        self.phase = "waiting_start" if mode == "ai_vs_ai" and not autostart else "turn_start"
        self.paused = False
        self.step_mode = False
        self.req = None
        self.req_t0 = 0.0
        self.req_deadline = 0.0
        self.think_t0 = 0.0
        self.pending = None
        self.reveal_t = 0.0
        self.post_t = 0.0
        self.playback: Playback | None = None
        self.results = None
        self.pause_menu = None
        self.logged = False
        self.replay_record: dict | None = None
        # human aiming
        self.sel: int | None = None
        self.aim_angle = math.pi / 2
        self.aim_power = 0.6
        self.ability = "none"
        self.dragging = False
        self.drag_moved = False
        self.preview_dirty = True
        self.preview_t = 0.0
        self.keys = set()
        self._build_controls()
        self._bind()
        self.task = app.taskMgr.add(self._update, "game-update")
        if mode == "replay" and replay.get("config_fingerprint") != config_fingerprint(self.match.cfg, arena_id):
            self.hud.show_banner("Note: game configuration changed since recording", T.WARN, 3.0)
        self._refresh_hud()

    # ------------------------------------------------------------------ helpers
    def fit_camera(self) -> None:
        """Frame the whole arena as large as possible without any HUD panel covering it.
        Re-applied automatically when the window is resized."""
        human_input = self.mode in ("human", "pvp")
        buttons = 5 if human_input else 6
        self.app.cam_rig.fit_arena(
            self.match.arena.radius + 0.3,
            lambda: self.hud.obstacles(power=human_input, analysis=self.has_ai, buttons=buttons))

    def _on_aspect(self) -> None:
        """Window resized: rebuild the HUD for the new shape (the camera re-frames itself)."""
        old = self.hud
        self.hud = HUD(self.app, self.match, self.show_analysis and self.has_ai,
                       on_tree=self.toggle_tree if self.has_ai else None)
        old.destroy()
        for p in (0, 1):
            self.hud.set_analysis(p, self.last_decisions[p])
        self.hud.set_explanation(self.last_expl)
        self._refresh_hud()

    @property
    def humans(self) -> list[int]:
        return [p for p in (0, 1) if self.match.specs[p].is_human]

    def _status(self) -> tuple[str, tuple]:
        s = self.match.state
        if self.replay_error:
            return f"REPLAY STOPPED: {self.replay_error}", T.DANGER
        if self.paused:
            return "PAUSED — press Resume (or Esc)", T.WARN
        if s.is_terminal:
            if self.series is not None and not self.series.is_over and self.phase == "over":
                return f"Round {len(self.series.rounds)} over — the next round starts shortly", T.GOLD
            return s.end_reason, T.GOLD
        p = s.to_move
        who = f"{PLAYER_NAMES[p].upper()} | {self.match.specs[p].label}"
        prefix = f"REPLAY {self.replay_index}/{len(self.replay['turns'])} | " if self.mode == "replay" else ""
        if self.series is not None:
            prefix = f"ROUND {self.series.round_number}/{self.series.total_rounds} | "
        if self.phase == "waiting_start":
            what = f"{self.series.total_rounds}-round series" if self.series is not None else "AI-vs-AI match"
            return f"Press START to begin the {what}", T.WARN
        if self.phase == "thinking":
            dots = "." * (1 + int(time.time() * 3) % 3)
            return f"{prefix}{who} is thinking{dots}", T.PLAYER_COLORS[p]
        if self.phase == "human":
            lead = f"{PLAYER_NAMES[p].upper()}'S TURN" if self.mode == "pvp" else "YOUR TURN"
            return f"{prefix}{lead} — aim and shoot", T.PLAYER_COLORS[p]
        if self.phase in ("animating", "post") and self.match.history:
            q = self.match.history[-1].player
            return f"{prefix}{PLAYER_NAMES[q].upper()} | {self.match.specs[q].label} shot", T.PLAYER_COLORS[q]
        if self.phase == "reveal":
            return f"{prefix}{who} shoots", T.PLAYER_COLORS[p]
        return prefix + who, T.PLAYER_COLORS[p]

    def _refresh_hud(self) -> None:
        txt, col = self._status()
        self.hud.refresh(self.match.state, txt, col)

    # ------------------------------------------------------------------ controls
    def _build_controls(self) -> None:
        self.ctrl_frame = DirectFrame(parent=self.app.a2dBottomRight, frameColor=(0, 0, 0, 0))
        if self.mode in ("ai_vs_ai", "replay"):
            specs = [("start", "START", self.on_start_pause), ("step", "STEP ONE TURN", self.on_step),
                     ("speed", "", self.on_speed), ("analysis", "", self.on_toggle_analysis),
                     ("restart", "RESTART", self.on_restart), ("menu", "MAIN MENU", self.on_menu)]
        else:
            specs = [("start", "PAUSE", self.on_start_pause), ("speed", "", self.on_speed),
                     ("analysis", "", self.on_toggle_analysis), ("restart", "RESTART", self.on_restart),
                     ("menu", "MAIN MENU", self.on_menu)]
        self.btns = {}
        for i, (key, lab, cmd) in enumerate(reversed(specs)):
            self.btns[key] = T.button(self.ctrl_frame, lab, cmd, pos=(-0.32, 0, 0.07 + i * 0.088), size=(0.27, 0.036),
                                      text_scale=0.034)
        self._label_controls()

    def _label_controls(self) -> None:
        waiting = self.mode == "ai_vs_ai" and self.phase == "waiting_start"
        self.btns["start"]["text"] = "START MATCH" if waiting else ("RESUME" if self.paused else "PAUSE")
        T.set_button_active(self.btns["start"], waiting)
        self.btns["speed"]["text"] = f"SPEED  {self.speed:g}x"
        self.btns["analysis"]["text"] = f"EXPLANATIONS: {'ON' if self.show_analysis else 'OFF'}"

    def _bind(self) -> None:
        ac = self.app.accept
        ac("mouse1", self.on_mouse_down)
        ac("mouse1-up", self.on_mouse_up)
        ac("mouse3", self.app.cam_rig.begin_drag)
        ac("mouse3-up", self.app.cam_rig.end_drag)
        ac("wheel_up", self.app.cam_rig.zoom, [0.92])
        ac("wheel_down", self.app.cam_rig.zoom, [1.08])
        ac("escape", self.toggle_pause_menu)
        ac("space", self.on_space)
        ac("tab", self.cycle_marble)
        ac("h", self.on_toggle_analysis)
        ac("p", self.on_start_pause)
        ac("r", self.on_restart)
        ac("m", self.on_toggle_mute)
        ac("c", self.app.cam_rig.reset_view)
        ac("t", self.toggle_tree)
        for k in ("a", "d", "w", "s", "arrow_left", "arrow_right", "arrow_up", "arrow_down", "q", "e"):
            ac(k, self.keys.add, [k])
            ac(k + "-up", self.keys.discard, [k])
        for i, ab in enumerate(("none", "precision", "power", "spin_left", "spin_right")):
            ac(str(i + 1), self.set_ability, [ab])

    def _unbind(self) -> None:
        for ev in ("mouse1", "mouse1-up", "mouse3", "mouse3-up", "wheel_up", "wheel_down", "escape", "space", "tab",
                   "h", "p", "r", "m", "c", "t", "1", "2", "3", "4", "5"):
            self.app.ignore(ev)
        for k in ("a", "d", "w", "s", "arrow_left", "arrow_right", "arrow_up", "arrow_down", "q", "e"):
            self.app.ignore(k)
            self.app.ignore(k + "-up")

    def on_toggle_mute(self) -> None:
        self.app.settings["muted"] = not self.app.settings.get("muted", False)
        self.app.apply_audio_settings()
        self.app.save_settings()
        self.hud.show_banner("SOUND MUTED" if self.app.settings["muted"] else "SOUND ON", T.MUTED, 0.8)

    def on_start_pause(self) -> None:
        if self.tree_popup is not None:
            self.toggle_tree()
            return
        if self.pause_menu is not None:
            # the pause menu owns the paused state: P closes it (same as RESUME / Esc)
            self.toggle_pause_menu()
            return
        self.app.audio.play("ui")
        if self.phase == "waiting_start":
            self.phase = "turn_start"
            self.paused = False
            self.hud.show_banner("FIGHT!", T.GOLD)
        elif not self.match.is_over and not self.replay_error:
            self.paused = not self.paused
            if self.paused:
                self.app.audio.set_rolling(0.0)
        self._label_controls()

    def on_step(self) -> None:
        if self.match.is_over or self.replay_error or self.pause_menu is not None or self.tree_popup is not None:
            return
        if self.phase == "waiting_start":
            self.phase = "turn_start"
        self.paused = False
        self.step_mode = True
        self._label_controls()

    def on_speed(self) -> None:
        i = SPEEDS.index(self.speed) if self.speed in SPEEDS else 1
        self.speed = SPEEDS[(i + 1) % len(SPEEDS)]
        if self.playback:
            self.playback.speed = self.speed
        self._label_controls()

    def on_toggle_analysis(self) -> None:
        self.show_analysis = not self.show_analysis
        self.hud.set_analysis_visible(self.show_analysis and self.has_ai)
        self.view.set_tags_visible(self.show_analysis)
        if not self.show_analysis and self.phase != "human":
            self.view.clear_paths()
        self._label_controls()

    def on_restart(self) -> None:
        """Restart the match; in a series, restart the whole series from round 1."""
        if self.series is not None:
            self.series.reset()
            self.match.first_player = self.series.first_player
            self.match.seed = self.base_seed
        self._reset_round()
        self.phase = "waiting_start" if self.mode == "ai_vs_ai" else "turn_start"
        self._label_controls()

    def _start_next_round(self, task=None):
        """Series: the next round on the same arena, with the other player shooting first."""
        s = self.series
        if s is None or s.is_over:
            return None
        self.match.first_player = s.next_first_player()
        self.match.seed = self.base_seed + len(s.rounds)
        self._reset_round()
        self.phase = "turn_start"                     # no second START press needed
        p = self.match.first_player
        self.hud.show_banner(f"ROUND {s.round_number}: {PLAYER_NAMES[p].upper()} SHOOTS FIRST",
                             T.PLAYER_COLORS[p], 1.6)
        self._label_controls()
        return None

    def _reset_round(self) -> None:
        self.app.worker.cancel_all()
        self.app.taskMgr.remove("show-results")
        self.app.taskMgr.remove("next-round")
        self.app.audio.set_rolling(0.0)
        self.req = None
        self._close_overlays()
        self.match.restart()
        self.view.reset_marbles()
        self.view.set_state(self.match.state)
        self.view.clear_paths()
        self.view.clear_aim()
        self.view.set_selected(None)
        self.hud.set_power(False)
        self.playback = None
        self.logged = False
        self.paused = False
        self.step_mode = False
        self.replay_index = 0
        self.replay_error = None
        self.last_decisions = [None, None]
        self.last_ai = None
        self.last_expl = None
        for p in (0, 1):
            self.hud.set_analysis(p, None)
        self.hud.set_explanation(None)

    def on_menu(self) -> None:
        self.app.goto("menu")

    def watch_replay(self) -> None:
        rec = self.replay if self.mode == "replay" else self.replay_record
        if rec is not None:
            self.app.goto("game", mode="replay", replay=rec)

    def toggle_pause_menu(self) -> None:
        if self.tree_popup is not None:
            self.toggle_tree()
            return
        if self.results is not None:
            return
        if self.pause_menu is not None:
            self.pause_menu.destroy()
            self.pause_menu = None
            self.paused = False
        else:
            self.paused = True
            self.app.audio.set_rolling(0.0)
            fr = T.panel(self.app.aspect2d, (-0.45, 0.45, -0.4, 0.4), (0, 0, 0))
            T.text(fr, "PAUSED", (0, 0.27), 0.08, T.TEXT, "center", "title")
            T.button(fr, "RESUME", self.toggle_pause_menu, pos=(0, 0, 0.12))
            T.button(fr, "RESTART MATCH", lambda: (self.toggle_pause_menu(), self.on_restart()), pos=(0, 0, -0.0))
            T.button(fr, "MAIN MENU", self.on_menu, pos=(0, 0, -0.12))
            T.text(fr, "Esc / P: resume  |  M: mute", (0, -0.3), 0.032, T.MUTED, "center")
            self.pause_menu = fr
        self._label_controls()

    # ------------------------------------------------------------------ SHOW TREE
    def toggle_tree(self) -> None:
        """Open / close the decision tree of the most recent AI decision (pauses the match)."""
        if self.tree_popup is not None:
            self.tree_popup.destroy()
            self.tree_popup = None
            self.paused = self._paused_before_tree
            self._label_controls()
            return
        if self.results is not None or self.pause_menu is not None:
            return
        if self.last_ai is None:
            self.hud.show_banner("No AI decision yet", T.WARN, 1.0)
            return
        from ui.tree_view import TreePopup
        state, dec = self.last_ai
        self._paused_before_tree = self.paused
        self.paused = True
        self.app.audio.set_rolling(0.0)
        spec = self.match.specs[dec.player]
        self.tree_popup = TreePopup(self.app, state, dec, spec.use_fuzzy, self.toggle_tree)
        self._label_controls()

    def _close_overlays(self) -> None:
        if self.tree_popup is not None:
            self.tree_popup.destroy()
            self.tree_popup = None
        if self.pause_menu is not None:
            self.pause_menu.destroy()
            self.pause_menu = None
        if self.results is not None:
            self.results.destroy()
            self.results = None

    # ------------------------------------------------------------------ human input
    def _human_turn(self) -> bool:
        return self.phase == "human" and not self.paused

    def on_mouse_down(self) -> None:
        if not self._human_turn():
            return
        pos = self.app.cam_rig.mouse_to_floor()
        if pos is None:
            return
        p = self.match.state.to_move
        mid = self.view.pick(pos[0], pos[1], owner=p)
        if mid is not None:
            # press on one of your marbles: select it and start a slingshot drag
            self._select(mid)
            self.dragging = True
            self.drag_moved = False
        elif self.sel is not None:
            # plain click on the floor: point the aim at that spot (Space shoots)
            m = self.match.state.marbles[self.sel]
            if math.hypot(pos[0] - m.x, pos[1] - m.y) > 0.3:
                self.aim_angle = math.atan2(pos[1] - m.y, pos[0] - m.x)
                self.preview_dirty = True

    def on_mouse_up(self) -> None:
        if not self.dragging:
            return
        self.dragging = False
        if self.drag_moved and self._human_turn():
            self.fire_human()

    def on_space(self) -> None:
        if self._human_turn() and self.sel is not None:
            self.fire_human()
        elif self.mode in ("ai_vs_ai", "replay"):
            self.on_start_pause()

    def set_ability(self, ab: str) -> None:
        if not self._human_turn():
            return
        s = self.match.state
        if ab != "none" and s.ability_count(s.to_move, ABILITY_TO_KEY[ab]) <= 0:
            self.hud.show_banner("No uses left", T.DANGER, 0.8)
            return
        self.ability = ab
        self.preview_dirty = True

    def cycle_marble(self) -> None:
        if not self._human_turn():
            return
        own = self.match.state.alive(self.match.state.to_move)
        if not own:
            return
        ids = [m.mid for m in own]
        self._select(ids[(ids.index(self.sel) + 1) % len(ids)] if self.sel in ids else ids[0])

    def _select(self, mid: int) -> None:
        self.sel = mid
        self.view.set_selected(mid, T.PLAYER_COLORS[self.match.state.to_move])
        self.preview_dirty = True

    def _default_aim(self) -> None:
        s = self.match.state
        own, opp = s.alive(s.to_move), s.alive(1 - s.to_move)
        if not own:
            return
        self._select(own[0].mid)
        m = s.marbles[self.sel]
        if opp:
            t = min(opp, key=lambda o: math.hypot(o.x - m.x, o.y - m.y))
            self.aim_angle = math.atan2(t.y - m.y, t.x - m.x)
        else:
            self.aim_angle = math.atan2(-m.y, -m.x)
        self.aim_power = 0.6
        self.ability = "none"

    def _update_drag(self) -> None:
        if not self.dragging or self.sel is None:
            return
        pos = self.app.cam_rig.mouse_to_floor()
        if pos is None:
            return
        m = self.match.state.marbles[self.sel]
        vx, vy = m.x - pos[0], m.y - pos[1]          # slingshot: pull back, shoot forward
        d = math.hypot(vx, vy)
        if d > 0.5:
            self.drag_moved = True
            self.aim_angle = math.atan2(vy, vx)
            rules = self.match.cfg.rules
            self.aim_power = max(rules.power_min, min(rules.power_max, (d - 0.3) / 5.0))
            self.preview_dirty = True

    def _current_action(self) -> Action | None:
        if self.sel is None:
            return None
        return Action(self.sel, self.aim_angle % (2 * math.pi), self.aim_power, self.ability, "human")

    def fire_human(self) -> None:
        a = self._current_action()
        if a is None or not is_legal(self.match.state, a, self.match.cfg.rules):
            self.hud.show_banner("Illegal shot", T.DANGER, 0.8)
            return
        self.dragging = False
        self.view.clear_paths()
        self.view.clear_aim()
        self._execute(a, None)

    # ------------------------------------------------------------------ AI
    def _request_ai(self) -> None:
        p = self.match.state.to_move
        agent = self.match.agent(p)
        spec = self.match.specs[p].agent_spec()
        budget = agent.config.time_budget
        snap = agent.opponent_model.snapshot() if agent.use_opponent_model else None
        self.req = self.app.worker.request(self.match.state, spec, self.match.current_seed(), snap, budget,
                                           self.match.cfg)
        self.req_t0 = time.perf_counter()
        # hard deadline: the search enforces its own budget; this only catches a stuck worker
        self.req_deadline = budget * 1.5 + 4.0
        self.think_t0 = time.perf_counter()

    def _poll_ai(self) -> None:
        status, dec, err = self.app.worker.poll(self.req)
        if status == "pending":
            if time.perf_counter() - self.req_t0 > self.req_deadline:
                self.app.worker.cancel_all()
                self.req = None
                self._on_decision(self.match.fallback_decision("AI exceeded its time limit"))
            return
        self.req = None
        if status != "done" or dec is None:
            self._on_decision(self.match.fallback_decision(f"AI {status} {err}".strip()))
            return
        if not is_legal(self.match.state, dec.action, self.match.cfg.rules):
            dec = self.match.fallback_decision("AI returned an illegal action")
        self._on_decision(dec)

    def _on_decision(self, dec, action: Action | None = None) -> None:
        """Show the chosen shot (and, if enabled, the alternatives) before executing it."""
        p = self.match.state.to_move
        self.pending = (action or dec.action, dec)
        a = self.pending[0]
        if dec is not None:
            self.last_decisions[p] = dec
            self.hud.set_analysis(p, dec)
            if not self.match.specs[p].is_human:
                self.last_ai = (self.match.state, dec)
                try:
                    from ai.explain import explain
                    self.last_expl = explain(self.match.state, dec, self.match.arena, self.match.cfg)
                except Exception:          # an explanation must never stop the match
                    self.last_expl = None
                self.hud.set_explanation(self.last_expl)
        m = self.match.state.marbles[a.marble_id]
        col = T.PLAYER_COLORS[p]
        self.view.set_selected(a.marble_id, col)
        self.view.show_aim(m.x, m.y, a.angle, min(1.0, a.power), col)
        reveal = 0.45
        if self.show_analysis and self.show_traj:
            paths, labels = [], []
            chosen_rank = None
            for rank, c in enumerate(dec.candidates if dec is not None else [], start=1):
                ca = Action.from_dict(c["action"])
                if ca == a:
                    chosen_rank = rank
                    continue
                risky = (c.get("risk") or 0) >= 0.6
                style, rgba = ("dashed", (1.0, 0.3, 0.3, 0.75)) if risky else ("thin", (0.85, 0.9, 1.0, 0.45))
                if is_legal(self.match.state, ca, self.match.cfg.rules):
                    cp = preview_paths(self.match.state, ca, self.match.arena, self.match.cfg, rgba, style,
                                       False, 2.0)[0]
                    paths += cp
                    if cp and cp[0][0]:
                        x, y = cp[0][0][-1]
                        labels.append((x, y, f"#{rank}", (1.0, 0.45, 0.45, 1) if risky else (0.9, 0.93, 1.0, 1)))
            mine = preview_paths(self.match.state, a, self.match.arena, self.match.cfg,
                                 (col[0], col[1], col[2], 0.95), "solid", True, 2.5)[0]
            paths += mine
            if dec is not None and mine and mine[0][0]:
                x, y = mine[0][0][-1]
                labels.insert(0, (x, y, f"#{chosen_rank} CHOSEN" if chosen_rank else "CHOSEN", (1.0, 0.85, 0.35, 1)))
            self.view.show_paths(paths, labels)
            reveal = 1.1
        self.reveal_t = reveal
        self.phase = "reveal"

    # ------------------------------------------------------------------ replay
    def _replay_next(self) -> None:
        turns = self.replay["turns"]
        if self.replay_index >= len(turns):
            self._replay_fail("the recording ends before the match is over")
            return
        t = turns[self.replay_index]
        try:
            action = Action.from_dict(t["action"])
        except (KeyError, TypeError, ValueError):
            self._replay_fail(f"turn {self.replay_index + 1} has no valid recorded action")
            return
        if not is_legal(self.match.state, action, self.match.cfg.rules):
            self._replay_fail(f"turn {self.replay_index + 1}: recorded action is illegal here")
            return
        dec = Decision.from_dict(t["decision"]) if t.get("decision") else None
        self._on_decision(dec, action)

    def _replay_fail(self, msg: str) -> None:
        self.replay_error = msg
        self.phase = "over"
        self.view.clear_aim()
        self.view.clear_paths()
        self.hud.show_banner("REPLAY STOPPED", T.DANGER, 2.5)

    # ------------------------------------------------------------------ execution / playback
    def _execute(self, action: Action, decision) -> None:
        try:
            res = self.match.apply(action, decision, record=True)
        except IllegalActionError as exc:   # cannot happen for validated input; never crash the UI
            if self.mode == "replay":
                self._replay_fail(str(exc))
            else:
                self.hud.show_banner("Illegal shot", T.DANGER, 0.8)
            return
        if self.mode == "replay":
            rec_fp = self.replay["turns"][self.replay_index].get("fingerprint")
            self.replay_index += 1
            if rec_fp and res.state.fingerprint() != rec_fp:
                self.replay_error = f"turn {self.replay_index} diverged from the recording"
        self.playback = Playback(res.frames, res.contacts, list(zip(res.eliminated, res.elimination_times)), self.speed)
        audio = self.app.audio
        audio.play("launch_power" if action.ability == "power" else "launch", 0.5 + 0.5 * min(1.0, action.power))
        if action.ability in ABILITY_SFX:
            audio.play(ABILITY_SFX[action.ability])
        self.view.clear_aim()
        self.view.clear_paths()
        self.hud.set_power(False)
        self.phase = "animating"
        self.last_result = res

    def _animate(self, dt: float) -> None:
        pb = self.playback
        events = pb.advance(dt)
        pos, alive, vel = pb.sample()
        self.view.apply_sample(pos, alive, vel, dt * self.speed)
        speed_sum = sum(math.hypot(vx, vy) for (vx, vy), al in zip(vel, alive) if al)
        self.app.audio.set_rolling(min(1.0, speed_sum / 10.0))
        if self.match.arena.has_dynamic_obstacles:
            self.view.set_clock(self.match.state.clock - self.last_result.sim_time + pb.t)
        audio = self.app.audio
        for e in events:
            if e.kind == "contact":
                if e.b >= 0:
                    self.view.spawn_contact(e.x, e.y, e.strength)
                    audio.play("hit" if e.strength >= 2.0 else "hit_soft", 0.3 + e.strength / 8,
                               0.9 + min(0.3, e.strength / 25))
                else:
                    self.view.spawn_contact(e.x, e.y, e.strength, (1.0, 0.75, 0.3))
                    audio.play("bumper", 0.25 + e.strength / 10)
            elif e.kind == "eliminated":
                vx, vy = vel[e.a] if e.a < len(vel) else (0.0, 0.0)
                self.view.spawn_elimination(e.a, vx, vy)
                owner = self.match.state.marbles[e.a].owner
                shooter = self.match.history[-1].player
                audio.play("eliminate")
                if owner == shooter:
                    self.hud.show_banner("SELF KNOCKOUT!", T.DANGER, 1.2)
                else:
                    audio.play("knockout_cheer")
                    self.hud.show_banner("KNOCKOUT!", T.PLAYER_COLORS[shooter], 1.2)
        if pb.done:
            self.playback = None
            audio.set_rolling(0.0)
            self.view.set_state(self.match.state)
            self.view.clear_paths()
            self.view.set_selected(None)
            self.post_t = 0.45 / max(0.5, self.speed)
            self.phase = "post"

    def _finish_match(self) -> None:
        self.phase = "over"
        if not self.logged and self.mode != "replay":
            try:
                log_match(self.match, {"ai_vs_ai": "arena", "human": "human", "pvp": "pvp"}[self.mode])
            except OSError:
                pass
            try:
                self.replay_record = record_from_match(self.match, self.mode)
                save_replay(self.replay_record)
            except OSError:
                self.replay_record = None
            self.app.session_matches.append(self.match.summary())
            self.app.last_match_decisions = self.match.decision_records()
            if self.series is not None:
                self.series.record(self.match.summary())
        self.logged = True
        if self.series is not None and not self.series.is_over:
            self._announce_round()
            return
        w = self.series.winner() if self.series is not None else self.match.state.winner
        if w == DRAW:
            self.app.audio.play("draw")
            self.hud.show_banner("DRAW", T.GOLD, 2.0)
        else:
            humans = self.humans
            lost = len(humans) == 1 and w != humans[0]
            self.app.audio.play("defeat" if lost else "victory")
            what = " THE SERIES" if self.series is not None else ""
            self.hud.show_banner(f"{PLAYER_NAMES[w].upper()} WINS{what}", T.PLAYER_COLORS[w], 2.0)
        self.app.taskMgr.doMethodLater(1.6, self._show_results, "show-results")

    def _announce_round(self) -> None:
        """Series: a round ended but more rounds remain."""
        r = self.series.rounds[-1]
        n = len(self.series.rounds)
        if r.winner == DRAW:
            self.app.audio.play("draw")
            self.hud.show_banner(f"ROUND {n}: DRAW", T.GOLD, 2.0)
        else:
            self.app.audio.play("victory")
            self.hud.show_banner(f"ROUND {n}: {PLAYER_NAMES[r.winner].upper()} WINS ({r.score[0]}:{r.score[1]})",
                                 T.PLAYER_COLORS[r.winner], 2.0)
        self.app.taskMgr.doMethodLater(2.4, self._start_next_round, "next-round")

    def _show_results(self, task=None):
        from ui.results_screen import ResultsScreen
        if self.results is None and self.match.is_over:
            self.hud.banner.hide()
            self.hud.banner_t = 0.0
            can_replay = self.mode == "replay" or self.replay_record is not None
            series = None
            if self.series is not None:
                series = self.series.to_dict()
                w = series["winner"]
                series["winner_label"] = (f"{self.match.seat_label(0)}  vs  {self.match.seat_label(1)}" if w == DRAW
                                          else self.match.seat_label(w))
            self.results = ResultsScreen(self.app, self.match.summary(), on_rematch=self.on_restart,
                                         on_menu=self.on_menu, on_lab=lambda: self.app.goto("lab"),
                                         on_replay=self.watch_replay if can_replay else None,
                                         replay_mode=self.mode == "replay", series=series)
        return None

    # ------------------------------------------------------------------ frame update
    def _update(self, task):
        dt = min(0.1, self.app.clock_dt())
        self.app.cam_rig.update(dt)
        self._camera_keys(dt)
        self.view.update(dt)
        self.hud.update(dt)
        if not self.paused:
            self._step_phase(dt)
        self._refresh_hud()
        return task.cont

    def _camera_keys(self, dt: float) -> None:
        k = self.keys
        cam = self.app.cam_rig
        human = self._human_turn()
        if "q" in k or ("arrow_left" in k and not human):
            cam.orbit(-70 * dt, 0)
        if "e" in k or ("arrow_right" in k and not human):
            cam.orbit(70 * dt, 0)
        if "arrow_up" in k:
            cam.orbit(0, 40 * dt)
        if "arrow_down" in k:
            cam.orbit(0, -40 * dt)
        if human and self.sel is not None:
            changed = False
            if "a" in k or "arrow_left" in k:
                self.aim_angle += math.radians(45) * dt
                changed = True
            if "d" in k or "arrow_right" in k:
                self.aim_angle -= math.radians(45) * dt
                changed = True
            rules = self.match.cfg.rules
            if "w" in k:
                self.aim_power = min(rules.power_max, self.aim_power + 0.4 * dt)
                changed = True
            if "s" in k:
                self.aim_power = max(rules.power_min, self.aim_power - 0.4 * dt)
                changed = True
            if changed:
                self.preview_dirty = True

    def _step_phase(self, dt: float) -> None:
        ph = self.phase
        if ph == "turn_start":
            if self.match.is_over:
                self._finish_match()
                return
            if self.replay_error:
                self._replay_fail(self.replay_error)
                return
            if self.match.state.turn > 0:
                self.app.audio.play("turn")
            if self.mode == "replay":
                self._replay_next()
            elif self.match.is_ai_turn():
                self._request_ai()
                self.phase = "thinking"
            else:
                self._default_aim()
                self.phase = "human"
                p = self.match.state.to_move
                lead = f"PLAYER {p + 1}'S TURN" if self.mode == "pvp" else "YOUR TURN"
                self.hud.show_banner(lead, T.PLAYER_COLORS[p], 0.9)
        elif ph == "thinking":
            if time.perf_counter() - self.think_t0 >= 0.35 / self.speed or self.req is None:
                self._poll_ai()
        elif ph == "reveal":
            self.reveal_t -= dt * self.speed
            if self.reveal_t <= 0:
                action, d = self.pending
                self.pending = None
                self._execute(action, d)
        elif ph == "animating":
            self._animate(dt)
        elif ph == "post":
            self.post_t -= dt
            if self.post_t <= 0 and not self.view.busy:
                self.phase = "turn_start"
                if self.step_mode and not self.match.is_over:
                    self.step_mode = False
                    self.paused = True
                    self._label_controls()
        elif ph == "human":
            self._human_frame(dt)

    def _human_frame(self, dt: float) -> None:
        self._update_drag()
        s = self.match.state
        p = s.to_move
        abil = {ab: s.ability_count(p, ABILITY_TO_KEY[ab]) for ab in ("precision", "power", "spin_left", "spin_right")}
        self.hud.set_power(True, self.aim_power, self.ability, p, abil,
                           "Drag your marble (slingshot) or click the floor to aim | A/D W/S fine-tune | "
                           "1-5 ability | Space shoot")
        if self.sel is None:
            return
        self.preview_t -= dt
        if self.preview_dirty and self.preview_t <= 0:
            self.preview_dirty = False
            self.preview_t = 0.05
            m = s.marbles[self.sel]
            col = T.PLAYER_COLORS[p]
            self.view.show_aim(m.x, m.y, self.aim_angle, self.aim_power, col)
            a = self._current_action()
            if self.show_traj and a is not None and is_legal(s, a, self.match.cfg.rules):
                self.view.show_paths(preview_paths(s, a, self.match.arena, self.match.cfg,
                                                   (col[0], col[1], col[2], 0.9), "solid", True, 1.6)[0])
            else:
                self.view.clear_paths()

    # ------------------------------------------------------------------ teardown
    def destroy(self) -> None:
        if self._on_aspect in self.app.cam_rig.aspect_listeners:
            self.app.cam_rig.aspect_listeners.remove(self._on_aspect)
        self.app.taskMgr.remove(self.task)
        self.app.taskMgr.remove("show-results")
        self.app.taskMgr.remove("next-round")
        self.app.worker.cancel_all()
        self.app.audio.set_rolling(0.0)
        self._unbind()
        self._close_overlays()
        self.hud.destroy()
        self.ctrl_frame.destroy()
        self.view.destroy()
