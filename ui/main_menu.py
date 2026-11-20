"""Main menu, match-setup screens, arena selection and the How-to-Play page."""

from __future__ import annotations

import math
import os
import random

from direct.gui.DirectGui import DirectFrame

from config import GAME_SUBTITLE, GAME_TAGLINE, GAME_TITLE
from game.actions import Action
from game.arena import arena_ids, get_arena
from game.match import PlayerSpec
from game.rules import initial_state
from game.simulation import simulate_turn
from ui import theme as T
from ui.arena_view import ArenaView
from ui.playback import Playback
from ui.widgets import Cycler, index_of

DIFFS = [("easy", "Easy"), ("normal", "Normal"), ("hard", "Hard")]
ALGOS = [("minimax", "Minimax + Alpha-Beta"), ("mcts", "MCTS (UCT)")]
ON_OFF = [(True, "On"), (False, "Off")]
ROUNDS = [(1, "1 round"), (2, "2 rounds (swap first shot)")]


def arena_options():
    return [(a, get_arena(a).name) for a in arena_ids()]


class MenuBackground:
    """A slowly orbiting arena with occasional real-physics demo shots."""

    def __init__(self, app, arena_id: str) -> None:
        self.app = app
        self.rng = random.Random()
        self.arena = get_arena(arena_id)
        self.view = ArenaView(app, self.arena, app.game_cfg, quality=app.settings.get("graphics_quality", "high"),
                              effects=app.settings.get("effects", "full"))
        self.state = initial_state(self.arena, app.game_cfg)
        self.view.set_state(self.state)
        self.pb: Playback | None = None
        self.timer = 1.5
        app.cam_rig.clear_fit()
        app.cam_rig.set_view(heading=20, pitch=40, dist=27, snap=True)
        app.cam_rig.auto_orbit = 4.0
        self.task = app.taskMgr.add(self._update, "menu-bg")

    def _update(self, task):
        dt = min(0.1, self.app.clock_dt())
        self.app.cam_rig.update(dt)
        self.view.update(dt)
        if self.pb is not None:
            self.pb.advance(dt)
            pos, alive, vel = self.pb.sample()
            self.view.apply_sample(pos, alive, vel, dt)
            if self.arena.has_dynamic_obstacles:
                self.view.set_clock(self.state.clock - self.sim_time + self.pb.t)
            if self.pb.done:
                self.pb = None
                self.view.set_state(self.state)
        else:
            self.timer -= dt
            if self.timer <= 0:
                self._demo_shot()
        return task.cont

    def _demo_shot(self) -> None:
        if self.state.is_terminal or self.state.turn > 14:
            self.view.reset_marbles()
            self.state = initial_state(self.arena, self.app.game_cfg)
            self.view.set_state(self.state)
        p = self.state.to_move
        own, opp = self.state.alive(p), self.state.alive(1 - p)
        s = self.rng.choice(own)
        t = self.rng.choice(opp)
        ang = math.atan2(t.y - s.y, t.x - s.x) + self.rng.uniform(-0.12, 0.12)
        res = simulate_turn(self.state, Action(s.mid, ang, self.rng.uniform(0.45, 0.8)), self.arena, self.app.game_cfg,
                            record=True)
        self.state = res.state
        self.sim_time = res.sim_time
        self.pb = Playback(res.frames, [], [], 1.0)
        self.timer = 2.5

    def destroy(self) -> None:
        self.app.taskMgr.remove(self.task)
        self.view.destroy()
        self.app.cam_rig.auto_orbit = 0.0


class Screen:
    """Base class: a full-screen DirectGUI layer."""

    def __init__(self, app) -> None:
        self.app = app
        self.root = DirectFrame(parent=app.aspect2d, frameColor=(0, 0, 0, 0))

    def destroy(self) -> None:
        self.root.destroy()

    def header(self, title: str, subtitle: str = "") -> None:
        T.text(self.root, title, (0, 0.83), 0.085, T.TEXT, "center", "title")
        if subtitle:
            T.text(self.root, subtitle, (0, 0.77), 0.036, T.MUTED, "center")


class MainMenu(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        left = DirectFrame(parent=app.a2dLeftCenter, frameColor=(0.02, 0.03, 0.07, 0.78), frameSize=(0, 1.08, -1.0, 1.0))
        self.left = left
        DirectFrame(parent=left, frameColor=(T.EDGE[0], T.EDGE[1], T.EDGE[2], 0.6), frameSize=(1.075, 1.08, -1, 1))
        T.text(left, GAME_TITLE, (0.1, 0.62), 0.13, T.TEXT, kind="title")
        DirectFrame(parent=left, frameColor=T.P1, frameSize=(0.1, 0.5, 0.585, 0.595))
        DirectFrame(parent=left, frameColor=T.P2, frameSize=(0.5, 0.92, 0.585, 0.595))
        T.text(left, GAME_SUBTITLE, (0.1, 0.52), 0.045, T.P1)
        T.text(left, GAME_TAGLINE, (0.1, 0.465), 0.034, T.MUTED)
        items = [
            ("AI vs AI ARENA", lambda: app.goto("setup_ai"), T.GOLD),
            ("HUMAN vs AI", lambda: app.goto("setup_human"), None),
            ("HUMAN vs HUMAN", lambda: app.goto("setup_pvp"), None),
            ("TOURNAMENT MODE", lambda: app.goto("tournament"), None),
            ("REPLAYS", lambda: app.goto("replays"), None),
            ("ALGORITHM LAB", lambda: app.goto("lab"), None),
            ("ARENA SELECTION", lambda: app.goto("arenas"), None),
            ("SETTINGS", lambda: app.goto("settings"), None),
            ("HOW TO PLAY", lambda: app.goto("howto"), None),
            ("EXIT", app.quit, T.DANGER),
        ]
        for i, (lab, cmd, acc) in enumerate(items):
            b = T.button(left, lab, cmd, pos=(0.5, 0, 0.34 - i * 0.087), size=(0.4, 0.036), text_scale=0.036,
                         accent=acc)
            if i == 0:
                T.set_button_active(b, True)
        T.text(left, "Featured: watch Minimax and MCTS battle on their own.", (0.1, -0.6), 0.028, T.GOLD)
        T.text(left, f"Arena: {get_arena(app.settings['arena']).name}   |   Difficulty: {app.settings['difficulty']}",
               (0.1, -0.66), 0.028, T.MUTED)
        T.text(left, "Inspired by Kancha, the traditional Bangladeshi marble game.", (0.1, -0.9), 0.026, T.MUTED)

    def destroy(self) -> None:
        self.left.destroy()
        super().destroy()


class SetupBase(Screen):
    def __init__(self, app, title: str, subtitle: str) -> None:
        super().__init__(app)
        self.panel = T.panel(app.aspect2d, (-0.85, 0.85, -0.62, 0.62), (0, 0, -0.05))
        T.text(self.panel, title, (0, 0.5), 0.07, T.TEXT, "center", "title")
        T.text(self.panel, subtitle, (0, 0.43), 0.032, T.MUTED, "center", wordwrap=48)
        self.rows: list[Cycler] = []
        self.row_step = 0.1

    def row(self, label, options, value, on_change=None) -> Cycler:
        i = len(self.rows)
        c = Cycler(self.panel, label, options, index_of(options, value), (-0.75, 0.32 - i * self.row_step), on_change,
                   width=1.5, label_w=0.62)
        self.rows.append(c)
        return c

    def rounds_row(self) -> Cycler:
        return self.row("Rounds", ROUNDS, self.app.settings.get("rounds", 1))

    def chosen_rounds(self) -> int:
        """Selected round count; remembered for the next match."""
        n = int(self.rounds.get())
        if self.app.settings.get("rounds") != n:
            self.app.settings["rounds"] = n
            self.app.save_settings()
        return n

    def destroy(self) -> None:
        self.panel.destroy()
        super().destroy()


MATCHUPS = [(("minimax", "mcts"), "Minimax  vs  MCTS"), (("mcts", "minimax"), "MCTS  vs  Minimax"),
            (("minimax", "minimax"), "Minimax  vs  Minimax"), (("mcts", "mcts"), "MCTS  vs  MCTS")]
SEEDS = [(None, "Random"), (1, "1"), (7, "7"), (42, "42"), (2024, "2024")]


class AIvsAISetup(SetupBase):
    def __init__(self, app) -> None:
        super().__init__(app, "AI vs AI ARENA",
                         "Two independent agents play the whole match by themselves. Pick the matchup and budgets.")
        st = app.settings
        self.row_step = 0.092
        self.matchup = self.row("Matchup (P1 vs P2)", MATCHUPS, ("minimax", "mcts"))
        self.d1 = self.row("Player 1 difficulty", DIFFS, st["difficulty"])
        self.d2 = self.row("Player 2 difficulty", DIFFS, st["difficulty"])
        self.arena = self.row("Arena", arena_options(), st["arena"])
        self.marbles = self.row("Marbles per player", [(n, str(n)) for n in (2, 3, 4, 5)], st["marbles_per_player"])
        self.fuzzy = self.row("Fuzzy risk evaluator", ON_OFF, st["use_fuzzy"])
        self.adapt = self.row("History-Based Adaptive AI", ON_OFF, st["use_opponent_model"])
        self.seed = self.row("Random seed (MCTS)", SEEDS, 42)
        self.rounds = self.rounds_row()
        T.button(self.panel, "BACK", lambda: app.goto("menu"), pos=(-0.42, 0, -0.53), size=(0.3, 0.05))
        b = T.button(self.panel, "ENTER ARENA", self.start, pos=(0.42, 0, -0.53), size=(0.3, 0.05), accent=T.GOLD)
        T.set_button_active(b, True)

    def start(self) -> None:
        k1, k2 = self.matchup.get()
        f, a = self.fuzzy.get(), self.adapt.get()
        specs = (PlayerSpec(k1, self.d1.get(), f, a), PlayerSpec(k2, self.d2.get(), f, a))
        seed = self.seed.get()
        if seed is None:
            seed = random.randrange(1, 10**6)
        self.app.goto("game", mode="ai_vs_ai", specs=specs, arena_id=self.arena.get(), marbles=self.marbles.get(),
                      seed=seed, rounds=self.chosen_rounds())


class HumanSetup(SetupBase):
    def __init__(self, app) -> None:
        super().__init__(app, "HUMAN vs AI",
                         "You control one side with the mouse; the AI decides its own shots with real search.")
        st = app.settings
        self.algo = self.row("Opponent algorithm", ALGOS, "minimax")
        self.diff = self.row("Opponent difficulty", DIFFS, st["difficulty"])
        self.side = self.row("You play as", [(0, "Player 1 (shoots first)"), (1, "Player 2")], 0)
        self.arena = self.row("Arena", arena_options(), st["arena"])
        self.marbles = self.row("Marbles per player", [(n, str(n)) for n in (2, 3, 4, 5)], st["marbles_per_player"])
        self.rounds = self.rounds_row()
        T.text(self.panel, "Difficulty changes the search budget: depth / beam width for Minimax,",
               (0, -0.3), 0.028, T.MUTED, "center")
        T.text(self.panel, "iterations / rollout depth for MCTS (see config.py DIFFICULTY_PRESETS).",
               (0, -0.34), 0.028, T.MUTED, "center")
        T.button(self.panel, "BACK", lambda: app.goto("menu"), pos=(-0.42, 0, -0.53), size=(0.3, 0.05))
        b = T.button(self.panel, "START MATCH", self.start, pos=(0.42, 0, -0.53), size=(0.3, 0.05), accent=T.GOLD)
        T.set_button_active(b, True)

    def start(self) -> None:
        st = self.app.settings
        ai = PlayerSpec(self.algo.get(), self.diff.get(), st["use_fuzzy"], st["use_opponent_model"])
        human = PlayerSpec("human")
        specs = (human, ai) if self.side.get() == 0 else (ai, human)
        self.app.goto("game", mode="human", specs=specs, arena_id=self.arena.get(), marbles=self.marbles.get(),
                      seed=random.randrange(1, 10**6), rounds=self.chosen_rounds())


class PvPSetup(SetupBase):
    def __init__(self, app) -> None:
        super().__init__(app, "HUMAN vs HUMAN",
                         "Two players share this computer and take alternating turns with the same rules and physics.")
        st = app.settings
        self.arena = self.row("Arena", arena_options(), st["arena"])
        self.marbles = self.row("Marbles per player", [(n, str(n)) for n in (2, 3, 4, 5)], st["marbles_per_player"])
        self.rounds = self.rounds_row()
        T.text(self.panel, "Player 1 (cyan) shoots first; with 2 rounds, Player 2 shoots first in round 2.",
               (0, -0.06), 0.029, T.MUTED, "center")
        T.text(self.panel, "The next player can act only after every marble has stopped. Controls: see How to Play.",
               (0, -0.1), 0.029, T.MUTED, "center")
        T.button(self.panel, "BACK", lambda: app.goto("menu"), pos=(-0.42, 0, -0.53), size=(0.3, 0.05))
        b = T.button(self.panel, "START MATCH", self.start, pos=(0.42, 0, -0.53), size=(0.3, 0.05), accent=T.GOLD)
        T.set_button_active(b, True)

    def start(self) -> None:
        self.app.goto("game", mode="pvp", specs=(PlayerSpec("human"), PlayerSpec("human")),
                      arena_id=self.arena.get(), marbles=self.marbles.get(), seed=random.randrange(1, 10**6),
                      rounds=self.chosen_rounds())


class ReplaysScreen(Screen):
    """Recent replay files; each is verified by re-simulation before it is listed as OK."""

    def __init__(self, app) -> None:
        super().__init__(app)
        from game.replay import ReplayError, list_replays, load_replay, verify_replay
        DirectFrame(parent=self.root, frameColor=(0.02, 0.03, 0.06, 0.85), frameSize=(-3, 3, -1, 1))
        self.header("REPLAYS", "Recorded matches are re-simulated from their legal actions with the real rules.")
        files = list_replays(limit=12)
        if not files:
            p = T.panel(self.root, (-0.9, 0.9, -0.2, 0.2), (0, 0, 0.1))
            T.text(p, "NO REPLAYS YET", (0, 0.06), 0.06, T.WARN, "center", "title")
            T.text(p, "Finish any match (or run a tournament) and its replay is saved automatically.",
                   (0, -0.04), 0.032, T.MUTED, "center")
        heads = [("Recorded", -1.45), ("Mode", -1.0), ("Arena", -0.78), ("Player 1 vs Player 2", -0.42),
                 ("Result", 0.38), ("Turns", 0.78), ("Verification", 0.93)]
        if files:
            for h, x in heads:
                T.text(self.root, h, (x, 0.66), 0.03, T.MUTED)
        for i, f in enumerate(files):
            y = 0.58 - i * 0.105
            try:
                rec = load_replay(f)
                v = verify_replay(rec)
                ok, msg = v.ok, ("verified" if v.ok else "diverges")
            except ReplayError as exc:
                rec, ok, msg = None, False, "unreadable"
            if rec is None:
                T.text(self.root, os.path.basename(f) + f"  ({msg})", (-1.45, y), 0.028, T.DANGER)
                continue
            labels = " vs ".join(("Human" if s["kind"] == "human" else s["kind"].upper()) for s in rec["specs"])
            res = rec.get("result", {})
            w = res.get("winner")
            result = "Draw" if w == -1 else (f"P{w + 1} wins {res['score'][0]}-{res['score'][1]}" if w in (0, 1) else "?")
            cols = [rec.get("created", "")[5:16], rec.get("source", ""), rec["arena"], labels, result,
                    str(len(rec["turns"])), msg]
            for (h, x), val in zip(heads, cols):
                T.text(self.root, val, (x, y), 0.028, (T.GOOD if ok else T.DANGER) if h == "Verification" else T.TEXT)
            T.button(self.root, "WATCH", lambda r=rec: app.goto("game", mode="replay", replay=r),
                     pos=(1.42, 0, y + 0.012), size=(0.12, 0.034), text_scale=0.03)
        T.button(self.root, "BACK", lambda: app.goto("menu"), pos=(0, 0, -0.93), size=(0.25, 0.045))


ARENA_NOTES = {
    "classic": ["Warm sandy court, the default learning arena", "No obstacles | sand: x1.4 rolling friction",
                "Pure aiming and positioning"],
    "neon": ["Dark metallic esports floor", "4 static bumpers (restitution 0.8)",
             "Grip Core x2.2 friction | Glide Pads x0.4"],
    "magnetic": ["2 rotating bumpers (angle = phase + 0.6 rad/s x clock)", "Magnetic vortices pull rolling marbles",
                 "Slip hazards x0.35 friction near the rim"],
}


class ArenaSelect(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.header("ARENA SELECTION", "The highlighted arena is used by default; the live preview shows it.")
        self.cards = []
        ids = arena_ids()
        for i, aid in enumerate(ids):
            a = get_arena(aid)
            x = (i - (len(ids) - 1) / 2) * 0.95
            c = T.panel(self.root, (-0.44, 0.44, -0.3, 0.3), (x, 0, -0.55))
            T.text(c, a.name, (0, 0.2), 0.042, T.TEXT, "center", "title")
            for k, line in enumerate(ARENA_NOTES.get(aid, [a.theme])):
                T.text(c, line, (0, 0.12 - k * 0.055), 0.027, T.MUTED, "center", wordwrap=30)
            b = T.button(c, "SELECT", self.pick, extra_args=[aid], pos=(0, 0, -0.22), size=(0.2, 0.04))
            self.cards.append((aid, b))
        T.button(self.root, "BACK", lambda: app.goto("menu"), pos=(0, 0, -0.93), size=(0.25, 0.045))
        self._mark()

    def pick(self, aid: str) -> None:
        self.app.settings["arena"] = aid
        self.app.save_settings()
        self.app.set_menu_arena(aid)
        self._mark()

    def _mark(self) -> None:
        for aid, b in self.cards:
            T.set_button_active(b, aid == self.app.settings["arena"])
            b["text"] = "SELECTED" if aid == self.app.settings["arena"] else "SELECT"


HOWTO = [
    ("Goal", "Knock every opposing marble out of the circular arena while keeping your own inside. "
             "A marble is out the moment its centre crosses the glowing boundary ring."),
    ("Turns", "Players alternate. On your turn pick one of your marbles, a direction and a power, optionally an "
              "ability, then shoot. All marbles roll, collide and settle before the next turn."),
    ("Abilities (2 uses each)", "Precision: your shooter brakes hard after its first hit (stop shot). "
                                "Power: 1.35x launch speed. Spin: the shot curves left or right."),
    ("End of match", "Win by eliminating all enemy marbles. If both sides lose their last marble on the same "
                     "shot it is a draw. After 60 shots: more marbles wins, then larger total distance from the rim."),
    ("Mouse", "Left-drag from one of your marbles like a slingshot: pull back, release to shoot. A plain click on "
              "the floor points the aim there (Space shoots). Right-drag rotates the camera, wheel zooms."),
    ("Keyboard", "A/D rotate aim | W/S power | 1 normal, 2 precision, 3 power, 4/5 spin | Tab next marble | "
                 "Space shoot | H explanations | P pause | Esc pause menu | Q/E/arrows camera | C re-centre camera."),
    ("AI vs AI", "Choose a matchup and press START MATCH. Use Pause, Step One Turn, Speed and Explanations; the "
                 "agents play without any human input."),
    ("Human vs Human", "Two players share the mouse and keyboard and alternate turns. The next player can act only "
                       "after every marble has stopped."),
    ("Replays", "Every finished match is saved. Watch it from the results screen or REPLAYS; it is re-simulated "
                "from the recorded actions and stops if anything differs. M mutes audio."),
]


class HowToPlay(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        p = T.panel(self.root, (-1.2, 1.2, -0.82, 0.82), (0, 0, 0))
        self.p = p
        T.text(p, "HOW TO PLAY", (0, 0.7), 0.07, T.TEXT, "center", "title")
        y = 0.6
        for head, body in HOWTO:
            T.text(p, head, (-1.12, y), 0.036, T.EDGE, kind="title")
            T.text(p, body, (-0.6, y), 0.029, T.TEXT, wordwrap=60)
            lines = max(1, len(body) // 58 + 1)
            y -= max(0.085, 0.036 * lines + 0.03)
        T.button(p, "BACK", lambda: app.goto("menu"), pos=(0, 0, -0.74), size=(0.25, 0.045))
