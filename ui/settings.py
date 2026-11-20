"""Settings screen (persisted to data/settings.json).

Every option shown here is wired to real behaviour:

* gameplay defaults -> setup screens; fuzzy/adaptive -> new AI agents
* AI decision visualisation / trajectory previews -> GameScreen overlay and lines
* shot animation speed -> Human-vs-AI and Human-vs-Human playback
* AI-vs-AI playback speed -> AI-vs-AI matches and replays (also changeable in-game)
* graphics quality -> anti-aliasing, marble mesh detail, rim light, motion trails
* visual effects -> collision sparks, elimination bursts, trails
* FPS meter, master / music / effects volume and mute -> applied immediately
"""

from __future__ import annotations

from config import load_settings
from ui import theme as T
from ui.main_menu import DIFFS, ON_OFF, Screen, arena_options
from ui.widgets import Cycler, index_of

VOLUME = [(round(v / 10, 1), f"{v * 10}%") for v in range(0, 11)]
SPEED = [(0.5, "0.5x"), (1.0, "1x"), (2.0, "2x"), (4.0, "4x")]
LIVE_AUDIO = ("master_volume", "music_volume", "sfx_volume", "muted")


class SettingsScreen(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        st = app.settings
        p = T.panel(self.root, (-1.5, 1.5, -0.8, 0.8), (0, 0, 0))
        self.p = p
        T.text(p, "SETTINGS", (0, 0.68), 0.07, T.TEXT, "center", "title")
        left = [
            ("arena", "Default arena", arena_options()),
            ("difficulty", "Default AI difficulty", DIFFS),
            ("marbles_per_player", "Marbles per player", [(n, str(n)) for n in (2, 3, 4, 5)]),
            ("use_fuzzy", "Fuzzy risk evaluator", ON_OFF),
            ("use_opponent_model", "History-Based Adaptive AI", ON_OFF),
            ("show_analysis", "AI decision visualisation", ON_OFF),
            ("show_trajectories", "Trajectory previews", ON_OFF),
            ("anim_speed", "Shot animation speed", SPEED),
            ("ai_playback_speed", "AI-vs-AI playback speed", SPEED),
        ]
        right = [
            ("graphics_quality", "Graphics quality", [("low", "Low"), ("medium", "Medium"), ("high", "High")]),
            ("effects", "Visual effects", [("off", "Off"), ("reduced", "Reduced"), ("full", "Full")]),
            ("show_fps", "FPS meter", ON_OFF),
            ("master_volume", "Master volume", VOLUME),
            ("music_volume", "Music volume", VOLUME),
            ("sfx_volume", "Sound-effects volume", VOLUME),
            ("muted", "Mute all audio", [(False, "Sound on"), (True, "Muted")]),
        ]
        T.text(p, "GAMEPLAY & AI", (-1.42, 0.55), 0.036, T.EDGE, kind="title")
        T.text(p, "GRAPHICS & AUDIO", (0.06, 0.55), 0.036, T.EDGE, kind="title")
        self.cyclers = {}
        for col, rows, x in ((0, left, -1.42), (1, right, 0.06)):
            for i, (key, lab, opts) in enumerate(rows):
                cb = (lambda v, k=key: self._live(k, v)) if key in LIVE_AUDIO else None
                self.cyclers[key] = Cycler(p, lab, opts, index_of(opts, st[key], 0), (x, 0.46 - i * 0.095), cb,
                                           width=1.36, label_w=0.66, scale=0.038)
        T.text(p, "Volume changes play immediately. Graphics quality and effects apply to the next arena shown.",
               (0.06, -0.24), 0.025, T.MUTED, wordwrap=54)
        T.text(p, "Audio: original music and sound effects in assets/audio (python -m tools.make_audio).",
               (0.06, -0.32), 0.025, T.MUTED, wordwrap=54)
        if app.audio.warnings:
            T.text(p, f"{len(app.audio.warnings)} audio file(s) missing - the game runs silently for those.",
                   (0.06, -0.4), 0.025, T.WARN, wordwrap=54)
        T.button(p, "CANCEL", self.cancel, pos=(-0.42, 0, -0.7), size=(0.3, 0.05))
        b = T.button(p, "SAVE", self.save, pos=(0.42, 0, -0.7), size=(0.3, 0.05), accent=T.GOLD)
        T.set_button_active(b, True)

    def _live(self, key: str, value) -> None:
        self.app.settings[key] = value
        self.app.apply_audio_settings()
        self.app.audio.play("hit")            # audible feedback at the new volume

    def cancel(self) -> None:
        self.app.settings = load_settings()   # discard unsaved (live) changes
        self.app.apply_audio_settings()
        self.app.goto("menu")

    def save(self) -> None:
        for k, c in self.cyclers.items():
            self.app.settings[k] = c.get()
        self.app.save_settings()
        self.app.apply_audio_settings()
        self.app.apply_graphics_settings()
        self.app.set_menu_arena(self.app.settings["arena"], force=True)
        self.app.goto("menu")
