"""Panda3D application shell: window, camera, shared services and screen switching."""

from __future__ import annotations

from panda3d.core import loadPrcFileData


def configure_panda(windowed_size=(1600, 900), offscreen: bool = False, fullscreen: bool = False) -> None:
    loadPrcFileData("", f"""
window-title Marble Fight - Intelligent Marble Battle Arena
win-size {windowed_size[0]} {windowed_size[1]}
fullscreen {'#t' if fullscreen else '#f'}
framebuffer-multisample 1
multisamples 4
sync-video 1
show-frame-rate-meter 0
textures-power-2 none
notify-level-display error
notify-level-express error
notify-level-pnmtext error
default-directnotify-level warning
""")
    if offscreen:
        loadPrcFileData("", "window-type offscreen\naudio-library-name null")


from direct.showbase.ShowBase import ShowBase  # noqa: E402

from ai.worker import AIWorker  # noqa: E402
from config import DEFAULT_CONFIG, load_settings, save_settings  # noqa: E402
from ui.camera import CameraRig  # noqa: E402
from ui.audio import AudioManager, PandaBackend  # noqa: E402


class MarbleFightApp(ShowBase):
    def __init__(self, use_process_worker: bool = True, audio_backend=None) -> None:
        super().__init__()
        self.setBackgroundColor(0.02, 0.025, 0.06, 1)
        self.disableMouse()
        self.game_cfg = DEFAULT_CONFIG
        self.settings = load_settings()
        self.audio = AudioManager(audio_backend or PandaBackend(self), self.settings)
        for w in self.audio.warnings:
            print("[audio]", w, "- run: python -m tools.make_audio")
        self.sound = self.audio            # older call sites use app.sound.play(...)
        self.apply_graphics_settings()
        self.cam_rig = CameraRig(self)
        self.worker = AIWorker(use_processes=use_process_worker)
        self.worker.warm_up()
        self.session_matches: list[dict] = []
        self.last_match_decisions: list[dict] = []
        self.screen = None
        self.menu_bg = None
        self.screen_name = ""
        self.exitFunc = self._on_exit
        self.goto("menu")

    def clock_dt(self) -> float:
        from direct.showbase.ShowBaseGlobal import globalClock
        return globalClock.getDt()

    def save_settings(self) -> None:
        save_settings(self.settings)

    def apply_graphics_settings(self) -> None:
        """Apply the settings that affect the whole window immediately."""
        from panda3d.core import AntialiasAttrib
        q = self.settings.get("graphics_quality", "high")
        if q == "low":
            self.render.setAntialias(AntialiasAttrib.MNone)
        else:
            self.render.setAntialias(AntialiasAttrib.MMultisample)
        self.setFrameRateMeter(bool(self.settings.get("show_fps", False)))

    def apply_audio_settings(self) -> None:
        self.audio.apply_settings(self.settings)

    # ------------------------------------------------------------------ screens
    def _ensure_menu_bg(self) -> None:
        if self.menu_bg is None:
            from ui.main_menu import MenuBackground
            self.menu_bg = MenuBackground(self, self.settings.get("arena", "classic"))

    def set_menu_arena(self, arena_id: str, force: bool = False) -> None:
        if self.menu_bg is not None and (force or self.menu_bg.arena.arena_id != arena_id):
            self.menu_bg.destroy()
            self.menu_bg = None
            self._ensure_menu_bg()

    def goto(self, name: str, **kw) -> None:
        from ui.algorithm_lab import AlgorithmLab
        from ui.game_screen import GameScreen
        from ui.main_menu import AIvsAISetup, ArenaSelect, HowToPlay, HumanSetup, MainMenu, PvPSetup, ReplaysScreen
        from ui.settings import SettingsScreen
        from ui.tournament_screen import TournamentScreen

        self.audio.play("ui")
        if self.screen is not None:
            self.screen.destroy()
            self.screen = None
        if name == "game":
            if self.menu_bg is not None:
                self.menu_bg.destroy()
                self.menu_bg = None
            self.audio.play_music("game")
            self.screen = GameScreen(self, **kw)
        else:
            self.audio.set_rolling(0.0)
            self.audio.play_music("menu")
            self._ensure_menu_bg()
            cls = {"menu": MainMenu, "setup_ai": AIvsAISetup, "setup_human": HumanSetup, "setup_pvp": PvPSetup,
                   "arenas": ArenaSelect, "settings": SettingsScreen, "howto": HowToPlay, "lab": AlgorithmLab,
                   "tournament": TournamentScreen, "replays": ReplaysScreen}[name]
            self.screen = cls(self, **kw)
        self.screen_name = name

    def quit(self) -> None:
        self.userExit()

    def _on_exit(self) -> None:
        try:
            self.audio.stop_all()
            self.worker.shutdown()
        except Exception:
            pass
