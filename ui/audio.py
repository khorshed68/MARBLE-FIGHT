"""Music and sound-effect system.

* Assets: original WAV files in ``assets/audio`` (made by ``python -m tools.make_audio``).
  A missing or unreadable file is skipped with a one-time warning; the game keeps running.
* Volumes: effective music = master x music, effective SFX = master x sfx x event gain;
  ``muted`` silences everything. Settings are saved in data/settings.json.
* Events (``play``) come from real game events: shot launch, ability use, each
  recorded collision contact, each elimination, turn changes and the match result.
  Nothing is triggered per frame except the rolling loop's *volume*, which follows
  the actual marble speeds during playback.
* Anti-spam: every event has a cooldown and a voice pool; at most
  ``MAX_STARTS_PER_WINDOW`` sounds may start in any 0.1 s window, so a pile-up of
  collisions cannot produce a wall of noise.

The backend is injectable so the logic is unit-tested without a sound card.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass

from config import PROJECT_ROOT

AUDIO_DIR = os.path.join(PROJECT_ROOT, "assets", "audio")
MAX_STARTS_PER_WINDOW = 6
WINDOW = 0.1


@dataclass(frozen=True)
class SfxSpec:
    file: str
    gain: float = 1.0
    cooldown: float = 0.05     # min seconds between two starts of this event
    voices: int = 1            # simultaneous copies allowed


EVENTS: dict[str, SfxSpec] = {
    "launch": SfxSpec("launch", 0.8, 0.1, 1),
    "launch_power": SfxSpec("launch_power", 0.9, 0.1, 1),
    "hit": SfxSpec("hit", 0.75, 0.035, 4),
    "hit_soft": SfxSpec("hit_soft", 0.55, 0.035, 3),
    "bumper": SfxSpec("bumper", 0.6, 0.06, 2),
    "eliminate": SfxSpec("eliminate", 0.9, 0.12, 2),
    "knockout_cheer": SfxSpec("knockout_cheer", 0.6, 0.3, 1),
    "ability_precision": SfxSpec("ability_precision", 0.7, 0.2, 1),
    "ability_power": SfxSpec("ability_power", 0.8, 0.2, 1),
    "ability_spin": SfxSpec("ability_spin", 0.7, 0.2, 1),
    "turn": SfxSpec("turn", 0.45, 0.3, 1),
    "ui": SfxSpec("ui", 0.4, 0.06, 2),
    "victory": SfxSpec("victory", 0.85, 1.0, 1),
    "defeat": SfxSpec("defeat", 0.85, 1.0, 1),
    "draw": SfxSpec("draw", 0.85, 1.0, 1),
}
MUSIC_TRACKS = {"menu": "menu", "game": "game"}


# ----------------------------------------------------------------------------- backends
class NullSound:
    """Stand-in used for missing files and in tests."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.volume = 1.0
        self.rate = 1.0
        self.loop = False
        self.playing = False
        self.plays = 0

    def play(self) -> None:
        self.playing = True
        self.plays += 1

    def stop(self) -> None:
        self.playing = False

    def setVolume(self, v: float) -> None:
        self.volume = v

    def setPlayRate(self, r: float) -> None:
        self.rate = r

    def setLoop(self, flag: bool) -> None:
        self.loop = flag

    def is_playing(self) -> bool:
        return self.playing


class NullBackend:
    def __init__(self, missing: set[str] | None = None) -> None:
        self.missing = missing or set()
        self.loaded: list[str] = []

    def load(self, path: str, music: bool = False):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in self.missing:
            return None
        self.loaded.append(name)
        return NullSound(name)


class PandaBackend:
    def __init__(self, base) -> None:
        self.base = base

    def load(self, path: str, music: bool = False):
        if not os.path.exists(path):
            return None
        from panda3d.core import Filename
        fn = Filename.fromOsSpecific(path)
        try:
            snd = self.base.loader.loadMusic(fn) if music else self.base.loader.loadSfx(fn)
        except Exception:
            return None
        if snd is None or type(snd).__name__ == "NullAudioSound":
            return None
        return _PandaSound(snd)


class _PandaSound:
    def __init__(self, snd) -> None:
        self.snd = snd

    def play(self) -> None:
        self.snd.play()

    def stop(self) -> None:
        self.snd.stop()

    def setVolume(self, v: float) -> None:
        self.snd.setVolume(v)

    def setPlayRate(self, r: float) -> None:
        self.snd.setPlayRate(r)

    def setLoop(self, flag: bool) -> None:
        self.snd.setLoop(flag)

    def is_playing(self) -> bool:
        return self.snd.status() == self.snd.PLAYING


# ----------------------------------------------------------------------------- manager
class AudioManager:
    def __init__(self, backend, settings: dict, audio_dir: str = AUDIO_DIR) -> None:
        self.backend = backend
        self.audio_dir = audio_dir
        self.warnings: list[str] = []
        self.log: deque = deque(maxlen=500)          # (time, event) actually started
        self.counts: dict[str, int] = {}
        self.suppressed = 0                           # starts refused by cooldown/limits
        self.master = self.music_vol = self.sfx_vol = 1.0
        self.muted = False
        self._pools: dict[str, list] = {}
        self._next: dict[str, int] = {}
        self._last: dict[str, float] = {}
        self._recent: deque = deque()
        self._music: dict[str, object] = {}
        self.current_track: str | None = None
        self._roll = None
        self._roll_level = 0.0
        self._load()
        self.apply_settings(settings)

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        for event, spec in EVENTS.items():
            pool = []
            for _ in range(spec.voices):
                snd = self.backend.load(os.path.join(self.audio_dir, "sfx", f"{spec.file}.wav"))
                if snd is None:
                    break
                pool.append(snd)
            if not pool:
                self.warnings.append(f"missing sound effect: {spec.file}.wav")
            self._pools[event] = pool
            self._next[event] = 0
        for track, file in MUSIC_TRACKS.items():
            snd = self.backend.load(os.path.join(self.audio_dir, "music", f"{file}.wav"), music=True)
            if snd is None:
                self.warnings.append(f"missing music: {file}.wav")
                continue
            snd.setLoop(True)
            self._music[track] = snd
        self._roll = self.backend.load(os.path.join(self.audio_dir, "sfx", "roll.wav"))
        if self._roll is None:
            self.warnings.append("missing sound effect: roll.wav")
        else:
            self._roll.setLoop(True)

    # ------------------------------------------------------------------ settings
    def apply_settings(self, settings: dict) -> None:
        self.master = _clamp(settings.get("master_volume", 0.8))
        self.music_vol = _clamp(settings.get("music_volume", 0.6))
        self.sfx_vol = _clamp(settings.get("sfx_volume", 0.9))
        self.muted = bool(settings.get("muted", False))
        self._apply_music_volume()
        self._apply_roll_volume()

    def music_gain(self) -> float:
        return 0.0 if self.muted else self.master * self.music_vol

    def sfx_gain(self) -> float:
        return 0.0 if self.muted else self.master * self.sfx_vol

    def _apply_music_volume(self) -> None:
        for name, snd in self._music.items():
            snd.setVolume(self.music_gain() if name == self.current_track else 0.0)

    # ------------------------------------------------------------------ music
    def play_music(self, track: str) -> None:
        if track == self.current_track:
            return
        old = self._music.get(self.current_track) if self.current_track else None
        if old is not None:
            old.stop()
        self.current_track = track
        snd = self._music.get(track)
        if snd is not None:
            snd.setVolume(self.music_gain())
            snd.play()

    def stop_music(self) -> None:
        for snd in self._music.values():
            snd.stop()
        self.current_track = None

    # ------------------------------------------------------------------ effects
    def play(self, event: str, intensity: float = 1.0, rate: float = 1.0) -> bool:
        """Start a sound for a real game event. Returns False if it was suppressed."""
        spec = EVENTS.get(event)
        pool = self._pools.get(event)
        now = time.perf_counter()
        if spec is None or not pool:
            return False
        if self.sfx_gain() <= 0.0:
            return False
        if now - self._last.get(event, -1e9) < spec.cooldown:
            self.suppressed += 1
            return False
        while self._recent and now - self._recent[0] > WINDOW:
            self._recent.popleft()
        if len(self._recent) >= MAX_STARTS_PER_WINDOW:
            self.suppressed += 1
            return False
        idx = self._next[event]
        snd = pool[idx]
        self._next[event] = (idx + 1) % len(pool)
        snd.setVolume(self.sfx_gain() * spec.gain * _clamp(intensity))
        snd.setPlayRate(max(0.5, min(2.0, rate)))
        snd.play()
        self._last[event] = now
        self._recent.append(now)
        self.log.append((now, event))
        self.counts[event] = self.counts.get(event, 0) + 1
        return True

    # backwards-compatible alias used by older screens
    def ui(self) -> None:
        self.play("ui")

    def set_rolling(self, level: float) -> None:
        """Rolling loop follows the summed marble speed (0 = silent)."""
        self._roll_level = _clamp(level)
        self._apply_roll_volume()

    def _apply_roll_volume(self) -> None:
        if self._roll is None:
            return
        v = self.sfx_gain() * 0.55 * self._roll_level
        if v > 0.005:
            self._roll.setVolume(v)
            if not self._roll.is_playing():
                self._roll.play()
                self.counts["roll"] = self.counts.get("roll", 0) + 1
        elif self._roll.is_playing():
            self._roll.stop()

    def stop_all(self) -> None:
        self.set_rolling(0.0)
        self.stop_music()


def _clamp(v, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return lo
