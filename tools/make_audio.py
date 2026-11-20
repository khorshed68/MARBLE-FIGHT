"""Generate Marble Fight's original audio assets (music loops + sound effects).

    python -m tools.make_audio            # writes assets/audio/music/*.wav and assets/audio/sfx/*.wav
    python -m tools.make_audio --force    # regenerate even if the files exist

Every sound is synthesised here from sine/saw/noise oscillators and envelopes,
so the assets are original work of this project (no third-party samples, no
licence restrictions). Output: 22.05 kHz, 16-bit mono WAV.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import struct
import sys
import wave

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT

RATE = 22050
AUDIO_DIR = os.path.join(PROJECT_ROOT, "assets", "audio")
TAU = 2 * math.pi


def write_wav(path: str, samples: list[float], gain: float = 0.9) -> None:
    peak = max(1e-9, max(abs(s) for s in samples))
    k = gain / peak
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s * k)) * 32000)) for s in samples))


def note(n: int) -> float:
    """MIDI note number -> frequency."""
    return 440.0 * 2 ** ((n - 69) / 12)


def adsr(t: float, dur: float, a: float = 0.01, d: float = 0.1, s: float = 0.6, r: float = 0.2) -> float:
    if t < 0 or t > dur + r:
        return 0.0
    if t < a:
        return t / a
    if t < a + d:
        return 1 - (1 - s) * (t - a) / d
    if t <= dur:
        return s
    return s * (1 - (t - dur) / r)


def _lowpass(xs: list[float], alpha: float) -> list[float]:
    out, y = [], 0.0
    for x in xs:
        y += alpha * (x - y)
        out.append(y)
    return out


# ----------------------------------------------------------------------------- music
def music_menu() -> list[float]:
    """Calm ambient loop: Am - F - C - G pads with a slow arpeggio (16 s, seamless)."""
    chords = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]
    bar = 4.0
    n = int(RATE * bar * len(chords))
    out = [0.0] * n
    rng = random.Random(3)
    for ci, ch in enumerate(chords):
        t0 = ci * bar
        for k, m in enumerate(ch):
            f = note(m)
            det = 1.003
            start = int(t0 * RATE)
            for i in range(int(bar * RATE) + int(0.6 * RATE)):
                j = (start + i) % n          # wrap the release into the loop start -> seamless
                t = i / RATE
                env = adsr(t, bar - 0.2, 0.9, 0.5, 0.75, 0.8)
                out[j] += env * 0.11 * (math.sin(TAU * f * t) + 0.6 * math.sin(TAU * f * det * t + k)
                                        + 0.25 * math.sin(TAU * f * 2 * t))
        # arpeggio of glassy plucks (8th notes)
        for step in range(8):
            m = ch[step % 3] + (12 if step >= 4 else 0) + 12
            f = note(m)
            st = t0 + step * bar / 8
            start = int(st * RATE)
            for i in range(int(0.45 * RATE)):
                j = (start + i) % n
                t = i / RATE
                out[j] += 0.07 * math.exp(-t * 7) * (math.sin(TAU * f * t) + 0.3 * math.sin(TAU * f * 3.01 * t))
    # gentle air noise
    noise = _lowpass([rng.uniform(-1, 1) for _ in range(n)], 0.02)
    return [o + 0.05 * z for o, z in zip(out, noise)]


def music_game() -> list[float]:
    """Driving arena loop at 112 BPM: kick, hats, bass and pluck lead (8 bars, seamless)."""
    bpm = 112
    beat = 60 / bpm
    bars = 8
    n = int(RATE * beat * 4 * bars)
    out = [0.0] * n
    rng = random.Random(7)
    prog = [45, 45, 41, 41, 48, 48, 43, 43]          # A A F F C C G G (bass roots)

    def add(start_s: float, dur_s: float, fn) -> None:
        start = int(start_s * RATE)
        for i in range(int(dur_s * RATE)):
            j = (start + i) % n
            out[j] += fn(i / RATE)

    for b in range(bars):
        root = prog[b]
        for q in range(4):
            t0 = (b * 4 + q) * beat
            # kick
            add(t0, 0.25, lambda t: 0.55 * math.sin(TAU * (50 + 90 * math.exp(-t * 30)) * t) * math.exp(-t * 14))
            # hats on off-beats
            add(t0 + beat / 2, 0.06, lambda t: 0.12 * rng.uniform(-1, 1) * math.exp(-t * 60))
            # bass: 8th notes
            for e in range(2):
                f = note(root + (12 if e else 0))
                add(t0 + e * beat / 2, beat / 2 * 0.95,
                    lambda t, f=f: 0.22 * adsr(t, beat / 2 * 0.8, 0.005, 0.05, 0.7, 0.03)
                    * (2 * ((f * t) % 1) - 1))
        # clap on beats 2 and 4
        for q in (1, 3):
            add((b * 4 + q) * beat, 0.15, lambda t: 0.18 * rng.uniform(-1, 1) * math.exp(-t * 25))
        # pluck lead motif (minor pentatonic over the root)
        motif = [0, 3, 7, 10, 12, 10, 7, 3] if b % 2 == 0 else [12, 10, 7, 5, 7, 3, 0, 3]
        for k, iv in enumerate(motif):
            f = note(root + 24 + iv)
            add((b * 4) * beat + k * beat / 2, 0.3,
                lambda t, f=f: 0.09 * math.exp(-t * 9) * (math.sin(TAU * f * t) + 0.4 * math.sin(TAU * 2 * f * t)))
    return out


# ----------------------------------------------------------------------------- sound effects
def _env_noise(dur: float, decay: float, seed: int, lp: float = 0.3) -> list[float]:
    rng = random.Random(seed)
    xs = [rng.uniform(-1, 1) for _ in range(int(RATE * dur))]
    xs = _lowpass(xs, lp)
    return [x * math.exp(-i / RATE * decay) for i, x in enumerate(xs)]


def sfx_launch(power: bool = False) -> list[float]:
    dur = 0.35 if power else 0.25
    whoosh = _env_noise(dur, 9 if power else 14, 11, 0.12)
    out = []
    for i, w in enumerate(whoosh):
        t = i / RATE
        f = (70 if power else 110) + (160 if power else 220) * math.exp(-t * 25)
        thump = math.sin(TAU * f * t) * math.exp(-t * (18 if power else 26))
        out.append(0.8 * thump + 0.6 * w)
    return out


def sfx_glass_hit(freq: float = 2600.0, seed: int = 1) -> list[float]:
    rng = random.Random(seed)
    n = int(RATE * 0.12)
    partials = [(freq, 1.0), (freq * 2.76, 0.5), (freq * 5.4, 0.25)]
    return [sum(a * math.sin(TAU * f * i / RATE) for f, a in partials) * math.exp(-i / RATE * 45)
            + 0.3 * rng.uniform(-1, 1) * math.exp(-i / RATE * 220) for i in range(n)]


def sfx_bumper() -> list[float]:
    n = int(RATE * 0.35)
    return [(math.sin(TAU * 420 * i / RATE) + 0.6 * math.sin(TAU * 1130 * i / RATE)
             + 0.3 * math.sin(TAU * 1980 * i / RATE)) * math.exp(-i / RATE * 14) for i in range(n)]


def sfx_roll() -> list[float]:
    """1.5 s loopable low rumble (rolling on a surface)."""
    n = int(RATE * 1.5)
    rng = random.Random(5)
    xs = _lowpass(_lowpass([rng.uniform(-1, 1) for _ in range(n)], 0.08), 0.15)
    # crossfade the end into the start so the loop is seamless
    fade = int(RATE * 0.2)
    for i in range(fade):
        a = i / fade
        xs[i] = xs[i] * a + xs[n - fade + i] * (1 - a)
    xs = xs[: n - fade]
    return [x * (0.8 + 0.2 * math.sin(TAU * 7 * i / RATE)) for i, x in enumerate(xs)]


def sfx_eliminate() -> list[float]:
    n = int(RATE * 0.9)
    noise = _env_noise(0.9, 3, 9, 0.05)
    return [0.7 * math.sin(TAU * (500 * math.exp(-i / RATE * 3) + 60) * i / RATE) * math.exp(-i / RATE * 3.5)
            + 0.5 * noise[i] for i in range(n)]


def _melody(notes: list[tuple[int, float]], wave_fn=math.sin, decay: float = 4.0, gap: float = 0.0) -> list[float]:
    total = sum(d for _, d in notes) + 0.6
    out = [0.0] * int(RATE * total)
    t0 = 0.0
    for m, d in notes:
        f = note(m)
        start = int(t0 * RATE)
        for i in range(int((d + 0.5) * RATE)):
            if start + i >= len(out):
                break
            t = i / RATE
            out[start + i] += (wave_fn(TAU * f * t) + 0.3 * wave_fn(TAU * 2 * f * t)) * math.exp(-t * decay)
        t0 += d + gap
    return out


def sfx_ability(kind: str) -> list[float]:
    n = int(RATE * 0.45)
    if kind == "precision":
        return [math.sin(TAU * 1760 * i / RATE) * math.exp(-i / RATE * 10)
                + 0.5 * math.sin(TAU * 2640 * i / RATE) * math.exp(-i / RATE * 14) for i in range(n)]
    if kind == "power":
        return [math.sin(TAU * (120 + 600 * (i / n) ** 2) * i / RATE) * (i / n) ** 0.5 * math.exp(-max(0, i / RATE - 0.35) * 30)
                for i in range(n)]
    # spin: warbling rise
    return [math.sin(TAU * (500 + 700 * i / n) * i / RATE + 3 * math.sin(TAU * 18 * i / RATE)) * math.sin(math.pi * i / n)
            for i in range(n)]


SFX = {
    "launch": lambda: sfx_launch(False),
    "launch_power": lambda: sfx_launch(True),
    "hit": lambda: sfx_glass_hit(2600, 1),
    "hit_soft": lambda: sfx_glass_hit(1900, 2),
    "bumper": sfx_bumper,
    "roll": sfx_roll,
    "eliminate": sfx_eliminate,
    "ability_precision": lambda: sfx_ability("precision"),
    "ability_power": lambda: sfx_ability("power"),
    "ability_spin": lambda: sfx_ability("spin"),
    "turn": lambda: _melody([(76, 0.08), (83, 0.12)], decay=9),
    "ui": lambda: _melody([(84, 0.05)], decay=30),
    "victory": lambda: _melody([(72, 0.14), (76, 0.14), (79, 0.14), (84, 0.5)], decay=3),
    "defeat": lambda: _melody([(67, 0.22), (63, 0.22), (60, 0.6)], decay=2.5),
    "draw": lambda: _melody([(69, 0.25), (69, 0.4)], decay=3, gap=0.05),
    "knockout_cheer": lambda: _melody([(79, 0.07), (86, 0.18)], decay=7),
}

MUSIC = {"menu": music_menu, "game": music_game}


def generate(force: bool = False, verbose: bool = True) -> list[str]:
    written = []
    for name, fn in MUSIC.items():
        p = os.path.join(AUDIO_DIR, "music", f"{name}.wav")
        if force or not os.path.exists(p):
            write_wav(p, fn(), 0.7)
            written.append(p)
    for name, fn in SFX.items():
        p = os.path.join(AUDIO_DIR, "sfx", f"{name}.wav")
        if force or not os.path.exists(p):
            write_wav(p, fn(), 0.9)
            written.append(p)
    if verbose:
        for p in written:
            print("wrote", os.path.relpath(p, PROJECT_ROOT))
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Marble Fight's original audio assets")
    ap.add_argument("--force", action="store_true")
    generate(ap.parse_args().force)
