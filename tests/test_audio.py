"""Audio manager logic (volumes, mute, rate limiting, missing assets) with a silent backend."""

import os
import time

import pytest

from ui.audio import AUDIO_DIR, EVENTS, MAX_STARTS_PER_WINDOW, AudioManager, NullBackend

SETTINGS = {"master_volume": 0.5, "music_volume": 0.6, "sfx_volume": 0.8, "muted": False}


def mgr(missing=None, **kw):
    return AudioManager(NullBackend(missing), {**SETTINGS, **kw})


def test_assets_are_present_in_the_project():
    for spec in EVENTS.values():
        assert os.path.exists(os.path.join(AUDIO_DIR, "sfx", f"{spec.file}.wav")), spec.file
    for track in ("menu", "game"):
        assert os.path.exists(os.path.join(AUDIO_DIR, "music", f"{track}.wav"))
    assert os.path.exists(os.path.join(AUDIO_DIR, "sfx", "roll.wav"))


def test_volume_math():
    a = mgr()
    assert a.sfx_gain() == pytest.approx(0.4)
    assert a.music_gain() == pytest.approx(0.3)
    assert a.play("hit", intensity=0.5)
    snd = a._pools["hit"][0]
    assert snd.volume == pytest.approx(0.4 * EVENTS["hit"].gain * 0.5)
    a.play_music("menu")
    assert a._music["menu"].volume == pytest.approx(0.3) and a._music["menu"].playing


def test_mute_and_live_settings():
    a = mgr()
    a.play_music("game")
    a.apply_settings({**SETTINGS, "muted": True})
    assert a._music["game"].volume == 0.0
    assert not a.play("launch")
    a.apply_settings({**SETTINGS, "master_volume": 1.0, "music_volume": 1.0})
    assert a._music["game"].volume == pytest.approx(1.0)
    a.apply_settings({"master_volume": "bogus", "music_volume": 7, "sfx_volume": -3})
    assert (a.master, a.music_vol, a.sfx_vol) == (0.0, 1.0, 0.0)


def test_cooldown_prevents_repeated_spam():
    a = mgr()
    assert a.play("turn")
    assert not a.play("turn")                    # within the 0.3 s cooldown
    assert a.suppressed == 1


def test_voice_pool_round_robin_and_global_cap():
    a = mgr()
    started = 0
    for _ in range(30):                           # a pile-up of collisions in one instant
        for ev in ("hit", "hit_soft", "bumper", "eliminate", "launch"):
            started += a.play(ev)
        time.sleep(0.04)
    starts = sorted(t for t, _ in a.log)
    for i in range(len(starts)):
        window = [t for t in starts if starts[i] <= t < starts[i] + 0.1]
        assert len(window) <= MAX_STARTS_PER_WINDOW
    assert a.suppressed > 0 and started == len(a.log)
    hits = a._pools["hit"]
    assert len(hits) == EVENTS["hit"].voices and all(s.plays > 0 for s in hits)


def test_missing_assets_are_handled_gracefully():
    a = mgr(missing={"hit", "menu", "roll"})
    assert any("hit.wav" in w for w in a.warnings)
    assert not a.play("hit")                       # silently skipped
    a.play_music("menu")                           # no exception
    a.set_rolling(1.0)
    assert a.play("bumper")


def test_unknown_event_is_ignored():
    assert not mgr().play("does-not-exist")


def test_rolling_loop_follows_speed():
    a = mgr()
    a.set_rolling(0.8)
    assert a._roll.playing and a._roll.loop
    assert a._roll.volume == pytest.approx(0.4 * 0.55 * 0.8)
    a.set_rolling(0.0)
    assert not a._roll.playing
