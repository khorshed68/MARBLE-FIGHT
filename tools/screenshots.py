"""Render every screen offscreen and save PNGs (used to verify the UI without a display).

    python main.py --screenshots shots
"""

from __future__ import annotations

import math
import os
import time

from game.match import PlayerSpec


def _pump(app, seconds: float = 0.0, frames: int = 3) -> None:
    end = time.time() + seconds
    n = 0
    while n < frames or time.time() < end:
        app.taskMgr.step()
        n += 1
        if seconds:
            time.sleep(0.01)


def _shot(app, out_dir: str, name: str) -> str:
    app.graphicsEngine.renderFrame()
    from panda3d.core import Filename
    path = os.path.abspath(os.path.join(out_dir, f"{name}.png"))
    ok = app.win.saveScreenshot(Filename.fromOsSpecific(path))
    print("saved" if ok else "FAILED to save", path, flush=True)
    return path


def capture_all(app, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name in ("menu", "setup_ai", "setup_human", "setup_pvp", "arenas", "settings", "howto", "tournament",
                 "replays", "lab"):
        app.goto(name)
        _pump(app, 0.4)
        _shot(app, out_dir, f"screen_{name}")

    # AI vs AI: real worker, real decisions
    app.goto("game", mode="ai_vs_ai", specs=(PlayerSpec("minimax", "easy"), PlayerSpec("mcts", "easy")),
             arena_id="neon", marbles=3, seed=42)
    _pump(app, 0.3)
    _shot(app, out_dir, "game_ai_waiting")
    gs = app.screen
    gs.on_start_pause()
    t0 = time.time()
    while gs.phase != "reveal" and time.time() - t0 < 60:
        _pump(app, 0.0, 1)
    _pump(app, 0.25)
    _shot(app, out_dir, "game_ai_reveal")
    t0 = time.time()
    while gs.phase != "animating" and time.time() - t0 < 30:
        _pump(app, 0.0, 1)
    _pump(app, 0.35)
    _shot(app, out_dir, "game_ai_rolling")
    # play the rest quickly and capture the results overlay
    gs.speed = 4.0
    t0 = time.time()
    while not gs.match.is_over and time.time() - t0 < 240:
        _pump(app, 0.0, 1)
    t0 = time.time()
    while gs.results is None and time.time() - t0 < 30:
        _pump(app, 0.0, 1)
    _pump(app, 0.3)
    _shot(app, out_dir, "game_ai_results")

    # Human vs AI with an aimed shot preview
    app.goto("game", mode="human", specs=(PlayerSpec("human"), PlayerSpec("minimax", "easy")),
             arena_id="magnetic", marbles=3, seed=1)
    _pump(app, 0.5)
    gs = app.screen
    gs.aim_angle += math.radians(10)
    gs.aim_power = 0.85
    gs.ability = "spin_left"
    gs.preview_dirty = True
    _pump(app, 0.3)
    _shot(app, out_dir, "game_human_aim")
    # fire the human shot and let the AI answer through the worker
    gs.fire_human()
    t0 = time.time()
    while not (gs.phase == "human" and gs.match.state.turn >= 2) and not gs.match.is_over and time.time() - t0 < 60:
        _pump(app, 0.0, 1)
    _pump(app, 0.4)
    print("human-vs-ai turns played:", gs.match.state.turn, "phase:", gs.phase, flush=True)
    _shot(app, out_dir, "game_human_after_ai")

    # Human vs Human: one shot each
    app.goto("game", mode="pvp", specs=(PlayerSpec("human"), PlayerSpec("human")), arena_id="neon", marbles=3, seed=5)
    gs = app.screen
    for _ in range(2):
        t0 = time.time()
        while gs.phase != "human" and time.time() - t0 < 30:
            _pump(app, 0.0, 1)
        gs.fire_human()
    t0 = time.time()
    while gs.phase != "human" and time.time() - t0 < 30:
        _pump(app, 0.0, 1)
    _pump(app, 0.3)
    _shot(app, out_dir, "game_pvp")

    # replay of the most recent saved match
    from game.replay import list_replays, load_replay
    files = list_replays(limit=1)
    if files:
        app.goto("game", mode="replay", replay=load_replay(files[0]))
        gs = app.screen
        t0 = time.time()
        while gs.phase != "animating" and time.time() - t0 < 30:
            _pump(app, 0.0, 1)
        _pump(app, 0.3)
        _shot(app, out_dir, "game_replay")
    print("audio events:", dict(sorted(app.audio.counts.items())), "suppressed:", app.audio.suppressed, flush=True)

    # a small real tournament through the UI
    app.goto("tournament")
    ts = app.screen
    ts.c["matches"].set(4)
    ts.run()
    t0 = time.time()
    while (ts.thread is not None or ts.result is None) and ts.error is None and time.time() - t0 < 600:
        _pump(app, 0.0, 1)
        time.sleep(0.02)
    _pump(app, 0.3)
    _shot(app, out_dir, "tournament_done")

    app.goto("lab")
    _pump(app, 0.3)
    _shot(app, out_dir, "lab_overview")
    for tab in ("search", "arenas", "tournaments", "last", "concepts"):
        app.screen.show(tab)
        _pump(app, 0.2)
        _shot(app, out_dir, f"lab_{tab}")
    app.goto("menu")
    _pump(app, 0.2)
