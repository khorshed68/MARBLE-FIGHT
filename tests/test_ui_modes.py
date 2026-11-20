"""End-to-end UI tests: the real Panda3D app, offscreen, driving every game mode.

Audio uses the silent NullBackend so the AudioManager's real event logic runs
and can be counted. Match logs and replays are redirected to a temp folder.
"""

import math
import time

import pytest

FAST = {"minimax": {"max_depth": 2, "root_width": 10, "inner_width": 5},
        "mcts": {"iterations": 25, "rollout_depth": 1, "candidate_width": 16}}


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    try:
        from ui.app import configure_panda
        configure_panda((960, 540), offscreen=True)
        from ui.app import MarbleFightApp
        from ui.audio import NullBackend
        a = MarbleFightApp(audio_backend=NullBackend())
    except Exception as exc:  # pragma: no cover - no OpenGL available
        pytest.skip(f"Panda3D offscreen window unavailable: {exc}")
    # fixed test settings: independent of the player's own data/settings.json (e.g. muted audio)
    a.settings.update(show_analysis=True, show_trajectories=True, effects="full", graphics_quality="high",
                      muted=False, master_volume=0.8, music_volume=0.6, sfx_volume=0.9, rounds=1)
    a.save_settings = lambda: None          # tests never overwrite the player's settings file
    a.apply_audio_settings()
    yield a
    a.worker.shutdown()
    a.destroy()


@pytest.fixture(autouse=True)
def isolated_files(tmp_path, monkeypatch):
    import ui.game_screen as gs
    from analytics import match_logger
    from game import replay
    hist = str(tmp_path / "hist.jsonl")
    monkeypatch.setattr(gs, "log_match", lambda m, src: match_logger.log_match(m, src, path=hist))
    monkeypatch.setattr(gs, "save_replay", lambda rec: replay.save_replay(rec, directory=str(tmp_path)))


def pump(app, until, limit=90.0):
    t0 = time.time()
    while not until():
        app.taskMgr.step()
        assert time.time() - t0 < limit, "timed out waiting for the game"


def spec(kind):
    from game.match import PlayerSpec
    return PlayerSpec(kind, "easy", overrides=dict(FAST.get(kind, {})))


def test_ai_vs_ai_mode_audio_pause_and_responsiveness(app):
    app.goto("game", mode="ai_vs_ai", specs=(spec("minimax"), spec("mcts")), arena_id="neon", marbles=3, seed=7)
    gs = app.screen
    assert gs.phase == "waiting_start"
    before = dict(app.audio.counts)
    gs.on_start_pause()                                    # START MATCH
    gaps, last = [], time.perf_counter()
    gs.speed = 4.0
    t0 = time.time()
    prev_phase = None
    while gs.match.state.turn < 3 and not gs.match.is_over:
        app.taskMgr.step()
        now = time.perf_counter()
        if gs.phase == "thinking" and prev_phase == "thinking":   # frame-to-frame gap while the AI searches
            gaps.append(now - last)
        last, prev_phase = now, gs.phase
        assert time.time() - t0 < 90
    assert gaps and max(gaps) < 0.25                       # frames keep coming while the AI thinks
    gained = {k: app.audio.counts.get(k, 0) - before.get(k, 0) for k in app.audio.counts}
    assert gained.get("launch", 0) + gained.get("launch_power", 0) >= 2
    assert gained.get("turn", 0) >= 1
    # pause freezes the match
    pump(app, lambda: gs.phase in ("thinking", "human", "turn_start", "reveal"))
    gs.on_start_pause()
    t = gs.match.state.turn
    end = time.time() + 1.0
    while time.time() < end:
        app.taskMgr.step()
    assert gs.match.state.turn == t and gs.paused
    gs.on_start_pause()
    assert not gs.paused


def test_human_vs_ai_mode(app):
    app.goto("game", mode="human", specs=(spec("human"), spec("minimax")), arena_id="magnetic", marbles=3, seed=2)
    gs = app.screen
    pump(app, lambda: gs.phase == "human")
    launches = app.audio.counts.get("launch", 0) + app.audio.counts.get("launch_power", 0)
    gs.aim_power = 0.7
    gs.fire_human()
    assert gs.phase == "animating"
    gs.on_space()                                          # input is ignored while the shot resolves
    assert gs.match.state.turn == 1
    pump(app, lambda: gs.phase == "human" or gs.match.is_over)
    assert gs.match.state.turn >= 2 or gs.match.is_over
    assert gs.match.history[1].decision is not None        # the AI decided turn 2 itself
    assert app.audio.counts.get("launch", 0) + app.audio.counts.get("launch_power", 0) >= launches + 2


def test_human_vs_human_mode_alternates_and_blocks_input(app):
    from game.match import PlayerSpec
    app.goto("game", mode="pvp", specs=(PlayerSpec("human"), PlayerSpec("human")), arena_id="classic", marbles=2,
             seed=3)
    gs = app.screen
    gs.speed = 4.0
    for expected in (0, 1, 0):
        pump(app, lambda: gs.phase == "human")
        assert gs.match.state.to_move == expected
        assert f"PLAYER {expected + 1}'S TURN" in gs._status()[0]
        gs.ability = "power" if expected == 1 else "none"
        gs.fire_human()
        turn = gs.match.state.turn
        gs.on_space()
        gs.fire_human() if gs.phase == "human" else None   # not reached: phase is 'animating'
        assert gs.match.state.turn == turn
    assert app.audio.counts.get("ability_power", 0) >= 1
    assert all(h.decision is None for h in gs.match.history)


def test_finish_results_and_replay_mode(app):
    app.goto("game", mode="ai_vs_ai", specs=(spec("minimax"), spec("mcts")), arena_id="classic", marbles=2, seed=11)
    gs = app.screen
    gs.match.turn_limit = 6
    gs.on_restart()                                        # apply the shorter limit to a fresh state
    gs.speed = 4.0
    gs.on_start_pause()
    pump(app, lambda: gs.results is not None, 180)
    end_sounds = sum(app.audio.counts.get(k, 0) for k in ("victory", "defeat", "draw"))
    assert end_sounds >= 1
    rec = gs.replay_record
    assert rec is not None and len(rec["turns"]) == gs.match.state.turn
    final = gs.match.state
    gs.watch_replay()
    rs = app.screen
    assert rs.mode == "replay"
    rs.speed = 4.0
    pump(app, lambda: rs.results is not None or rs.replay_error is not None, 180)
    assert rs.replay_error is None
    assert rs.match.state.fingerprint() == final.fingerprint()
    assert rs.match.state.winner == final.winner
    app.goto("menu")


def _rim_ndc(app, radius):
    from panda3d.core import Point2, Point3
    pts = []
    for k in range(72):
        a = 2 * math.pi * k / 72
        q = app.camera.getRelativePoint(app.render, Point3(radius * math.cos(a), radius * math.sin(a), 0))
        o = Point2()
        assert app.camLens.project(q, o), "part of the arena is outside the camera view"
        pts.append((o.x, o.y))
    return pts


@pytest.mark.parametrize("mode", ["ai_vs_ai", "human", "pvp"])
def test_whole_arena_is_visible_and_clear_of_the_hud(app, mode):
    from game.match import PlayerSpec
    specs = {"ai_vs_ai": (spec("minimax"), spec("mcts")), "human": (PlayerSpec("human"), spec("mcts")),
             "pvp": (PlayerSpec("human"), PlayerSpec("human"))}[mode]
    app.goto("game", mode=mode, specs=specs, arena_id="magnetic", marbles=3, seed=1)
    gs = app.screen
    for _ in range(5):
        app.taskMgr.step()
    R = gs.match.arena.radius

    def check():
        app.cam_rig.set_view(snap=True)
        human = mode in ("human", "pvp")
        rects = gs.hud.obstacles(power=human, analysis=gs.has_ai, buttons=5 if human else 6)
        pts = _rim_ndc(app, R)
        for x, y in pts:
            assert -1 < x < 1 and -1 < y < 1, (x, y)
            for x0, x1, y0, y1 in rects:
                assert not (x0 < x < x1 and y0 < y < y1), f"rim point {(x, y)} under a HUD panel"
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    area = check()
    assert area > 0.6                     # the arena fills a large part of the screen
    # a window resize re-frames the camera and rebuilds the HUD without errors
    old_hud = gs.hud
    app.cam_rig._refit()
    assert gs.hud is not old_hud
    check()
    app.goto("menu")
    assert gs._on_aspect not in app.cam_rig.aspect_listeners


def test_clicking_a_marble_on_screen_selects_it(app):
    """Screen -> floor picking stays exact with the fitted camera (film offset)."""
    from game.match import PlayerSpec
    from panda3d.core import Point2, Point3
    app.goto("game", mode="human", specs=(PlayerSpec("human"), spec("minimax")), arena_id="neon", marbles=3, seed=2)
    gs = app.screen
    pump(app, lambda: gs.phase == "human")
    app.cam_rig.set_view(snap=True)
    for m in gs.match.state.alive(0):
        q = app.camera.getRelativePoint(app.render, Point3(m.x, m.y, 0))
        ndc = Point2()
        assert app.camLens.project(q, ndc)
        fx, fy = app.cam_rig.screen_to_floor(ndc)
        assert math.hypot(fx - m.x, fy - m.y) < 1e-3
        assert gs.view.pick(fx, fy, owner=0) == m.mid
    app.goto("menu")


def test_p_key_closes_the_pause_menu_instead_of_resuming_behind_it(app):
    app.goto("game", mode="ai_vs_ai", specs=(spec("minimax"), spec("mcts")), arena_id="classic", marbles=2, seed=2)
    gs = app.screen
    gs.on_start_pause()                                    # START MATCH
    gs.toggle_pause_menu()                                 # Esc
    assert gs.paused and gs.pause_menu is not None
    gs.on_step()                                           # stepping is ignored while the menu is open
    assert gs.paused and gs.pause_menu is not None
    gs.on_start_pause()                                    # P
    assert not gs.paused and gs.pause_menu is None         # resumed AND the menu is gone
    gs.on_start_pause()                                    # P again: ordinary pause, no menu
    assert gs.paused and gs.pause_menu is None


def test_c_key_restores_zoom_tilt_and_heading(app):
    from game.match import PlayerSpec
    app.goto("game", mode="pvp", specs=(PlayerSpec("human"), PlayerSpec("human")), arena_id="neon", marbles=2, seed=1)
    rig = app.cam_rig
    home = (rig.td, rig.tp, rig.th)
    rig.zoom(1.3)
    rig.orbit(40.0, 15.0)
    assert (rig.td, rig.tp, rig.th) != home
    rig.reset_view()
    assert (rig.td, rig.tp, rig.th) == pytest.approx(home)


def test_hud_rebuild_keeps_dots_for_eliminated_marbles(app):
    from game.match import PlayerSpec
    app.goto("game", mode="pvp", specs=(PlayerSpec("human"), PlayerSpec("human")), arena_id="classic", marbles=3,
             seed=1)
    gs = app.screen
    s = gs.match.state
    first = next(m for m in s.marbles if m.owner == 0)
    gs.match.state = s.with_changes(marbles=tuple(m if m.mid != first.mid else m.__class__(m.mid, m.owner, m.x, m.y, False)
                                                  for m in s.marbles))
    gs._on_aspect()                                        # what a window resize does
    dots = gs.hud.pp[0]["dots"]
    assert len(dots) == 3                                  # 2 alive + 1 eliminated (grey)
    assert gs.hud.pp[0]["count"].getText() == "2"


def test_two_round_series_swaps_first_shooter_and_shows_series_result(app):
    from game.state import DRAW
    app.goto("game", mode="ai_vs_ai", specs=(spec("minimax"), spec("mcts")), arena_id="classic", marbles=2, seed=21,
             rounds=2)
    gs = app.screen
    assert gs.series is not None and gs.series.total_rounds == 2
    gs.match.turn_limit = 6
    gs.on_restart()                                        # apply the shorter limit to a fresh state
    assert gs.match.state.to_move == 0                     # round 1: Player 1 shoots first
    gs.speed = 4.0
    gs.on_start_pause()
    pump(app, lambda: len(gs.series.rounds) == 1, 180)
    assert gs.results is None                              # no results screen between rounds
    pump(app, lambda: gs.match.state.turn == 0 and gs.phase != "over", 30)
    assert gs.match.first_player == 1 and gs.match.state.to_move == 1   # round 2: Player 2 shoots first
    assert "ROUND 2/2" in gs._status()[0]
    pump(app, lambda: gs.results is not None, 180)
    s = gs.series
    assert s.is_over and [r.first_player for r in s.rounds] == [0, 1]
    assert s.winner() in (0, 1, DRAW)
    gs.on_restart()                                        # REMATCH restarts the whole series
    assert s.rounds == [] and gs.match.first_player == 0 and gs.phase == "waiting_start"
    app.goto("menu")


def test_setup_screens_offer_rounds_and_remember_the_choice(app, monkeypatch):
    monkeypatch.setattr(app, "save_settings", lambda: None)   # never touch the real settings file
    for name in ("setup_ai", "setup_human", "setup_pvp"):
        app.goto(name)
        scr = app.screen
        scr.rounds.set(2)
        assert scr.chosen_rounds() == 2 and app.settings["rounds"] == 2
        scr.rounds.set(1)
        assert scr.chosen_rounds() == 1
    app.goto("menu")


def test_algorithm_lab_search_tree_tab_shows_the_real_search(app):
    from ai.search_trace import demo_positions
    app.goto("lab")
    lab = app.screen
    lab.show("tree")
    for scene, _, _ in demo_positions():
        for depth in (2, 1):
            lab.tree_opts["depth"] = depth
            lab._tree_set("scene", scene)
            tr = lab._trace()
            assert tr.value == pytest.approx(tr.minimax_value)      # pruning never changes the answer
            assert tr.best_child is not None and tr.depth == depth
            app.taskMgr.step()
    lab._tree_set("fuzzy", False)
    assert lab._trace().root.children
    app.goto("menu")


def test_algorithm_lab_mcts_tree_steps_through_real_iterations(app):
    from ui.algorithm_lab import MCTS_ITERS
    app.goto("lab")
    lab = app.screen
    lab.show("tree")
    lab._tree_set("algo", "mcts")
    st = lab._stepper()
    assert st.iterations == 0 and st.root.visits == 0
    lab._mcts_step(1)
    assert st.iterations == 1 and len(st.root.children) == 1
    lab._mcts_step(10)
    assert st.iterations == 11 and st.root.visits == 11
    lab._mcts_step(10 ** 6)                                   # RUN ALL stops at the budget
    assert st.done and st.iterations == MCTS_ITERS
    assert sum(c.visits for c in st.root.children) == MCTS_ITERS
    lab._mcts_reset()
    assert lab._stepper() is not st and lab._stepper().iterations == 0
    lab._tree_set("algo", "minimax")
    app.taskMgr.step()
    app.goto("menu")


def test_algorithm_lab_tree_for_a_recorded_match(app, tmp_path):
    from ai.search_trace import recent_matches
    from game.match import Match
    from game.replay import record_from_match, save_replay
    m = Match("classic", spec("minimax"), spec("mcts"), marbles=2, seed=4, turn_limit=6)
    m.run_to_end()
    save_replay(record_from_match(m, "arena"), "lab", str(tmp_path))
    app.goto("lab")
    lab = app.screen
    lab._recent_matches = recent_matches(5, str(tmp_path))  # this test's match, not the player's files
    lab.show("tree")
    lab._tree_set("scene", "match0")
    assert lab._match() is not None and lab.tree_opts["turn"] == 0
    for turn in range(len(m.history)):
        lab._tree_set("turn", turn)
        tr = lab._trace()
        assert tr.root.state.fingerprint() == lab._match().states[turn].fingerprint()
        assert tr.root_player == m.history[turn].player
    lab._tree_set("algo", "mcts")
    lab._mcts_step(5)
    assert lab._stepper().iterations == 5
    lab._tree_set("scene", "trap")                         # back to a demo resets the turn
    assert lab.tree_opts["turn"] == 0 and lab._match() is None
    app.goto("menu")
