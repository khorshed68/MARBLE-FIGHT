"""Marble Fight — Intelligent Marble Battle Arena.

Run ``python main.py`` to start the 3-D game.

Extra options::

    python main.py --windowed 1280x720      choose the window size
    python main.py --fullscreen
    python main.py --ai-vs-ai               jump straight into a Minimax-vs-MCTS match
    python main.py --screenshots DIR        render every screen offscreen to PNG files (self-check)
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _parse(argv=None):
    ap = argparse.ArgumentParser(description="Marble Fight - Intelligent Marble Battle Arena")
    ap.add_argument("--windowed", default="1600x900", help="window size WxH (default 1600x900)")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--ai-vs-ai", action="store_true", help="start a Minimax vs MCTS match immediately")
    ap.add_argument("--arena", default=None, help="arena id for --ai-vs-ai (classic, neon, magnetic)")
    ap.add_argument("--screenshots", metavar="DIR", default=None,
                    help="render all screens offscreen into DIR and exit (verification aid)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(argv)
    try:
        w, h = (int(v) for v in args.windowed.lower().split("x"))
    except ValueError:
        w, h = 1600, 900
    from ui.app import configure_panda
    configure_panda((w, h), offscreen=bool(args.screenshots), fullscreen=args.fullscreen)
    from ui.app import MarbleFightApp
    if args.screenshots:
        from tools.screenshots import capture_all
        from ui.audio import NullBackend
        app = MarbleFightApp(audio_backend=NullBackend())   # offscreen: count audio events silently
        capture_all(app, args.screenshots)
        app.worker.shutdown()
        return 0
    app = MarbleFightApp()
    if args.ai_vs_ai:
        from game.match import PlayerSpec
        app.goto("game", mode="ai_vs_ai", specs=(PlayerSpec("minimax"), PlayerSpec("mcts")),
                 arena_id=args.arena or app.settings["arena"], marbles=app.settings["marbles_per_player"], seed=42,
                 autostart=True)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
