"""Match replays: record the legal actions of a match and re-simulate them.

A replay file stores the match configuration (arena, marbles, first player,
seed, turn limit, both seats), and for every turn the executed action, the
decision statistics shown at the time, the eliminations and a fingerprint of
the authoritative state after the shot.

Replaying never trusts the stored positions: it rebuilds the initial state
from the configuration, validates each stored action with the normal rules and
re-runs ``simulate_turn``. After every turn the new state's fingerprint must
equal the recorded one; otherwise replay stops with :class:`ReplayError` at
that turn. Missing or malformed actions are reported, never invented.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from config import DATA_DIR, DEFAULT_CONFIG, GameConfig
from game.actions import Action, IllegalActionError, validate_action
from game.arena import get_arena
from game.rules import initial_state
from game.simulation import TurnResult, simulate_turn
from game.state import GameState

REPLAY_DIR = os.path.join(DATA_DIR, "replays")
REPLAY_VERSION = 1


class ReplayError(Exception):
    """The replay cannot continue (missing data, illegal action or divergence)."""


def config_fingerprint(cfg: GameConfig, arena_id: str) -> str:
    """Hash of every rule/physics constant and the arena layout used by a match."""
    arena = get_arena(arena_id)
    blob = json.dumps({"physics": asdict(cfg.physics), "rules": asdict(cfg.rules),
                       "arena": {"radius": arena.radius, "friction": arena.base_friction,
                                 "zones": [asdict(z) for z in arena.zones],
                                 "bumpers": [asdict(b) for b in arena.bumpers],
                                 "rotating": [asdict(b) for b in arena.rotating_bumpers],
                                 "magnets": [asdict(m) for m in arena.magnets]}}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def record_from_match(match, source: str = "arena") -> dict[str, Any]:
    """Build a replay record from a (usually finished) Match."""
    st = match.state
    return {
        "version": REPLAY_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "arena": match.arena.arena_id,
        "marbles": match.cfg.rules.marbles_per_player,
        "first_player": match.first_player,
        "seed": match.seed,
        "turn_limit": st.turn_limit,
        "specs": [s.to_dict() for s in match.specs],
        "config_fingerprint": config_fingerprint(match.cfg, match.arena.arena_id),
        "turns": [{"turn": h.turn, "player": h.player, "action": h.action, "decision": h.decision,
                   "eliminated": list(h.eliminated), "fingerprint": h.fingerprint} for h in match.history],
        "result": {"winner": st.winner, "end_reason": st.end_reason, "score": [st.count(0), st.count(1)],
                   "turns": st.turn},
    }


def save_replay(record: dict[str, Any], name: str | None = None, directory: str = REPLAY_DIR) -> str:
    os.makedirs(directory, exist_ok=True)
    if name is None:
        name = time.strftime("R%Y%m%d-%H%M%S") + f"-{record.get('seed', 0)}"
    path = os.path.join(directory, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh)
    return path


def load_replay(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ReplayError(f"Cannot read replay file: {exc}") from exc
    for key in ("arena", "marbles", "first_player", "seed", "turn_limit", "specs", "turns"):
        if key not in rec:
            raise ReplayError(f"Replay file is missing '{key}'")
    if rec.get("version") != REPLAY_VERSION:
        raise ReplayError(f"Unsupported replay version {rec.get('version')}")
    return rec


def list_replays(directory: str = REPLAY_DIR, limit: int = 12) -> list[str]:
    files = glob.glob(os.path.join(directory, "*.json"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:limit]


class ReplayPlayer:
    """Steps through a replay with the authoritative rules and physics."""

    def __init__(self, record: dict[str, Any], cfg: GameConfig = DEFAULT_CONFIG) -> None:
        from ai.factory import with_marbles
        self.record = record
        self.cfg = with_marbles(cfg, int(record["marbles"]))
        try:
            self.arena = get_arena(record["arena"])
        except KeyError as exc:
            raise ReplayError(str(exc)) from exc
        self.config_matches = record.get("config_fingerprint") == config_fingerprint(self.cfg, self.arena.arena_id)
        self.reset()

    def reset(self) -> None:
        r = self.record
        self.state: GameState = initial_state(self.arena, self.cfg, first_player=int(r["first_player"]),
                                              turn_limit=int(r["turn_limit"]))
        self.index = 0

    @property
    def total(self) -> int:
        return len(self.record["turns"])

    @property
    def done(self) -> bool:
        return self.index >= self.total

    def current_turn(self) -> dict[str, Any]:
        if self.done:
            raise ReplayError("No more recorded turns")
        return self.record["turns"][self.index]

    def step(self, record_frames: bool = False) -> TurnResult:
        t = self.current_turn()
        if "action" not in t or not isinstance(t["action"], dict):
            raise ReplayError(f"Turn {self.index + 1}: recorded action is missing")
        try:
            action = Action.from_dict(t["action"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayError(f"Turn {self.index + 1}: malformed action ({exc})") from exc
        if t.get("player") is not None and t["player"] != self.state.to_move:
            raise ReplayError(f"Turn {self.index + 1}: recorded player {t['player'] + 1} but rules say "
                              f"player {self.state.to_move + 1} moves")
        try:
            validate_action(self.state, action, self.cfg.rules)
        except IllegalActionError as exc:
            raise ReplayError(f"Turn {self.index + 1}: recorded action is illegal here ({exc})") from exc
        res = simulate_turn(self.state, action, self.arena, self.cfg, record=record_frames)
        fp = t.get("fingerprint")
        if fp and res.state.fingerprint() != fp:
            hint = "" if self.config_matches else " (the game configuration changed since recording)"
            raise ReplayError(f"Turn {self.index + 1}: replay diverged from the recorded match{hint}")
        self.state = res.state
        self.index += 1
        return res


@dataclass
class Verification:
    ok: bool
    turns_checked: int
    message: str
    final_state: GameState | None = None


def verify_replay(record: dict[str, Any], cfg: GameConfig = DEFAULT_CONFIG) -> Verification:
    """Re-simulate every turn and compare with the recorded states and result."""
    try:
        p = ReplayPlayer(record, cfg)
        while not p.done:
            p.step()
    except ReplayError as exc:
        return Verification(False, p.index if "p" in locals() else 0, str(exc))
    res = record.get("result", {})
    st = p.state
    if res and (res.get("winner") != st.winner or res.get("score") != [st.count(0), st.count(1)]):
        return Verification(False, p.index, "Re-simulated result differs from the recorded result", st)
    return Verification(True, p.index, f"Verified: {p.index} turns reproduce the recorded match exactly", st)
