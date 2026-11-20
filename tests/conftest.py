"""Shared fixtures for the Marble Fight test-suite."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import DEFAULT_CONFIG  # noqa: E402
from game.arena import get_arena  # noqa: E402
from game.state import GameState, MarbleState  # noqa: E402


def make_state(marbles, to_move=0, arena_id="classic", abilities=((2, 2, 2), (2, 2, 2)), turn=0, turn_limit=60):
    """Build a state from [(owner, x, y), ...] or [(owner, x, y, alive), ...]."""
    ms = []
    for i, m in enumerate(marbles):
        owner, x, y = m[:3]
        alive = m[3] if len(m) > 3 else True
        ms.append(MarbleState(i, owner, float(x), float(y), alive))
    return GameState(tuple(ms), to_move, turn, abilities, arena_id, turn_limit)


@pytest.fixture
def cfg():
    return DEFAULT_CONFIG


@pytest.fixture
def classic():
    return get_arena("classic")
