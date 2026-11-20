"""Two-round series: the fair answer to the first-move advantage.

Measured: the player who shoots first wins about 68 % of single matches. A
series plays the same pairing twice and swaps who shoots first, so each side
gets the first-move advantage exactly once. The rules and physics of each
round are unchanged (every round is an ordinary match, logged and replayable
on its own); only the overall result is decided here:

1. more round wins -> series winner;
2. equal round wins -> more marbles remaining in total over both rounds;
3. still equal -> the series is a draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from game.state import DRAW


@dataclass
class RoundResult:
    first_player: int
    winner: int                    # 0, 1 or DRAW
    score: tuple[int, int]         # marbles remaining for each player at the end of the round
    turns: int
    end_reason: str


@dataclass
class Series:
    total_rounds: int = 2
    first_player: int = 0          # who shoots first in round 1
    rounds: list[RoundResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.total_rounds < 1:
            raise ValueError("a series needs at least one round")
        if self.first_player not in (0, 1):
            raise ValueError("first_player must be 0 or 1")

    # ------------------------------------------------------------------ progress
    @property
    def round_number(self) -> int:
        """1-based number of the round being played (or the last one, when over)."""
        return min(len(self.rounds) + 1, self.total_rounds)

    @property
    def is_over(self) -> bool:
        return len(self.rounds) >= self.total_rounds

    def first_player_of(self, round_index: int) -> int:
        """Who shoots first in round ``round_index`` (0-based): alternates every round."""
        return self.first_player if round_index % 2 == 0 else 1 - self.first_player

    def next_first_player(self) -> int:
        return self.first_player_of(len(self.rounds))

    def record(self, summary: dict[str, Any]) -> RoundResult:
        """Store a finished round from :meth:`Match.summary`."""
        if self.is_over:
            raise ValueError("the series is already over")
        if summary.get("winner") is None:
            raise ValueError("the round is not finished")
        expected = self.next_first_player()
        if summary.get("first_player", expected) != expected:
            raise ValueError("round was played with the wrong first player")
        r = RoundResult(expected, int(summary["winner"]), (int(summary["score"][0]), int(summary["score"][1])),
                        int(summary["turns"]), str(summary.get("end_reason", "")))
        self.rounds.append(r)
        return r

    def reset(self) -> None:
        self.rounds.clear()

    # ------------------------------------------------------------------ result
    def round_wins(self) -> tuple[int, int]:
        return (sum(1 for r in self.rounds if r.winner == 0), sum(1 for r in self.rounds if r.winner == 1))

    def total_marbles(self) -> tuple[int, int]:
        return (sum(r.score[0] for r in self.rounds), sum(r.score[1] for r in self.rounds))

    def winner(self) -> int | None:
        """Series winner (0, 1 or DRAW), or None while rounds remain."""
        if not self.is_over:
            return None
        w = self.round_wins()
        if w[0] != w[1]:
            return 0 if w[0] > w[1] else 1
        m = self.total_marbles()
        if m[0] != m[1]:
            return 0 if m[0] > m[1] else 1
        return DRAW

    def reason(self) -> str:
        if not self.is_over:
            return f"Round {self.round_number} of {self.total_rounds}"
        w, m = self.round_wins(), self.total_marbles()
        if w[0] != w[1]:
            return f"Won more rounds ({w[0]}-{w[1]})"
        if m[0] != m[1]:
            return f"Rounds tied {w[0]}-{w[1]}; decided by marbles remaining over both rounds ({m[0]}-{m[1]})"
        return f"Rounds tied {w[0]}-{w[1]} and marbles remaining tied {m[0]}-{m[1]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rounds": self.total_rounds,
            "rounds": [{"round": i + 1, "first_player": r.first_player, "winner": r.winner, "score": list(r.score),
                        "turns": r.turns, "end_reason": r.end_reason} for i, r in enumerate(self.rounds)],
            "round_wins": list(self.round_wins()),
            "total_marbles": list(self.total_marbles()),
            "winner": self.winner(),
            "reason": self.reason(),
        }
