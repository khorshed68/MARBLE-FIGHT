"""Engine-independent playback of a recorded shot.

The physics engine records one frame per fixed timestep. ``Playback`` replays
those frames at any viewing speed, interpolating positions, and reports
collision/elimination events as playback time passes them. It never changes
the outcome: the final frame *is* the authoritative state. Kept free of
Panda3D so it can be unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlaybackEvent:
    kind: str          # "contact" or "eliminated"
    t: float
    a: int
    b: int = -1
    strength: float = 0.0
    x: float = 0.0
    y: float = 0.0


class Playback:
    def __init__(self, frames: list, contacts: list | None = None,
                 eliminations: list[tuple[int, float]] | None = None, speed: float = 1.0) -> None:
        if not frames:
            raise ValueError("Playback needs at least one frame")
        self.frames = frames
        self.speed = speed
        self.t = 0.0
        self.duration = frames[-1][0]
        self._events: list[PlaybackEvent] = []
        for c in contacts or []:
            self._events.append(PlaybackEvent("contact", c.t, c.a, c.b, c.impulse, c.x, c.y))
        for mid, t in eliminations or []:
            self._events.append(PlaybackEvent("eliminated", t, mid))
        self._events.sort(key=lambda e: e.t)
        self._next_event = 0
        self._idx = 0

    @property
    def done(self) -> bool:
        return self.t >= self.duration

    def advance(self, real_dt: float) -> list[PlaybackEvent]:
        """Move playback forward by ``real_dt`` wall seconds; return events passed."""
        self.t = min(self.duration, self.t + max(0.0, real_dt) * self.speed)
        out = []
        while self._next_event < len(self._events) and self._events[self._next_event].t <= self.t + 1e-12:
            out.append(self._events[self._next_event])
            self._next_event += 1
        return out

    def finish(self) -> list[PlaybackEvent]:
        return self.advance(float("inf"))

    def sample(self) -> tuple[list[tuple[float, float]], tuple[bool, ...], list[tuple[float, float]]]:
        """Interpolated (positions, alive flags, velocities) at the current playback time."""
        fr = self.frames
        while self._idx + 1 < len(fr) and fr[self._idx + 1][0] <= self.t:
            self._idx += 1
        while self._idx > 0 and fr[self._idx][0] > self.t:
            self._idx -= 1
        a = fr[self._idx]
        if self._idx + 1 >= len(fr):
            n = len(a[1])
            return [(a[1][i], a[2][i]) for i in range(n)], a[3], [(0.0, 0.0)] * n
        b = fr[self._idx + 1]
        span = b[0] - a[0]
        u = 0.0 if span <= 0 else (self.t - a[0]) / span
        n = len(a[1])
        pos = []
        vel = []
        for i in range(n):
            if a[3][i] and not b[3][i]:
                # eliminated during this frame: hold the last on-table position
                pos.append((a[1][i], a[2][i]))
                vel.append(((b[1][i] - a[1][i]) / span if span else 0.0, (b[2][i] - a[2][i]) / span if span else 0.0))
                continue
            pos.append((a[1][i] + (b[1][i] - a[1][i]) * u, a[2][i] + (b[2][i] - a[2][i]) * u))
            vel.append(((b[1][i] - a[1][i]) / span if span else 0.0, (b[2][i] - a[2][i]) / span if span else 0.0))
        alive = a[3] if u < 1.0 else b[3]
        return pos, alive, vel
