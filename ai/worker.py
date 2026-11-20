"""Run AI decisions off the render thread, with cancellation.

Python threads share one interpreter lock, so a CPU-heavy search in a thread
would stutter the 3-D view. :class:`AIWorker` therefore runs searches in one
dedicated **process** (true parallelism); if a process cannot be started it
falls back to a thread with the same interface.

Safety rules:

* Requests and results are plain data (state dict, agent spec, seed,
  opponent-model snapshot -> decision dict). The worker never sees Panda3D
  objects, so it cannot touch the scene graph; the UI applies results on the
  main thread when it polls.
* **Cancellation** is cooperative and per request: a shared counter
  ``cancel_upto`` holds the highest cancelled request id. A search for request
  ``rid`` polls ``cancel_upto >= rid`` between physics simulations and stops
  within one simulation, returning its best legal action so far (flagged
  ``cancelled``). Results of cancelled requests are discarded.
* If a cancelled job still has not returned after ``stuck_after`` seconds the
  worker process is terminated and restarted (hard cancellation).
* The caller (the UI) always has a legal fallback shot if no decision arrives
  in time.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import queue
import threading
import time
from typing import Any, Callable

from ai.base_agent import Decision
from ai.factory import agent_from_spec
from ai.opponent_model import OpponentModel
from config import DEFAULT_CONFIG, GameConfig
from game.state import GameState


def compute_decision(payload: dict[str, Any], cancel: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Rebuild the agent from plain data and decide (picklable, process-safe)."""
    state = GameState.from_dict(payload["state"])
    cfg: GameConfig = payload.get("game_cfg") or DEFAULT_CONFIG
    agent = agent_from_spec(payload["spec"], state.to_move, seed=payload.get("seed", 0), game_cfg=cfg)
    snap = payload.get("opponent_model")
    if snap:
        agent.opponent_model = OpponentModel.from_snapshot(snap)
    return agent.decide(state, payload.get("time_budget"), cancel).to_dict()


def _serve(req_q, res_q, cancel_upto) -> None:
    """Worker loop (runs in the child process or in the fallback thread)."""
    while True:
        item = req_q.get()
        if item is None:
            return
        rid, payload = item
        if cancel_upto.value >= rid:
            res_q.put((rid, "cancelled", None))
            continue
        try:
            res = compute_decision(payload, lambda: cancel_upto.value >= rid)
            res_q.put((rid, "done", res))
        except Exception as exc:  # reported to the UI, which plays a fallback shot
            res_q.put((rid, "error", repr(exc)))


class _ThreadValue:
    def __init__(self) -> None:
        self.value = 0


class AIWorker:
    def __init__(self, use_processes: bool = True, stuck_after: float = 5.0) -> None:
        self._ids = itertools.count(1)
        self.stuck_after = stuck_after
        self.use_processes = use_processes
        self.restarts = 0
        self._start()

    # ------------------------------------------------------------------ lifecycle
    def _start(self) -> None:
        self._pending: dict[int, float] = {}       # rid -> submit time
        self._cancelled: dict[int, float] = {}     # rid -> cancel time
        self._results: dict[int, tuple[str, Any]] = {}
        self.proc = None
        self.thread = None
        if self.use_processes:
            try:
                ctx = mp.get_context("spawn")
                self.req_q = ctx.Queue()
                self.res_q = ctx.Queue()
                self.cancel_upto = ctx.Value("q", 0, lock=False)
                self.proc = ctx.Process(target=_serve, args=(self.req_q, self.res_q, self.cancel_upto),
                                        daemon=True, name="marble-fight-ai")
                self.proc.start()
                self.mode = "process"
                return
            except Exception:  # pragma: no cover - platform dependent
                self.proc = None
        self.req_q = queue.Queue()
        self.res_q = queue.Queue()
        self.cancel_upto = _ThreadValue()
        self.thread = threading.Thread(target=_serve, args=(self.req_q, self.res_q, self.cancel_upto),
                                       daemon=True, name="marble-fight-ai")
        self.thread.start()
        self.mode = "thread"

    def warm_up(self) -> None:
        """The process is started in the constructor; nothing else is needed."""

    def restart(self) -> None:
        """Hard cancellation: kill the worker and start a fresh one."""
        self.restarts += 1
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.join(1.0)
            except Exception:
                pass
        else:
            self.cancel_upto.value = 1 << 60  # make the old thread drop everything
            self.req_q.put(None)
        self._start()

    def shutdown(self) -> None:
        try:
            self.cancel_upto.value = 1 << 60
            self.req_q.put(None)
            if self.proc is not None:
                self.proc.join(1.0)
                if self.proc.is_alive():
                    self.proc.terminate()
        except Exception:
            pass

    # ------------------------------------------------------------------ requests
    def request(self, state: GameState, spec: dict[str, Any], seed: int, opponent_snapshot: dict | None,
                time_budget: float | None, game_cfg: GameConfig = DEFAULT_CONFIG) -> int:
        if self.proc is not None and not self.proc.is_alive():
            self.restart()
        payload = {"state": state.to_dict(), "spec": spec, "seed": seed, "opponent_model": opponent_snapshot,
                   "time_budget": time_budget, "game_cfg": game_cfg}
        rid = next(self._ids)
        self._pending[rid] = time.perf_counter()
        self.req_q.put((rid, payload))
        return rid

    def _drain(self) -> None:
        while True:
            try:
                rid, status, data = self.res_q.get_nowait()
            except (queue.Empty, OSError, EOFError):
                break
            self._pending.pop(rid, None)
            if rid in self._cancelled:
                self._cancelled.pop(rid, None)
                self._results[rid] = ("cancelled", None)
            else:
                self._results[rid] = (status, data)

    def poll(self, rid: int) -> tuple[str, Decision | None, str]:
        """Return ("pending" | "done" | "error" | "cancelled" | "unknown", decision, error text)."""
        self._drain()
        self._check_stuck()
        if rid in self._results:
            status, data = self._results.pop(rid)
            if status == "done":
                return "done", Decision.from_dict(data), ""
            return status, None, "" if data is None else str(data)
        if rid in self._pending:
            if rid in self._cancelled:
                return "cancelled", None, ""
            return "pending", None, ""
        return "unknown", None, ""

    def elapsed(self, rid: int) -> float:
        t0 = self._pending.get(rid)
        return time.perf_counter() - t0 if t0 is not None else 0.0

    def cancel(self, rid: int) -> None:
        if rid in self._pending:
            self.cancel_upto.value = max(self.cancel_upto.value, rid)
            self._cancelled[rid] = time.perf_counter()

    def cancel_all(self) -> None:
        """Cancel every outstanding request (their results will be ignored)."""
        if self._pending:
            self.cancel(max(self._pending))
            for rid in self._pending:
                self._cancelled.setdefault(rid, time.perf_counter())

    def busy(self) -> bool:
        self._drain()
        return bool(self._pending)

    def _check_stuck(self) -> None:
        now = time.perf_counter()
        stuck = [rid for rid, t in self._cancelled.items() if rid in self._pending and now - t > self.stuck_after]
        if stuck:
            self.restart()
