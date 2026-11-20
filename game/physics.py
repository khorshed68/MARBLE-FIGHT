"""Deterministic 2-D tabletop marble physics.

The arena is simulated on a horizontal plane; the renderer later lifts the 2-D
positions into the 3-D scene. This module has **no** rendering, sound or global
state: it only transforms numbers, so the AI can call it thousands of times.

Model (documented in AI_ALGORITHMS.md, section 4):

* Fixed timestep ``dt`` (1/60 s). Each frame is split into ``n`` equal
  sub-steps so that no marble travels more than ``max_substep_travel`` per
  sub-step (adaptive but deterministic: ``n`` depends only on the state).
  Because max_substep_travel (0.18) is far below a marble diameter (0.9),
  marbles cannot tunnel through each other or through bumpers.
* Semi-implicit Euler integration. Forces (magnets, spin) change velocity
  first, then rolling friction removes speed: ``|v| -= (decel*mu + drag*|v|)*h``,
  clamped at zero so friction can never reverse or add motion.
* Marble-marble contacts: equal-radius circle geometry. Overlap is removed by
  moving both marbles apart along the contact normal (mass weighted), then an
  impulse with restitution ``e`` is applied only if they are approaching.
  ``e <= 1`` guarantees no kinetic energy is created by a collision.
* Bumpers are infinite-mass circles (static or rotating with a known velocity).
* Elimination: a marble is eliminated the moment its centre is farther than
  ``arena.radius + elimination_tolerance`` from the arena centre. It is removed
  from all further collisions immediately and can never come back.
* Settling: once every surviving marble is asleep (speed below ``stop_speed``,
  snapped to exactly zero) for ``settle_frames`` consecutive frames.
* Safety cap: after ``max_sim_time`` seconds the simulation is truncated: all
  velocities are set to zero where the marbles stand, eliminations already
  decided are kept, and ``truncated=True`` is reported for debugging.
* Update order is always marble index order, so identical inputs give
  bit-identical outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from config import PhysicsConfig
from game.arena import Arena


@dataclass
class Contact:
    """A collision event (used for effects and for the event feed)."""

    t: float
    a: int            # marble index
    b: int            # other marble index, or -1 for an obstacle
    impulse: float
    x: float
    y: float


@dataclass
class PhysicsResult:
    xs: list[float]
    ys: list[float]
    alive: list[bool]
    eliminations: list[tuple[int, float]]          # (marble index, time of elimination)
    sim_time: float
    frames_simulated: int
    substeps: int
    truncated: bool
    marble_contacts: int
    obstacle_contacts: int
    shooter_first_hit: int                          # index of first marble the shooter touched, -1 if none
    contacts: list[Contact] = field(default_factory=list)
    frames: list[tuple[float, tuple[float, ...], tuple[float, ...], tuple[bool, ...]]] | None = None
    truncated_vx: list[float] | None = None        # velocities discarded by the safety cap (debugging/tests)
    truncated_vy: list[float] | None = None


def run_physics(
    xs_in: list[float],
    ys_in: list[float],
    alive_in: list[bool],
    shooter: int,
    launch_vx: float,
    launch_vy: float,
    arena: Arena,
    cfg: PhysicsConfig,
    clock: float = 0.0,
    spin: float = 0.0,
    precision: bool = False,
    record: bool = False,
    record_contacts: bool = True,
) -> PhysicsResult:
    """Simulate one shot until every marble settles (or the safety cap).

    Inputs are copied; the caller's lists are never modified.
    """
    n = len(xs_in)
    xs = list(xs_in)
    ys = list(ys_in)
    alive = list(alive_in)
    vx = [0.0] * n
    vy = [0.0] * n
    if not (0 <= shooter < n) or not alive[shooter]:
        raise ValueError("shooter must be an alive marble index")
    vx[shooter] = launch_vx
    vy[shooter] = launch_vy

    r = cfg.marble_radius
    two_r = 2.0 * r
    two_r_sq = two_r * two_r
    m = cfg.marble_mass
    inv_m = 1.0 / m
    e_mm = cfg.restitution_marble
    e_ob = cfg.restitution_obstacle
    dt = cfg.dt
    decel0 = cfg.rolling_decel
    drag = cfg.linear_drag
    stop = cfg.stop_speed
    vmax = cfg.max_speed
    r_elim = arena.radius + cfg.elimination_tolerance
    r_elim_sq = r_elim * r_elim
    max_frames = int(math.ceil(cfg.max_sim_time / dt))
    zones = arena.zones
    base_mu = arena.base_friction
    magnets = arena.magnets
    static_obs = [(b.x, b.y, b.r) for b in arena.bumpers]
    rot_obs = arena.rotating_bumpers
    spin_curve = cfg.spin_curve
    spin_decay = math.exp(-1.0 / max(1e-6, cfg.spin_decay_time))  # per second factor base

    fric_mult = [1.0] * n
    spin_s = float(spin)
    braked = False
    first_hit = -1

    eliminations: list[tuple[int, float]] = []
    contacts: list[Contact] = []
    marble_contacts = 0
    obstacle_contacts = 0
    frames = [] if record else None
    if record:
        frames.append((0.0, tuple(xs), tuple(ys), tuple(alive)))

    t = 0.0
    settle = 0
    frame = 0
    substeps_total = 0
    truncated = False
    trunc_vx = trunc_vy = None
    range_n = range(n)

    while True:
        if frame >= max_frames:
            truncated = True
            trunc_vx, trunc_vy = list(vx), list(vy)
            for i in range_n:
                vx[i] = 0.0
                vy[i] = 0.0
            break

        # --- adaptive deterministic sub-stepping ---------------------------
        fastest = 0.0
        for i in range_n:
            if alive[i]:
                s2 = vx[i] * vx[i] + vy[i] * vy[i]
                if s2 > fastest:
                    fastest = s2
        fastest = math.sqrt(fastest)
        nsub = max(1, int(math.ceil(fastest * dt / cfg.max_substep_travel)))
        h = dt / nsub

        for _sub in range(nsub):
            substeps_total += 1
            # 1) forces + friction + integration
            for i in range_n:
                if not alive[i]:
                    continue
                ivx = vx[i]
                ivy = vy[i]
                if ivx == 0.0 and ivy == 0.0:
                    continue
                px = xs[i]
                py = ys[i]
                speed = math.sqrt(ivx * ivx + ivy * ivy)
                if magnets:
                    pull = speed if speed < 1.0 else 1.0
                    for mg in magnets:
                        dx = mg.x - px
                        dy = mg.y - py
                        d2 = dx * dx + dy * dy
                        if d2 < mg.r * mg.r and d2 > 1e-12:
                            d = math.sqrt(d2)
                            k = mg.strength * pull * h / d
                            ivx += dx * k
                            ivy += dy * k
                if i == shooter and spin_s != 0.0 and speed > 1e-9:
                    # lateral acceleration perpendicular to velocity (turn rate = curve*spin rad/s)
                    k = spin_curve * spin_s * h
                    nvx = ivx - ivy * k
                    nvy = ivy + ivx * k
                    # keep speed unchanged by the pure rotation
                    ns = math.sqrt(nvx * nvx + nvy * nvy)
                    if ns > 1e-12:
                        cur = math.sqrt(ivx * ivx + ivy * ivy)
                        ivx = nvx * cur / ns
                        ivy = nvy * cur / ns
                speed = math.sqrt(ivx * ivx + ivy * ivy)
                mu = base_mu
                for z in zones:
                    zx = px - z.x
                    zy = py - z.y
                    if zx * zx + zy * zy <= z.r * z.r:
                        mu = base_mu * z.friction
                        break
                mu *= fric_mult[i]
                new_speed = speed - (decel0 * mu + drag * speed) * h
                if new_speed <= 0.0 or speed <= 1e-12:
                    ivx = 0.0
                    ivy = 0.0
                else:
                    if new_speed > vmax:
                        new_speed = vmax
                    f = new_speed / speed
                    ivx *= f
                    ivy *= f
                vx[i] = ivx
                vy[i] = ivy
                xs[i] = px + ivx * h
                ys[i] = py + ivy * h

            if spin_s != 0.0:
                spin_s *= spin_decay ** h
                if abs(spin_s) < 1e-3:
                    spin_s = 0.0

            # 2) marble-marble collisions (fixed pair order)
            for i in range_n:
                if not alive[i]:
                    continue
                for j in range(i + 1, n):
                    if not alive[j]:
                        continue
                    dx = xs[j] - xs[i]
                    dy = ys[j] - ys[i]
                    d2 = dx * dx + dy * dy
                    if d2 >= two_r_sq:
                        continue
                    if d2 > 1e-18:
                        d = math.sqrt(d2)
                        nx = dx / d
                        ny = dy / d
                    else:
                        d = 0.0
                        nx, ny = 1.0, 0.0
                    # positional correction (equal masses -> half each)
                    corr = (two_r - d) * 0.5
                    xs[i] -= nx * corr
                    ys[i] -= ny * corr
                    xs[j] += nx * corr
                    ys[j] += ny * corr
                    rvn = (vx[j] - vx[i]) * nx + (vy[j] - vy[i]) * ny
                    if rvn < 0.0:
                        jimp = -(1.0 + e_mm) * rvn / (inv_m + inv_m)
                        vx[i] -= jimp * inv_m * nx
                        vy[i] -= jimp * inv_m * ny
                        vx[j] += jimp * inv_m * nx
                        vy[j] += jimp * inv_m * ny
                        marble_contacts += 1
                        if record_contacts:
                            contacts.append(Contact(t, i, j, jimp, xs[i] + nx * r, ys[i] + ny * r))
                        if i == shooter or j == shooter:
                            if first_hit < 0:
                                first_hit = j if i == shooter else i
                            if precision and not braked:
                                fric_mult[shooter] = cfg.precision_brake_multiplier
                                braked = True
                            spin_s *= 0.5

            # 3) obstacle collisions (infinite mass)
            if static_obs or rot_obs:
                if rot_obs:
                    tt = clock + t
                    obs = static_obs + [
                        (rb.orbit * math.cos(rb.phase + rb.omega * tt),
                         rb.orbit * math.sin(rb.phase + rb.omega * tt), rb.r) for rb in rot_obs
                    ]
                    ovel = [(0.0, 0.0)] * len(static_obs) + [rb.velocity(tt) for rb in rot_obs]
                else:
                    obs = static_obs
                    ovel = None
                for i in range_n:
                    if not alive[i]:
                        continue
                    for k, (ox, oy, orr) in enumerate(obs):
                        dx = xs[i] - ox
                        dy = ys[i] - oy
                        lim = r + orr
                        d2 = dx * dx + dy * dy
                        if d2 >= lim * lim:
                            continue
                        if d2 > 1e-18:
                            d = math.sqrt(d2)
                            nx = dx / d
                            ny = dy / d
                        else:
                            d = 0.0
                            nx, ny = 1.0, 0.0
                        xs[i] = ox + nx * lim
                        ys[i] = oy + ny * lim
                        ovx, ovy = ovel[k] if ovel else (0.0, 0.0)
                        rvx = vx[i] - ovx
                        rvy = vy[i] - ovy
                        vn = rvx * nx + rvy * ny
                        if vn < 0.0:
                            vx[i] -= (1.0 + e_ob) * vn * nx
                            vy[i] -= (1.0 + e_ob) * vn * ny
                            obstacle_contacts += 1
                            if record_contacts:
                                contacts.append(Contact(t, i, -1, -(1.0 + e_ob) * vn, ox + nx * orr, oy + ny * orr))
                            if i == shooter:
                                spin_s *= 0.5

            # 4) eliminations
            for i in range_n:
                if alive[i] and xs[i] * xs[i] + ys[i] * ys[i] > r_elim_sq:
                    alive[i] = False
                    vx[i] = 0.0
                    vy[i] = 0.0
                    eliminations.append((i, t + h))

            t += h

        frame += 1

        # --- sleeping / settling ------------------------------------------
        all_asleep = True
        any_alive = False
        for i in range_n:
            if not alive[i]:
                continue
            any_alive = True
            if vx[i] * vx[i] + vy[i] * vy[i] < stop * stop:
                vx[i] = 0.0
                vy[i] = 0.0
            else:
                all_asleep = False

        if record:
            frames.append((t, tuple(xs), tuple(ys), tuple(alive)))

        if not any_alive:
            break
        if all_asleep:
            settle += 1
            if settle >= cfg.settle_frames:
                break
        else:
            settle = 0

    return PhysicsResult(
        xs=xs, ys=ys, alive=alive, eliminations=eliminations, sim_time=t,
        frames_simulated=frame, substeps=substeps_total, truncated=truncated,
        marble_contacts=marble_contacts, obstacle_contacts=obstacle_contacts,
        shooter_first_hit=first_hit, contacts=contacts, frames=frames,
        truncated_vx=trunc_vx, truncated_vy=trunc_vy,
    )


def kinetic_energy(vx: list[float], vy: list[float], mass: float = 1.0) -> float:
    return 0.5 * mass * sum(a * a + b * b for a, b in zip(vx, vy))
