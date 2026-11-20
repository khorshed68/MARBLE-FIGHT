"""Tests for the deterministic physics engine and the transition function."""

import math
import random
from dataclasses import replace

import pytest

from config import DEFAULT_CONFIG, PhysicsConfig
from game.actions import Action, IllegalActionError
from game.arena import Arena, Bumper, RotatingBumper, SurfaceZone, get_arena
from game.physics import run_physics
from game.simulation import simulate_turn
from tests.conftest import make_state

PHYS = DEFAULT_CONFIG.physics
FRICTIONLESS = replace(PHYS, rolling_decel=0.0, linear_drag=0.0, max_sim_time=0.6)
PLAIN = Arena("plain", "Plain", "", 10.0, 1.0)


def shoot(xs, ys, vx, vy, arena=PLAIN, cfg=PHYS, shooter=0, **kw):
    return run_physics(list(xs), list(ys), [True] * len(xs), shooter, vx, vy, arena, cfg, **kw)


# ----------------------------------------------------------------------------- friction
def test_friction_decelerates_and_stops():
    res = shoot([0.0], [0.0], 5.0, 0.0, record=True)
    assert not res.truncated
    dist = res.xs[0]
    ideal = 5.0 ** 2 / (2 * PHYS.rolling_decel)   # constant deceleration only
    assert 0.5 * ideal < dist < ideal              # drag makes it stop sooner
    speeds = [(b[1][0] - a[1][0]) / (b[0] - a[0]) for a, b in zip(res.frames, res.frames[1:]) if b[0] > a[0]]
    assert all(s2 <= s1 + 1e-9 for s1, s2 in zip(speeds, speeds[1:]))   # monotone slow-down


def test_higher_power_travels_farther():
    short = shoot([0.0], [0.0], 3.0, 0.0).xs[0]
    long = shoot([0.0], [0.0], 6.0, 0.0).xs[0]
    assert long > short > 0


# ----------------------------------------------------------------------------- collisions
def test_head_on_collision_momentum_and_restitution():
    e = FRICTIONLESS.restitution_marble
    res = shoot([-3.0, 0.0], [0.0, 0.0], 4.0, 0.0, cfg=FRICTIONLESS)
    assert res.truncated   # frictionless never settles -> safety cap
    v0, v1 = res.truncated_vx
    assert v0 + v1 == pytest.approx(4.0, abs=1e-9)             # momentum conserved
    assert v1 == pytest.approx((1 + e) / 2 * 4.0, abs=1e-9)    # 1-D equal-mass result
    assert v0 == pytest.approx((1 - e) / 2 * 4.0, abs=1e-9)
    assert res.marble_contacts == 1
    assert res.shooter_first_hit == 1


def test_collision_never_creates_energy():
    rng = random.Random(1)
    for _ in range(30):
        xs = [rng.uniform(-4, 4) for _ in range(4)]
        ys = [rng.uniform(-4, 4) for _ in range(4)]
        # keep them apart
        if any(math.hypot(xs[i] - xs[j], ys[i] - ys[j]) < 1.0 for i in range(4) for j in range(i)):
            continue
        a = rng.uniform(0, 2 * math.pi)
        res = shoot(xs, ys, 6 * math.cos(a), 6 * math.sin(a), cfg=replace(FRICTIONLESS, max_sim_time=1.5))
        if not res.truncated:      # every marble left the arena before the cap
            continue
        ke = sum(vx * vx + vy * vy for vx, vy, al in zip(res.truncated_vx, res.truncated_vy, res.alive) if al)
        assert ke <= 36.0 + 1e-6


def test_glancing_collision_deflects_both():
    res = shoot([-3.0, 0.0], [0.0, 0.5], 5.0, 0.0)
    assert res.marble_contacts >= 1
    assert res.ys[1] > 0.5       # target pushed up-and-forward
    assert res.ys[0] < 0.0       # shooter deflected the other way


def test_no_tunnelling_at_high_speed():
    cfg = replace(PHYS, max_speed=40.0)
    res = shoot([-2.0, 0.0], [0.0, 0.0], 35.0, 0.0, cfg=cfg)
    assert res.marble_contacts >= 1
    assert res.shooter_first_hit == 1


def test_dead_marbles_do_not_collide():
    with_dead = run_physics([-3.0, 0.0], [0.0, 0.0], [True, False], 0, 4.0, 0.0, PLAIN, PHYS)
    alone = run_physics([-3.0], [0.0], [True], 0, 4.0, 0.0, PLAIN, PHYS)
    assert with_dead.marble_contacts == 0
    assert with_dead.xs[0] == pytest.approx(alone.xs[0])
    assert with_dead.xs[1] == 0.0 and not with_dead.alive[1]


# ----------------------------------------------------------------------------- boundary
def test_marble_shot_outward_is_eliminated():
    res = shoot([8.0], [0.0], 6.0, 0.0)
    assert res.alive == [False]
    assert res.eliminations and res.eliminations[0][0] == 0


def test_marble_shot_inward_survives():
    res = shoot([8.0], [0.0], -6.0, 0.0)
    assert res.alive == [True]
    assert math.hypot(res.xs[0], res.ys[0]) <= PLAIN.radius


def test_elimination_rule_is_centre_beyond_radius():
    # a marble that stops with its centre just inside the rim is safe, just outside is out
    res_in = shoot([9.0], [0.0], math.sqrt(2 * PHYS.rolling_decel * 0.5), 0.0, cfg=replace(PHYS, linear_drag=0.0))
    assert res_in.alive[0] and 9.3 < res_in.xs[0] <= 10.0


def test_eliminated_marble_never_returns():
    arena = Arena("b", "b", "", 10.0, 1.0, bumpers=(Bumper(11.2, 0.0, 0.6),))   # contact would happen beyond the rim
    res = shoot([8.0], [0.0], 8.0, 0.0, arena=arena)
    assert res.alive == [False]


def test_diagonal_elimination():
    res = shoot([5.0], [5.0], 6.0, 6.0)
    assert res.alive == [False]


# ----------------------------------------------------------------------------- obstacles & surfaces
def test_static_bumper_rebound():
    arena = Arena("b", "b", "", 10.0, 1.0, bumpers=(Bumper(3.0, 0.0, 0.6),))
    res = shoot([0.0], [0.0], 6.0, 0.0, arena=arena)
    assert res.obstacle_contacts == 1
    assert res.xs[0] < 1.0      # bounced back well short of the contact point (x = 1.95)


def test_rotating_bumper_position_is_function_of_clock():
    rb = RotatingBumper(4.0, 0.0, 0.5, 0.7)
    assert rb.position(0.0) == pytest.approx((4.0, 0.0))
    assert rb.position(math.pi) == pytest.approx((4.0 * math.cos(0.5 * math.pi), 4.0 * math.sin(0.5 * math.pi)))
    arena = get_arena("magnetic")
    s0 = make_state([(0, 2.5, 0.0), (1, 0.0, 8.0)], arena_id="magnetic")
    a = Action(0, 0.0, 1.0)                     # straight into the bumper parked at (3.8, 0) at clock 0
    r1 = simulate_turn(s0, a, arena)
    r2 = simulate_turn(s0.with_changes(clock=2.0), a, arena)   # bumper has orbited away by then
    assert r1.obstacle_contacts >= 1 and r1.state.marbles[0].alive
    assert r2.obstacle_contacts == 0 and not r2.state.marbles[0].alive


def test_surface_dependent_friction():
    glide = Arena("g", "g", "", 10.0, 1.0, zones=(SurfaceZone("glide", 0, 0, 9.0, 0.4),))
    sticky = Arena("s", "s", "", 10.0, 1.0, zones=(SurfaceZone("sticky", 0, 0, 9.0, 2.2),))
    d_norm = shoot([-5.0], [0.0], 5.0, 0.0).xs[0]
    d_glide = shoot([-5.0], [0.0], 5.0, 0.0, arena=glide).xs[0]
    d_sticky = shoot([-5.0], [0.0], 5.0, 0.0, arena=sticky).xs[0]
    assert d_sticky < d_norm < d_glide


def test_magnet_bends_rolling_marble():
    from game.arena import Magnet
    arena = Arena("m", "m", "", 10.0, 1.0, magnets=(Magnet(0.0, 1.5, 2.5, 3.0),))
    res = shoot([-4.0], [0.0], 6.0, 0.0, arena=arena)
    assert res.ys[0] > 0.05


# ----------------------------------------------------------------------------- abilities
def test_spin_shot_curves(classic):
    s = make_state([(0, -6.0, 0.0), (1, 0.0, 8.0)])
    left = simulate_turn(s, Action(0, 0.0, 0.65, "spin_left"), classic).state.marbles[0]
    right = simulate_turn(s, Action(0, 0.0, 0.65, "spin_right"), classic).state.marbles[0]
    straight = simulate_turn(s, Action(0, 0.0, 0.65), classic).state.marbles[0]
    assert left.y > 0.3 and right.y < -0.3 and abs(straight.y) < 1e-9


def test_power_shot_is_stronger(classic):
    s = make_state([(0, -6.0, 0.0), (1, 0.0, 8.0)])
    normal = simulate_turn(s, Action(0, 0.0, 0.5), classic).state.marbles[0].x
    power = simulate_turn(s, Action(0, 0.0, 0.5, "power"), classic).state.marbles[0].x
    assert power > normal


def test_precision_shot_brakes_after_contact(classic):
    s = make_state([(0, -4.0, 0.2), (1, 0.0, 0.0)])
    normal = simulate_turn(s, Action(0, 0.0, 0.8), classic).state.marbles[0]
    prec = simulate_turn(s, Action(0, 0.0, 0.8, "precision"), classic).state.marbles[0]
    d_n = math.hypot(normal.x + 4.0, normal.y - 0.2)
    d_p = math.hypot(prec.x + 4.0, prec.y - 0.2)
    assert d_p < d_n


# ----------------------------------------------------------------------------- settle / limits / stability
def test_settling_detection():
    res = shoot([0.0], [0.0], 3.0, 0.0)
    assert not res.truncated
    assert res.frames_simulated < PHYS.max_sim_time / PHYS.dt


def test_max_duration_truncates_deterministically():
    cfg = replace(PHYS, rolling_decel=0.0, linear_drag=0.0, max_sim_time=1.0)
    res = shoot([0.0, 2.0], [0.0, 3.0], 0.5, 0.0, cfg=cfg)
    assert res.truncated
    assert res.sim_time == pytest.approx(1.0, abs=PHYS.dt)
    assert res.xs[0] == pytest.approx(0.5 * res.sim_time, rel=1e-6)


def test_numerical_stability_random_shots():
    arena = get_arena("neon")
    rng = random.Random(7)
    for _ in range(60):
        n = 6
        xs, ys = [], []
        while len(xs) < n:
            x, y = rng.uniform(-7, 7), rng.uniform(-7, 7)
            if math.hypot(x, y) < 8.5 and all(math.hypot(x - a, y - b) > 1.0 for a, b in zip(xs, ys)) \
                    and all(math.hypot(x - ox, y - oy) > 1.2 for ox, oy, _ in arena.obstacle_positions(0)):
                xs.append(x)
                ys.append(y)
        a = rng.uniform(0, 2 * math.pi)
        sp = rng.uniform(1, PHYS.shot_speed_max * 1.35)
        res = run_physics(xs, ys, [True] * n, 0, sp * math.cos(a), sp * math.sin(a), arena, PHYS)
        for i in range(n):
            assert math.isfinite(res.xs[i]) and math.isfinite(res.ys[i])
            if res.alive[i]:
                assert math.hypot(res.xs[i], res.ys[i]) <= arena.radius + 1e-9
        alive = [i for i in range(n) if res.alive[i]]
        for i in alive:
            for j in alive:
                if i < j:
                    assert math.hypot(res.xs[i] - res.xs[j], res.ys[i] - res.ys[j]) > 2 * PHYS.marble_radius - 0.02


# ----------------------------------------------------------------------------- transition function
def test_simulation_is_deterministic():
    arena = get_arena("magnetic")
    s = make_state([(0, 0.0, -5.0), (0, 2.0, -4.5), (1, 0.0, 5.0), (1, -2.0, 4.5)], arena_id="magnetic")
    a = Action(1, 1.7, 1.0, "spin_left")
    r1 = simulate_turn(s, a, arena, record=True)
    r2 = simulate_turn(s, a, arena, record=True)
    assert r1.state == r2.state
    assert r1.frames == r2.frames


def test_original_state_is_not_mutated(classic):
    s = make_state([(0, 0.0, -5.0), (1, 0.0, 5.0)])
    before = s.to_dict()
    simulate_turn(s, Action(0, math.pi / 2, 1.0, "power"), classic)
    assert s.to_dict() == before


def test_turn_result_updates_state(classic):
    s = make_state([(0, 0.0, -5.0), (1, 0.0, 5.0)])
    r = simulate_turn(s, Action(0, 0.3, 0.5), classic)
    assert r.state.to_move == 1 and r.state.turn == 1
    assert r.state.clock == pytest.approx(r.sim_time)
    assert r.state.last_shot.marble_id == 0


@pytest.mark.parametrize("action,msg", [
    (Action(1, 0.0, 0.5), "own marble"),
    (Action(9, 0.0, 0.5), "does not exist"),
    (Action(0, 0.0, 1.5), "outside"),
    (Action(0, 0.0, 0.01), "outside"),
    (Action(0, float("nan"), 0.5), "finite"),
    (Action(0, 0.0, 0.5, "teleport"), "Unknown ability"),
    (Action(2, 0.0, 0.5), "eliminated"),
])
def test_invalid_actions_rejected(classic, action, msg):
    s = make_state([(0, 0.0, -5.0), (1, 0.0, 5.0), (0, 30.0, 0.0, False)])
    with pytest.raises(IllegalActionError, match=msg):
        simulate_turn(s, action, classic)


def test_ability_without_uses_rejected(classic):
    s = make_state([(0, 0.0, -5.0), (1, 0.0, 5.0)], abilities=((0, 0, 0), (2, 2, 2)))
    for ab in ("precision", "power", "spin_left", "spin_right"):
        with pytest.raises(IllegalActionError, match="left"):
            simulate_turn(s, Action(0, 0.0, 0.5, ab), classic)


def test_shooting_on_finished_match_rejected(classic):
    s = make_state([(0, 0.0, -5.0), (1, 0.0, 5.0)]).with_changes(winner=0)
    with pytest.raises(IllegalActionError, match="over"):
        simulate_turn(s, Action(0, 0.0, 0.5), classic)
