# Marble Fight — AI Algorithms and Design

*Marble Fight: Intelligent Marble Battle Arena* — "Aim Smart. Strike Hard. Own the Arena."

This document explains how the game and its AI work, in the order a project
presentation would cover them. Every constant quoted here is taken from the code
(`config.py`, `data/arenas.json`, `ai/*.py`). Section 21 contains **measured**
results and says exactly how they were produced.

---

## 1. Overall game design

Marble Fight is a 3-D take on *Kancha*, the traditional Bangladeshi marble game.
Two players each own 3 glass marbles (configurable 1–7) inside a circular arena of
radius 10. On a turn the player picks **one of their marbles, a direction, a power
and optionally an ability**, shoots, and then every marble rolls, collides and
settles. A marble whose centre leaves the circle is eliminated. You win by
eliminating all enemy marbles.

The project is split into strictly separated layers:

| Layer | Package | Knows about rendering? |
|---|---|---|
| Rules, state, physics, transition | `game/` | no |
| AI agents, evaluation, fuzzy logic, opponent model | `ai/` | no |
| Match controller (seats, history, events) | `game/match.py` | no |
| Analytics, tournaments, logging | `analytics/` | no |
| 3-D view, HUD, menus | `ui/` | yes |

The 3-D scene only **plays back** frames produced by the physics engine; it never
moves a marble by itself. This is why the AI's predictions and what you see are
always identical.

## 2. Game-state representation (`game/state.py`)

Between turns all marbles are at rest, so a state needs no velocities:

```
GameState(
  marbles:   tuple[MarbleState(mid, owner, x, y, alive)]
  to_move:   0 | 1
  turn:      completed shots so far
  abilities: ((precision, power, spin) uses for P1, (...) for P2)
  arena_id, turn_limit,
  clock:     seconds of simulated motion so far (drives rotating bumpers)
  winner:    None | 0 | 1 | -1 (draw), end_reason
)
```

States are **frozen dataclasses**. A transition always returns a new state, so
search algorithms can explore thousands of futures without touching the live match
(tested in `test_original_state_is_not_mutated`).

## 3. Legal-action representation (`game/actions.py`)

`Action(marble_id, angle, power, ability)`:

* `angle` — radians in the world frame, any finite value;
* `power` — normalised launch power in `[0.15, 1.0]`;
* `ability` — `none`, `precision`, `power`, `spin_left`, `spin_right`.

`validate_action` rejects: a finished match, the wrong player, unknown/enemy/
eliminated marbles, non-finite angle or power, power out of range, unknown
abilities, and abilities with no uses left. **Every** action — human, Minimax,
MCTS, fallback — passes through this function before it is simulated.

**Candidate generator (shared by both agents).** Search needs a finite action set.
`generate_candidate_actions` produces, for each own marble:

* 16 uniform directions × 3 power levels (0.35 / 0.65 / 1.0);
* a gentle retreat toward the centre;
* for each enemy marble: a **direct hit** (medium, high power) and a
  **ghost-ball push** — the aim point that makes the struck marble travel radially
  outward toward the nearest boundary (target − 2r·outward unit vector), used when
  the cut angle is under 75°;
* ability variants (Power / Precision on aimed shots, Spin on uniform directions
  and as a curve around a blocking marble) while uses remain.

Each candidate gets a cheap geometric `priority` (enemy edge danger, distance,
blocked line, outward alignment, own risk). Priorities only **order** candidates;
the agents always evaluate them with the real physics. At the opening position the
generator returns about 250 legal candidates.

## 4. Deterministic physics model (`game/physics.py`)

A 2-D tabletop simulation; the renderer lifts positions into 3-D.

| Quantity | Value |
|---|---|
| Timestep | fixed `dt = 1/60 s`, split into `n = ceil(v_max·dt / 0.18)` equal sub-steps |
| Marble radius / mass | 0.45 / 1.0 |
| Launch speed | `power × 10.5` units/s (Power Shot ×1.35) |
| Rolling friction | `|v| ← |v| − (3.2·μ + 0.15·|v|)·h`, clamped at 0 (never reverses) |
| Surface friction μ | arena base (Classic sand 1.4, Neon 0.9, Magnetic 1.0) × zone multiplier (Grip ×2.2, Glide ×0.4, Slip ×0.35) |
| Marble–marble restitution | e = 0.90 |
| Marble–bumper restitution | e = 0.80 (bumpers have infinite mass) |
| Sleep threshold | speed < 0.06 → set to exactly 0 |
| Settled | all marbles asleep for 3 consecutive frames |
| Safety cap | 14 s of simulated time |

**Integration** is semi-implicit Euler: forces (magnets, spin) update velocity,
friction removes speed, then position is advanced.

**Collisions** use circle–circle geometry. If two marbles overlap, both are moved
apart along the contact normal `n` by half the overlap (equal masses). If they are
approaching (`v_rel·n < 0`) an impulse is applied:

```
j = −(1 + e) · (v_rel · n) / (1/m1 + 1/m2)
v1 ← v1 − (j/m1)·n,   v2 ← v2 + (j/m2)·n
```

Momentum is conserved exactly and, because `e ≤ 1`, kinetic energy can never
increase (tests: `test_head_on_collision_momentum_and_restitution`,
`test_collision_never_creates_energy`). Sub-steps limit travel to 0.18 per step —
far below a marble diameter (0.9) — so marbles cannot tunnel through each other
(`test_no_tunnelling_at_high_speed`).

**Elimination rule (used everywhere).** A marble is eliminated the instant its
centre is farther than `R + tolerance` from the arena centre (`R = 10`,
`tolerance = 0`). It is removed from all further collisions immediately and can
never return.

**Abilities (exact effects).**
* *Precision (stop) shot*: after the shooter's first marble contact, its friction is
  multiplied by 4, so it stops near the impact point instead of following through.
* *Power shot*: launch speed × 1.35.
* *Spin shot*: the shooter's velocity is rotated at `0.9 × spin` rad/s (spin = ±1,
  decaying with time constant 1.2 s, halved on every contact) — the shot curves.

**Arena features.**
* *Static bumpers* (Neon Crystal): infinite-mass circles.
* *Rotating bumpers* (Magnetic Hazard): position `= orbit·(cos θ, sin θ)` with
  `θ = phase + ω·clock`. `clock` is part of the state and only advances while
  marbles move, so the AI reproduces the bumpers' future positions exactly.
* *Magnetic vortices*: pull rolling marbles toward their core with acceleration
  `strength × min(1, speed)`; the pull fades as a marble slows, so friction always
  wins eventually and every shot settles.

**Truncation rule.** If 14 s of motion elapse, all velocities are set to zero where
the marbles stand, decided eliminations are kept, and `truncated = True` is
recorded in the turn result and the event feed (it has never occurred in normal
play; see the tournament logs).

**Balancing note (measured, not guessed).** The Classic court originally used
friction 1.0. A 24-match tournament showed Player 1 winning **16 of 16** Classic
matches (avg ≈ 7 turns): the opening shot decided the game, so Classic told us
nothing about the algorithms. Sweeping the base friction with real
Minimax-vs-MCTS matches (10–16 per value) gave P1:P2 = 9:1 at 1.0, 8:4 at 1.3,
12:4 at 1.35, 2:8 at 1.4, 4:8 at 1.5 and 5:5 at 1.8 (but every 1.8 game reached the
60-turn limit). The seat bias flips chaotically near the crossover, so no single
value is "fair"; 1.4 (a sandy court) was chosen because it produces long,
strategic games (≈ 39 turns) like the other arenas. Fairness for algorithm
comparison comes from **seat rotation** in tournaments (Section 16), not from the
arena. The data from the old setting is kept in `data/archive/` and
`reports/archive/`.

**Shot power (measured, not guessed).** Players found a 100 % shot too weak at the
earlier launch speed of 9.5 units/s: on the sandy Classic court it rolled only 8.3 units
(the arena is 20 across). Raising the speed changes the game balance, so each candidate
was played out in real Minimax-vs-MCTS matches with alternating seats:

| Max launch speed | 100 % shot range, Classic / Neon | Seat 1 : seat 2 wins | Avg match length |
|---|---|---|---|
| 9.5 (previous) | 8.3 / 11.8 units | 7 : 23 (30 matches) | long, many reach the 60-turn limit |
| **10.5 (chosen)** | **9.9 / 14.1 units** | **29 : 19 (48 matches); 75 : 33 over all 108 measured** | about 9–13 turns |
| 11.0 | 10.8 / 15.2 units | 39 : 9 (48 matches) | 7–27 turns |
| 12.5 | 13.6 / 19.0 units | 16 : 2 (18 matches) and 25 : 5 (30) | about 5 turns |

Above about 11 units/s almost every shot is a knockout, so the game turns into a race that
the first shooter wins; moving the start ring toward the centre (radius 5 → 2.5) did not
fix that (seat 1 still won 16–17 of 18). 10.5 is the strongest setting that stays reasonably
balanced; seat 1 still wins about 69 % of matches there (75 of 108), so tournaments
alternate seats to keep algorithm comparisons fair, and every match setup offers a
**two-round series** (`game/series.py`) in which the other player shoots first in round 2;
the series is decided by round wins, then by marbles remaining over both rounds. The
rules and physics of each round are unchanged, so existing replays stay valid. The Power Shot ability multiplies it by 1.35 (≈ 17 units on Classic, ≈ 24 on
Neon), and the HUD shows that boosted strength when it is selected. Results measured at the
old speed are archived in `data/archive/speed_9.5/` and `reports/archive/speed_9.5/`; the
rejected 12.5 tournament is in `data/archive/speed_12.5_rejected/`.

## 5. The turn transition function (`game/simulation.py`)

```
simulate_turn(state, action) -> TurnResult(state', eliminated, sim_time, truncated, contacts, frames, ...)
```

It validates the action, converts it to a launch velocity, runs the physics until
the arena settles, removes eliminated marbles, decrements ability uses, advances
`turn` and `clock`, switches the player, and checks terminal conditions. It never
mutates its input and is fully deterministic (`test_simulation_is_deterministic`).
**The live match, the trajectory preview, Minimax and MCTS all call this one
function.**

## 6. Minimax and its role (`ai/minimax.py`)

Minimax treats the game as a two-player, zero-sum, perfect-information game. One
**ply is one complete turn** (choose a shot and resolve all physics); players never
alternate during physics frames. From the point of view of the agent (MAX):

```
value(s) = evaluate(s)                                    if depth = 0 or s terminal
value(s) = max_a [ edge(s,a) + value(result(s,a)) ]       if MAX to move
value(s) = min_a [ edge(s,a) + value(result(s,a)) ]       if MIN to move
```

`edge(s,a)` is the fuzzy-risk penalty of the shot (Section 11): `−w·risk` when the
agent shoots, `+w·risk` when the opponent shoots (`w = 30`). It is zero for a shot
that wins the match, so fuzzy logic can never override a terminal win.

**Search procedure per decision**

1. Generate candidates; keep the best `root_width` at the root and `inner_width`
   at inner nodes (beam limits by priority).
2. *Move ordering*: when a node's children are interior nodes, all of them are
   simulated (the results are cached and reused) and ordered by static value —
   best first for MAX, worst first for MIN. When the children are leaves, they are
   simulated lazily in priority order so that cut-offs save physics work.
3. *Iterative deepening*: depth 1, 2, … `max_depth`. The best move of the previous
   iteration is searched first. If the node or time budget is exhausted during an
   iteration, that iteration is discarded and the last **complete** depth is used.
   If even depth 1 cannot finish, the highest-priority legal shot is played and the
   decision is flagged as a fallback.
4. A transposition-style **simulation cache** (state key + action → child, risk)
   avoids re-simulating the same shot in later iterations.

**Budgets (`config.py DIFFICULTY_PRESETS`)** — the difficulty really changes the search:

| Preset | depth | root width | inner width | node budget | time cap |
|---|---|---|---|---|---|
| Easy | 2 | 12 | 6 | 400 | 1.5 s |
| Normal | 3 | 24 | 10 | 3000 | 3 s |
| Hard | 3 | 36 | 14 | 7000 | 8 s |

The node budget counts **new physics simulations**, so it is machine-independent.

## 7. Alpha-Beta pruning and its efficiency

Alpha-beta carries the best value MAX can already guarantee (α) and the best value
MIN can already guarantee (β). As soon as `α ≥ β` at a node, its remaining
children cannot influence the decision and are skipped (a *cutoff*). With edge
values, each child is searched with the shifted window `(α − e, β − e)`, which keeps
pruning exact.

Correctness is tested three ways:
* the textbook tree (Russell & Norvig fig. 5.5) — value 3, exactly 2 branches pruned;
* 600 random trees (with and without edge values) — alpha-beta value equals plain
  minimax every time and never visits more nodes;
* real marble positions — both functions run on the same `MarbleSearchProblem` and
  agree (`test_alphabeta_agrees_with_plain_minimax_on_game_state`).

The agent reports `nodes_visited`, `nodes_evaluated`, `cutoffs`, `branches_pruned`,
`depth_reached` and `decision_time` for every decision. Measured savings are in
Section 21.

## 8. Monte Carlo Tree Search (`ai/mcts.py`)

Each iteration performs the four standard phases:

1. **Selection** — from the root, while the node is fully expanded (see widening),
   move to the child with the largest UCT value.
2. **Expansion** — add one untried action, in candidate-priority order, and
   resolve it with `simulate_turn`. *Progressive widening* limits a node with `N`
   visits to `ceil(2 · N^0.5)` children; without it the ~250 root actions would
   each get one visit and the tree would never deepen.
3. **Simulation (rollout)** — from the new node play `rollout_depth` further turns.
   Policy: with probability ε = 0.2 a random shot from the 12 best-priority
   candidates; otherwise sample 3 of them, simulate each, keep the one with the
   best `evaluate(child, mover) − 30·risk`. The final state is scored:
   terminal → 1 / 0.5 / 0; otherwise `logistic((evaluation + path edge values) / 160)`.
4. **Backpropagation** — add the reward to every node on the path.

The final move is the root child with the **most visits** (ties: higher mean reward).
If the budget allows no iteration, a legal fallback shot is returned and flagged.

| Preset | iterations | rollout depth | candidate width | time cap |
|---|---|---|---|---|
| Easy | 40 | 1 | 20 | 1.5 s |
| Normal | 160 | 2 | 48 | 3 s |
| Hard | 500 | 2 | 64 | 8 s |

Randomness uses `random.Random(seed)`; the match derives a per-decision seed from
the match seed, turn and player, so seeded games replay identically
(`test_seeded_matches_are_reproducible`).

## 9. The UCT formula

```
UCT(child) = W_child / N_child  +  c · sqrt( ln N_parent / N_child ),     c = 1/√2
```

`W/N` is the mean reward (exploitation); the square-root term grows for rarely
tried children (exploration). An unvisited child scores +∞.

**Reward perspective.** The reward `r ∈ [0,1]` of an iteration is computed for the
root player. Each node stores rewards for the player who made the move *into* that
node (`node.mover`): it adds `r` if `mover` is the root player, otherwise `1 − r`.
Therefore every parent simply maximises `W/N` over its children, whichever player
is to move there (`test_backpropagation_uses_mover_perspective`).

## 10. Fuzzy membership functions (`ai/fuzzy_risk.py`)

The evaluator estimates the danger a shot creates **for the shooter's own side**.

| Variable | Fuzzy sets (trapezoid `T(a,b,c,d)` / triangle `Δ(a,b,c)`) |
|---|---|
| edge distance (units) | near `T(0,0,1.5,3.5)`, medium `Δ(2,4.5,7)`, far `T(5.5,7.5,20,20)` |
| power (effective) | low `T(0,0,0.3,0.5)`, medium `Δ(0.35,0.65,0.9)`, high `T(0.75,1,2,2)` |
| alignment (cos shot vs outward radial) | inward `T(−1,−1,−0.6,−0.2)`, sideways `Δ(−0.5,0,0.5)`, toward_edge `T(0.3,0.7,1,1)` |
| predicted stop margin (units) | unsafe `T(0,0,0.8,2)`, marginal `Δ(1,2.5,4)`, safe `T(3,5,20,20)` |
| opponent threat [0,1] | low `T(0,0,0.2,0.45)`, high `T(0.35,0.65,1,1)` |
| surface friction | slippery `T(0,0,0.5,0.8)`, normal `Δ(0.6,1,1.5)`, grippy `T(1.3,1.8,5,5)` |
| **output risk** [0,1] | low `T(0,0,0.15,0.35)`, moderate `Δ(0.25,0.5,0.75)`, high `Δ(0.6,0.78,0.92)`, very_high `T(0.85,0.95,1,1)` |

Neighbouring sets overlap, so a value such as edge distance 3 is partly *near* and
partly *medium*. The crisp inputs come from the actual simulated shot: the stop
margin is the shooter's predicted final distance to the rim (0 if it is
eliminated) and the threat is how well the best enemy marble is lined up to push
that stopping point outward.

## 11. Fuzzy rules and defuzzification

Mamdani inference with 14 rules (AND = min, OR = max, NOT = 1 − x):

1. edge near AND power high → very_high
2. edge far AND power medium AND stop NOT unsafe → low
3. alignment toward_edge AND power high → high
4. stop safe → low
5. stop unsafe → very_high
6. stop marginal → moderate
7. threat high AND stop marginal → high
8. threat high AND stop safe → moderate
9. friction slippery AND power high → high
10. alignment inward AND (power low OR medium) AND stop NOT unsafe → low
11. edge near AND alignment toward_edge → high
12. edge medium AND power medium AND sideways AND stop NOT unsafe → moderate
13. threat low AND stop safe → low
14. friction grippy AND stop safe → low

Pipeline: **fuzzification** → **rule evaluation** (firing strengths) →
**aggregation** (each consequent set is clipped at its strength, then the max is
taken pointwise) → **centroid defuzzification** over 101 points of [0,1]. Because
the stop-margin sets cover the whole universe, at least one rule always fires
(tested on a 2 160-point input grid). A self-eliminating shot scores ≈ 0.95; a
shot ending safely in the centre ≈ 0.3.

The "NOT stop unsafe" clauses were added after a test showed that a shot which
knocks out its own marble only scored 0.64, because gentle-shot rules were
averaging the result down. The fix keeps the rule base honest: once the
simulation says the shooter dies, the "safe-looking shot" rules no longer apply.

**Integration.** Minimax adds `−30·risk` (own shot) / `+30·risk` (opponent's shot)
as an edge value; MCTS uses the same edge values in its rewards and uses risk in
the rollout policy. It can be switched off per agent (`use_fuzzy`) for controlled
experiments, and it is never applied to a match-winning shot.

## 12. Opponent-history mechanism ("History-Based Adaptive AI", `ai/opponent_model.py`)

Each agent owns an `OpponentModel` that observes only the opponent's **executed**
shots (it never sees future moves). Each shot is classified geometrically:
*aggressive* (aimed within 22° of one of our marbles; split into power ≥ 0.8 and
touch shots), *defensive* (moving toward the centre), or *reposition*. It counts
power, preferred direction sector (8 sectors), targeted marbles, high-power shots
from near the boundary, and success per category.

These counts become small, explainable evaluation adjustments:

```
aggression     = (aggressive + 1) / (shots + 2)              (Laplace smoothing)
defense_mult   = clamp(1 + 1.2·(aggression − 0.5) + 0.3·(edge_power_rate − 0.5), 0.4, 1.8)
threat_mult    = clamp(1 + 0.6·(avg_power − 0.65), 0.6, 1.5)
protect[m]     = 0.6 · share of aggressive shots aimed at our marble m   (after ≥ 2 such shots)
```

`defense_mult` scales the penalty for our marbles near the rim, `threat_mult`
scales the opponent's attack lines, and `protect` adds extra danger weight to the
most-targeted marble. With an empty history every multiplier is exactly 1 (the
default policy). History is reset between matches. No machine learning or neural
network is involved — it is transparent counting.

## 13. Evaluation and reward functions (`ai/evaluation.py`)

`evaluate(state, perspective)`:

* **Terminal**: win `= +(10 000 + 50·survivors − turn)`, loss is the negative, draw
  `= 0`. Faster wins and slower losses are preferred.
* **Non-terminal** (always far below 10 000):

| Feature | Formula |
|---|---|
| material | `120 · (own − enemy)` |
| edge danger | `danger(m) = clamp((4 − d_edge)/4, 0, 1)²`; own: `−40 · defense_mult · Σ danger·(1+protect)`; enemy: `+30 · Σ danger` |
| attack lines | for each enemy marble, best unblocked shooter whose line pushes it outward: `alignment · enemy_danger · distance factor`. Weighted 26 for the side to move, 10 for the waiting side (× threat_mult for the opponent) |
| safety | `1.5 · (own average rim distance − enemy average)` |
| abilities | `4 · (own remaining uses − enemy remaining uses)` |

MCTS maps the evaluation (plus path edge values) to a reward with
`1 / (1 + e^(−v/160))`, and uses exact 1 / 0.5 / 0 for terminal states.

## 14. One physics engine for both algorithms

Both agents inherit from `BaseAgent`, whose `transition()` calls the shared
`simulate_turn`. They use the same candidate generator, evaluation function, fuzzy
evaluator and opponent model. They differ only in **how they spend simulations**:
Minimax spends them breadth-first over a pruned game tree; MCTS spends them on
sampled trajectories guided by UCT.

**Transition cache.** Each agent caches `(state key, action) → (child state, risk)`
during a decision. It is a pure memo of the deterministic transition, so it cannot
change results: tests check that cached children equal fresh `simulate_turn` results and
that Minimax and MCTS make identical decisions (same value, same node counts) with the
cache switched off; only the number of physics simulations differs.

**Computation safety (`ai/worker.py`, `ai/base_agent.Clock`).**
* The AI never touches Panda3D objects. In the game the search runs in a dedicated
  **worker process**; the main thread only polls for results each frame and updates the
  scene itself. Requests and results are plain data (state dict, agent spec, seed,
  opponent-model snapshot → decision dict).
* Budgets: node budget (Minimax), iteration budget (MCTS) and a time budget for both.
  When a budget runs out Minimax keeps its last complete depth and MCTS its most-visited
  root child; `budget_exhausted` is recorded.
* **Cancellation**: every request has an id; a shared counter `cancel_upto` marks
  cancelled ids. Searches poll it between physics simulations (via `Clock.expired`), so a
  cancelled search stops within one simulation (measured ≈ 20 ms end to end) and reports
  `cancelled`. If a cancelled job does not return within 5 s the worker process is
  terminated and restarted. The game cancels outstanding searches on restart, on leaving
  the match and when the hard deadline (`1.5 × time budget + 4 s`) passes.
* **Fallback**: if no decision arrives in time, the worker reports an error, or a returned
  action is illegal, the game plays the agent's legal fallback shot (highest-priority
  legal candidate) and logs it in the event feed.

## 14b. Match replays (`game/replay.py`)

A replay stores the match configuration (arena, marbles, first player, seed, turn limit,
seats, and a hash of every physics/rule constant and the arena layout) plus, per turn,
the executed action, the AI decision statistics, the eliminations and a fingerprint of
the state after the shot (positions rounded to 10⁻⁶, alive flags, turn, abilities, arena
clock, winner).

Replaying rebuilds the initial state from the configuration and, for each turn,
validates the recorded action with the normal rules and re-runs `simulate_turn`. Because
the transition function is deterministic, the new fingerprint must equal the recorded
one; otherwise the replay stops with `ReplayError` naming the turn (and noting if the
configuration hash differs). Missing, malformed or illegal actions are reported, never
filled in. Tests verify that replays reproduce the final state and winner in all three
arenas and in Human-vs-Human, that tampering with one action is detected at exactly that
turn, and that a changed arena configuration is detected.

## 15. Search complexity and practical limitations

With `b ≈ 250` candidates per turn, a full 2-ply tree would need 62 500 physics
simulations (~45 s at ~0.7 ms each), and 3 plies over 15 million. Beam widths
reduce this to at most `24 + 24·10 + 24·10·10 = 2 664` simulations at Normal depth 3,
and alpha-beta plus ordering cut the real number further (≈ 500 measured). MCTS
costs roughly `iterations × (1 + rollout_depth × 3)` simulations.

Limitations: beams can miss a good shot that has a low geometric priority; the
heuristic is hand-tuned; rollouts are short; the physics is deterministic, so the
game has no aiming error (both players, and the human preview, see the same
model).

## 16. Tournament methodology (`analytics/tournament.py`)

* For pairing `A:B`, match *i* seats A as Player 1 when `⌊i / pairings⌋` is even
  and B otherwise, so each algorithm shoots first equally often.
* Arenas rotate `classic → neon → magnetic`.
* Match *i* uses seed `base_seed + i`; MCTS derives per-decision seeds from it.
* Budgets are node/iteration counts, so results reproduce across machines unless a
  time cap is hit (recorded as `budget_exhausted`).
* Matches run in parallel processes; each match is independent, so parallelism does
  not change results.
* Each finished match is appended to `data/match_history.jsonl`; the tournament
  writes `data/tournaments/<id>.json` and `reports/tournament_<id>.md`.

## 17. Performance metrics

Per decision: decision time, physics simulations, fuzzy risk of the chosen shot,
fallback, budget-exhausted and cancelled flags. Minimax adds nodes visited/evaluated, cutoffs, branches pruned,
depth reached and backed-up evaluation. MCTS adds iterations, average rollout
depth, maximum tree depth, root children, visits and mean reward of the chosen
action. Per match: winner, score, turns, duration, eliminations, self-eliminations,
truncations and fallbacks. The Lab's **Search Tree** tab draws one real Alpha-Beta search
(`ai/search_trace.py` records the same steps as `alphabeta`; the tests require identical
values and statistics) and lets the real MCTS agent build its tree one iteration at a time
(`MCTSStepper` calls the agent's own select / expand / rollout / backpropagate; after N steps
the tree equals the one `decide` builds with N iterations). Both views also accept any turn of
the last five saved matches, whose positions are rebuilt by re-simulating the replay. The Algorithm Lab aggregates these from the history
file only; with no data it shows an empty state. Metrics never measured for an agent
(e.g. alpha-beta prunes for MCTS, or fuzzy risk when the evaluator was disabled) are
`None` and displayed as `n/a`, never as 0. Results are also broken down by algorithm
pairing × arena (mirror matches by seat).

## 18. Test strategy (`tests/`, 204 tests, `python -m pytest -q`)

* `test_physics.py` — friction, momentum and restitution, energy, glancing hits,
  tunnelling, dead marbles, elimination, bumpers, rotating bumpers, surfaces,
  magnets, all abilities, settling, truncation, numerical stability (60 random
  shots), determinism, immutability, invalid actions.
* `test_rules.py` — placement (3 arenas × 1–6 marbles, symmetric), turn order,
  ability limits, victory, defeat, simultaneous-elimination draw, turn-limit
  scoring and exact ties, restart, serialisation.
* `test_minimax.py` — textbook tree, 600 random trees vs plain minimax, edge values,
  terminal evaluation, immediate-win preference, avoiding self-elimination,
  statistics, node/time budgets, iterative-deepening fallback.
* `test_mcts.py` — UCT formula, backpropagation perspective, selection, expansion,
  visit-count invariants, progressive widening, terminal rewards, fixed-seed
  reproducibility, tiny budgets, finding an immediate win.
* `test_fuzzy_risk.py` — membership functions, rule activation (AND/OR/NOT),
  aggregation, centroid, output range and coverage, monotonic behaviour, real-shot
  risk, and the opponent model.
* `test_integration.py` — all four AI pairings, Human-vs-AI progression, illegal
  human input, pause/resume/restart, fallback, playback equals the authoritative
  result, worker process/thread decisions, logging/metrics, tournaments.

* `test_ai_safety.py` — clock cancellation, immediate and mid-search cancellation of both
  agents (from another thread), partial MCTS results after cancel, time-budget exhaustion,
  fallback legality on random positions, transition-cache correctness and cached-vs-uncached
  agreement, worker cancel + next request in process and thread mode, worker error
  reporting, hard restart of a stuck worker, main-loop responsiveness during a search.
* `test_replay.py` — replays reproduce the match in every arena, step-by-step agreement with
  the live history, Human-vs-Human replay, same seed ⇒ same match, tampered / missing /
  illegal actions, truncated recordings, configuration-change detection, file validation.
* `test_audio.py` — bundled assets exist, volume maths, mute, live setting changes and
  clamping, cooldowns, voice pools and the global start cap, missing files, rolling loop.
* `test_ui_modes.py` — the real Panda3D app offscreen: AI vs AI (audio events, pause,
  frame gaps < 0.25 s while the AI searches), Human vs AI (input ignored while a shot
  resolves; the AI replies itself), Human vs Human (alternation, turn banner, input
  blocking), and a finished match → results → replay that ends in the identical state.

MCTS tests never require the same move across different seeds; they check
invariants and use controlled positions.

## 19. Fairness and reproducibility

* Both players use the same `validate_action`, physics, abilities and limits.
* Starting positions are point-symmetric; rotating bumpers are symmetric under a
  180° rotation, so neither side gets a geometric advantage.
* Both agents use the same candidate generator and evaluation; neither sees hidden
  information (the opponent model only reads executed shots).
* Seeds: match seed → per-decision seed; identical configuration and seeds replay
  the identical match.
* Starting seats alternate in tournaments.

## 20. Known limitations and future work

* The candidate generator's beam can hide unusual but strong shots; learning a
  priority function would help.
* Rollouts are short (≤ 2 turns) for speed; longer or parallel rollouts would make
  MCTS stronger at higher cost.
* Minimax depth is limited to 3 plies at interactive speeds; a stronger
  transposition table with state hashing across decisions would help.
* The physics is 2-D (no marble jumps, no true rolling resistance model, no spin
  transfer between marbles).
* Results with few matches are noisy; use 30+ matches per configuration for claims.

## 21. Measured results

All numbers below were produced at the current configuration (launch speed 10.5) by
actually running the code on this machine (12-core laptop CPU). Full tables are in the
generated files in `reports/`. Results from earlier configurations are archived (see the
shot-power note in Section 4).

### 21.1 Alpha-Beta vs plain Minimax on saved states
Source: `reports/benchmark_normal_20261023-215605.md` (`python -m tools.benchmark --rebuild-states`),
9 saved positions (3 arenas × start / after 2 / after 4 turns), depth 2, root width 12, reply width 8.

* Identical root value from plain Minimax and Alpha-Beta in **9 of 9** states.
* Nodes visited: plain 101–109, Alpha-Beta 24–62 (about **43–76 % fewer nodes**).
* Normal-budget decision time on these states: Minimax 0.01–0.81 s, MCTS 0.22–1.36 s.

### 21.2 Minimax vs MCTS tournaments (Normal budget, 30 matches each)
Pairing minimax:mcts, arenas rotated classic → neon → magnetic, seats alternated, base seed 42,
History-Based Adaptive AI on.

| Metric | Fuzzy risk **ON** (`T20261023-215607`) | Fuzzy risk **OFF** (`T20261023-215651`) |
|---|---|---|
| Minimax wins / MCTS wins / draws | **13 / 17 / 0** | **15 / 15 / 0** |
| Minimax avg / max decision time | 0.559 s / 1.89 s | 0.504 s / 1.72 s |
| MCTS avg / max decision time | 1.093 s / 2.81 s | 0.952 s / 2.43 s |
| Minimax nodes visited / branches pruned | 40 430 / 56 575 | 51 425 / 67 045 |
| Minimax average depth reached | 2.75 plies | 2.75 plies |
| MCTS iterations / avg rollout depth / avg visits of chosen shot | 22 240 / 1.45 / 17.2 | 27 680 / 1.35 / 17.5 |
| Physics simulations (Minimax / MCTS) | 49 911 / 100 186 | 61 244 / 116 781 |
| Avg fuzzy risk of chosen shots (Minimax / MCTS) | 0.178 / 0.200 | n/a (disabled) |
| Wins per arena: Classic / Neon / Magnetic | MM 6–4 MCTS / MM 6–4 MCTS / MCTS 9–1 MM | 5–5 / MM 8–2 MCTS / MCTS 8–2 MM |
| Average match length (turns): Classic / Neon / Magnetic | 11.7 / 8.7 / 7.2 | 10.0 / 12.9 / 11.7 |
| Seat-1 / seat-2 wins | 24 / 6 | 22 / 8 |
| Truncated simulations / fallback shots / budget cut-offs | 0 / 0 / 0 | 0 / 0 / 0 |

**Interpretation (with caution — 30 matches per configuration).**
* At this shot power the two algorithms are evenly matched overall (13–17 and 15–15).
  Minimax decides about twice as fast and uses about half the physics simulations.
* The arena matters more than the algorithm: MCTS dominated the Magnetic Hazard Arena
  (17 of 20 across both runs), whose rotating bumpers and vortices reward sampling many
  futures, while Minimax won most matches on the Neon Crystal Arena (14 of 20).
* Fuzzy risk on vs off changed the head-to-head by two matches, which is within sampling
  noise for 30 matches; a larger experiment is needed before claiming an effect.
* There is a **first-mover advantage** at this power (seat 1 won 46 of 60). The tournament
  alternates seats, so each algorithm moved first exactly 15 times per tournament and the
  comparison stays fair; single matches should not be over-interpreted.
* Alpha-Beta pruning skipped more branches than Minimax visited (56 575 pruned vs 40 430
  visited with fuzzy on).
