# MARBLE FIGHT — Intelligent Marble Battle Arena

**Aim Smart. Strike Hard. Own the Arena.**

A 3-D marble-battle game inspired by *Kancha*, the traditional Bangladeshi marble
game, built for a university Artificial Intelligence course. Two sides shoot glass
marbles inside a circular arena; marbles roll, collide, bounce off bumpers and are
eliminated when they leave the circle. Two genuinely different AI agents —
**Minimax with Alpha-Beta pruning** and **Monte Carlo Tree Search (UCT)** — plan
their shots by simulating the real physics. A **fuzzy-logic risk evaluator** and a
**History-Based Adaptive AI** opponent model support both.

For the full academic explanation see **[AI_ALGORITHMS.md](AI_ALGORITHMS.md)**.

---

## Features

* 3-D Panda3D arena with procedural glass marbles (swirl cores, specular shells,
  player-colour rings, shadows, rolling, motion trails, collision sparks and
  elimination bursts) — no downloaded assets.
* Deterministic custom 2-D physics (fixed timestep, sub-stepping, circle collisions
  with restitution, friction zones, static and rotating bumpers, magnetic vortices).
* Three arenas: **Classic Kancha Court** (sandy, high friction), **Neon Crystal Arena** (bumpers,
  grip/glide pads), **Magnetic Hazard Arena** (rotating bumpers, magnetic vortices, slip hazards).
* Abilities with real physics effects: Precision (stop) shot, Power shot, Spin shot.
* Game modes: **AI vs AI Arena** (Minimax vs MCTS, MCTS vs Minimax, Minimax vs Minimax,
  MCTS vs MCTS), **Human vs AI**, **Human vs Human** (local, alternating turns),
  **Tournament Mode**, **Replays**, **Algorithm Lab**.
* **Match replays**: every finished match (and every tournament match) is saved and can be
  re-watched; the replay re-simulates the recorded legal actions with the real rules and
  stops with an error at the first turn that does not reproduce the recording.
* **Audio**: original menu and arena music plus 16 sound effects (launch, power launch,
  rolling loop, glass hits, bumper impacts, elimination, knockout, three abilities, turn
  change, UI, victory, defeat, draw), triggered by real game events, with master / music /
  effects volume and mute.
* **Settings** that all do something: graphics quality, visual effects, FPS meter, AI
  decision visualisation, trajectory previews, shot animation speed, AI-vs-AI playback
  speed, audio volumes, gameplay defaults.
* AI safety: searches run in a separate worker process, can be cancelled within one physics
  simulation, respect node/iteration/time budgets and always have a legal fallback shot.
* AI analysis overlay: chosen shot, alternative candidate trajectories, high-risk
  candidates (red dashed), nodes, prunes, iterations, visits, rewards, fuzzy risk, time.
* Results screen and Algorithm Lab built only from measured data (empty state when none).
* **Two-round series** (option in every match setup): the same pairing plays twice and the
  other player shoots first in round 2, which cancels the measured first-move advantage.
* **Search Tree view** (Algorithm Lab): the real Alpha-Beta search drawn as a tree, with the
  alpha/beta window of every node, the cutoffs and the shots that were pruned; and the real
  MCTS tree, built one iteration at a time (selection, expansion, rollout, backpropagation).
* Search trees for **your own matches**: pick one of the last 5 saved matches and any turn.
* **Why this shot?** During every match the AI panel explains each AI decision in plain
  words, ranks the compared shots (#1 CHOSEN, #2, ... tagged on the board) and **SHOW TREE**
  draws the decision tree of that moment.
* 204 automated tests (`pytest`), including offscreen end-to-end tests of every game mode.

## Game rules

* Each player starts with 3 marbles (configurable 2–5 in the menus, 1–7 in code) on
  point-symmetric spots. Player 1 shoots first.
* On your turn choose **one of your marbles, a direction, a power** and optionally an
  **ability** (2 uses of each per match). The shot is resolved until every marble stops.
* A marble is **eliminated** the moment its centre crosses the glowing boundary ring.
* **Win** by eliminating every enemy marble. If both sides lose their last marble on
  the same shot, it is a **draw**.
* After **60 shots** the match is scored: more surviving marbles wins; if equal, the
  larger total distance of surviving marbles from the boundary wins; an exact tie is a draw.
* **Rounds** (setup screens): *1 round* is a single match. *2 rounds (swap first shot)* plays
  a second round on the same arena in which Player 2 shoots first. The series winner is
  the player with more round wins; on 1-1 (or two draws) the player with more marbles
  remaining over both rounds wins; if that is equal too, the series is a draw. Each round is
  logged and saved as its own replay; RESTART / REMATCH restarts the whole series.
  Why: the first shooter wins about 68 % of single matches (measured), so a single match
  is not fully fair; two rounds give each side the first shot exactly once.

## Technology

Python 3.11+ (tested on 3.14.6) · Panda3D 1.10 (rendering, DirectGUI, audio) ·
custom deterministic physics · pytest · JSON for arenas, settings and logs.
No internet connection, paid API or external AI service is used.

## Folder structure

```
Marble Fight/
├── main.py                 entry point
├── config.py               every gameplay/AI constant + difficulty presets
├── requirements.txt
├── data/arenas.json        arena layouts (zones, bumpers, magnets, colours)
├── game/                   state, actions, rules, physics, simulation, arena, match controller, series, replay
├── ai/                     base_agent, minimax, mcts, evaluation, fuzzy_risk, opponent_model, worker, factory,
│                           search_trace (records the Alpha-Beta tree / steps MCTS for the Lab),
│                           explain (plain-language reasons for a decision)
├── ui/                     app, main_menu, game_screen, arena_view, hud, results_screen,
│                           algorithm_lab, tree_view (search-tree drawing + SHOW TREE popup),
│                           tournament_screen, settings, camera, geometry, audio, playback, widgets
├── analytics/              metrics, tournament (also a CLI), match_logger
├── tools/                  benchmark.py, screenshots.py (offscreen self-check), make_audio.py (original audio)
├── tests/                  physics, rules, minimax, mcts, fuzzy_risk, integration, ai_safety, replay, audio, ui_modes
├── reports/                generated tournament/benchmark reports + TOURNAMENT_REPORT_TEMPLATE.md
├── assets/audio/           original music + sound effects (shipped; regenerate with tools/make_audio.py)
├── assets/generated/       floor textures (created on first run)
├── data/replays/           saved match replays
├── README.md
└── AI_ALGORITHMS.md
```

## Installation (Windows)

Open **PowerShell** or **Command Prompt** in the project folder (`E:\Marble Fight`):

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell refuses to run `activate`, use `.venv\Scripts\Activate.ps1` after
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or use Command Prompt.

## Launch

```bat
python main.py
```

Options: `python main.py --windowed 1280x720`, `python main.py --fullscreen`,
`python main.py --ai-vs-ai --arena neon` (starts a Minimax-vs-MCTS match immediately).

## Controls

| Action | Mouse / key |
|---|---|
| Select one of your marbles | left-click it (or **Tab** to cycle) |
| Aim + power (slingshot) | press on your marble, drag away from the target direction, release to shoot |
| Aim at a point | left-click the floor (then **Space** to shoot) |
| Fine aim / power | **A / D** rotate, **W / S** power |
| Ability | **1** normal, **2** precision, **3** power, **4** spin left, **5** spin right |
| Shoot | release the drag or **Space** |
| Camera | right-drag orbit, mouse wheel zoom, **Q / E** and arrow keys, **C** re-centre |
| Explanations overlay | **H** |
| Decision tree of the last AI shot (pauses) | **T** or **SHOW TREE** in the AI panel |
| Pause | **P**, or **Esc** for the pause menu (resume / restart / menu) |
| Restart match | **R** |
| Mute / unmute audio | **M** |

**Screen fit.** In every match the camera automatically frames the *whole* arena,
rim included, inside the space the HUD panels leave free, for any window size
(16:9, 16:10, ultra-wide, 4:3/5:4, maximised). Resizing the window re-frames the
camera and resizes the HUD; press **C** after zooming or orbiting to return to the full view (distance, tilt and heading).

## Human vs AI

Main menu → **HUMAN vs AI** → choose the opponent algorithm, its difficulty, your side
and the arena → **START MATCH**. On your turn the HUD shows the power meter and
abilities; the trajectory preview shows your shooter's path to its first contact and
the struck marble's initial path. Difficulty changes the real search budget
(Minimax depth/beam, MCTS iterations/rollouts — see `config.py DIFFICULTY_PRESETS`).

## Human vs Human

Main menu → **HUMAN vs HUMAN** → choose arena and marbles → **START MATCH**. Two
players share the mouse and keyboard; the top bar and a banner always show whose turn
it is (Player 1 cyan, Player 2 orange). Input is accepted only after every marble has
stopped, and both players use exactly the same legal actions and physics as the AI modes.

## AI vs AI (main demonstration)

Main menu → **AI vs AI ARENA** → choose the matchup, the difficulty of each side,
arena, marbles, fuzzy risk on/off, adaptive AI on/off and the seed → **ENTER ARENA**
→ **START MATCH**. The agents then play on their own. Controls on the right:
**PAUSE/RESUME**, **STEP ONE TURN**, **SPEED** (0.5x–4x), **EXPLANATIONS ON/OFF**,
**RESTART**, **MAIN MENU**. Before each shot the overlay shows the chosen trajectory
(solid), alternatives (thin) and high-risk candidates (red dashed), each tagged with
its rank (**#1 CHOSEN**, #2, #3 ...). Shortcut: `python main.py --ai-vs-ai`.

**Why did the AI choose this shot?** The AI ANALYSIS panel shows, for the latest AI
decision, a **WHY THIS SHOT** list and the ranked **CANDIDATES**:

* green `+` reasons, red `-` costs, yellow `~` notes, for example *Knocks out enemy
  marble M3*, *Pushes enemy M2 toward the edge (3.0 -> 2.5 from the rim)*, *Leaves own M0
  only 1.2 from the edge*, *Low fuzzy risk (0.13)*, *Next best: ... value 27.4 vs 35.8
  after the reply*, *Search proved a forced win*;
* the size of the search (shots compared, turns ahead, branches pruned; or MCTS
  iterations and visits);
* the candidates with their Minimax value or MCTS visits / mean reward; the numbers match
  the tags on the board.

Every reason is measured: the chosen shot is re-run through the same physics and the
position before and after is compared (marbles, distance to the rim, attack lines), and
the risk, values and visits are the ones the search reported (`ai/explain.py`).
**SHOW TREE** (or **T**) pauses the match and draws the decision tree for exactly that
position (Minimax with its cutoffs, or MCTS with step buttons), next to the shot the
match AI played and its reasons. The tree uses the Lab's small display budget, so the
popup says whether its choice is the same as the shot that was played. **T**, **P** or
**Esc** closes it and the match continues.

## Replays

* At the end of any match the results screen offers **WATCH REPLAY**; the main menu's
  **REPLAYS** page lists the 12 most recent recordings (including tournament matches)
  and shows whether each one **verifies** (re-simulates to the identical result).
* Replay files live in `data/replays/*.json`: the configuration (arena, marbles, first
  player, seed, turn limit, seats, a hash of all rule/physics constants) and, per turn,
  the legal action, the AI decision statistics shown at the time, the eliminations and a
  fingerprint of the resulting state.
* During playback each recorded action is validated with the normal rules and simulated
  again. Missing, malformed or illegal actions, or a state that differs from the
  recording, stop the replay with a clear message; nothing is invented. Controls are the
  same as AI vs AI (pause, step, speed, explanations, restart).

## Audio

* Music: `assets/audio/music/menu.wav` (ambient loop) and `game.wav` (112 BPM arena loop).
  Effects: `assets/audio/sfx/*.wav`. All are **original**, synthesised by
  `python -m tools.make_audio` (no third-party samples, no licence restrictions);
  `python -m tools.make_audio --force` regenerates them.
* Sounds follow real events: one launch per shot, ability sounds, a glass click per
  recorded marble contact (soft or hard by impulse), bumper clangs, elimination and
  knockout stingers, a chime on each turn change, and victory / defeat / draw. The
  rolling loop's volume follows the actual summed marble speed during playback.
* Anti-spam: per-event cooldowns, small voice pools and a cap of 6 sound starts per
  0.1 s window, so pile-ups never become noise.
* Settings → master, music and sound-effects volume (0–100 %) and mute (also **M** in a
  match), saved in `data/settings.json`. Missing audio files are skipped (the game keeps
  running and Settings shows a warning).

## Tournament

In the game: **TOURNAMENT MODE** → set number of matches, pairings, arena rotation,
budget, start rotation, seed → **RUN TOURNAMENT**. Or from the command line:

```bat
python -m analytics.tournament --matches 24 --pairing minimax:mcts --arenas classic,neon,magnetic --difficulty normal --seed 42
```

Add `--pairing minimax:minimax --pairing mcts:mcts` for more pairings, `--no-fuzzy`
or `--no-adapt` for controlled experiments, `--workers 1` to run sequentially.
Results go to `data/tournaments/<id>.json`, `reports/tournament_<id>.md` and the
match history used by the Algorithm Lab.

## Algorithm Lab

Main menu → **ALGORITHM LAB**. Tabs: Overview (win rates, decision time, simulations
per decision), Search Metrics (average and maximum decision time per algorithm; Minimax
nodes visited, leaves evaluated, alpha-beta cutoffs, branches pruned, average and maximum
depth; MCTS iterations, physics simulations, rollout depth, selected-action visits, reward
estimate, tree depth), **Search Tree**, Arenas (results by arena and by algorithm pairing ×
arena), Tournaments, Last Match (per-turn decision time and fuzzy risk), How the AI Works.

**Search Tree** runs the real Minimax agent's Alpha-Beta search on a chosen position and
draws it: the root (MAX), the 4 best shots, the opponent's 3 best replies to each (MIN),
the value and fuzzy-risk edge of every node, the alpha/beta window, the **cutoffs** and the
**pruned** replies (red, dashed: never simulated) and, in gold, the principal variation.
Options: position (knockout trap, Classic opening, Neon opening, or one of your **last 5
saved matches** with a **Turn** selector), depth 1 or 2, fuzzy penalty on/off. It also shows that plain Minimax on the same tree gives the identical
value with more nodes. The beam is narrow so the tree fits on screen; the algorithm is the
same function the agent uses (`ai/search_trace.py`, checked by `tests/test_search_trace.py`).
Try the knockout trap at depth 1 and then depth 2: the greedy choice (a knockout) changes
to a safer shot once the opponent's reply is searched.

Set **Algorithm** to **MCTS (UCT)** to watch the real MCTS agent build its tree (40 iterations,
1-turn rollouts, seed 7). **NEXT ITERATION** runs exactly one iteration and explains its four
phases: the selection path (green), the shot added by expansion (NEW), the rollout reward
and the backpropagation (N + 1, W + reward). Each box shows N (visits), Q (mean reward of
the player who made that shot) and UCT; grey dashed boxes are shots that progressive
widening has not opened yet. **+10**, **RUN ALL** and **RESET** move faster; gold marks the
current choice (most visits) and the panel compares it with Minimax's choice. After N
steps the tree is exactly the one `MCTSAgent.decide` builds with N iterations
(`ai/search_trace.py: MCTSStepper`, checked by `tests/test_search_trace.py`).

**Which game is the tree from?** The panel under the mini map names the position. With
*Last match* / *Recent match 2..5*, it shows that match's date, players, arena and final
score, and the **Turn** selector picks the moment (Turn 3 / 9 | P1 Minimax = the third shot,
taken by Player 1). The position is rebuilt exactly by re-simulating the saved replay, so
the tree is computed on the real board of that turn. The mini map draws the shot that was
**actually played** (white) next to the tree's choice (gold), and the panel shows what the
match's own search reported (Minimax value / depth / nodes, MCTS iterations / visits, or
"human shot"). The tree uses a small budget so it can be drawn; the match used the full
difficulty budget, so the two choices can differ. Replays recorded with other game
settings are not offered, because their positions cannot be rebuilt exactly.
`n/a` means a metric was not measured for that agent; a measured zero is shown as `0`.
All numbers come from `data/match_history.jsonl`; **CLEAR HISTORY** empties it.

## Tests and benchmarks

```bat
python -m pytest -q
python -m tools.benchmark --difficulty normal
python main.py --screenshots shots
python -m tools.make_audio
```

`tests/test_ui_modes.py` starts the real game offscreen and plays AI vs AI, Human vs
AI, Human vs Human and a replay (audio events are counted by a silent backend);
`tests/test_ai_safety.py` covers budgets, cancellation, the worker process, fallback and
responsiveness; `tests/test_replay.py` covers replay determinism and error handling.

The benchmark runs both agents on the same saved states (`data/benchmark_states.json`)
and compares Alpha-Beta with plain Minimax on identical search problems. The
screenshot command renders every screen offscreen (a quick UI self-check).

## AI in one paragraph each

* **Minimax + Alpha-Beta** — searches my shot, the opponent's best reply and my
  answer (one ply = one full physics turn) with iterative deepening, move ordering,
  beam limits and a simulation cache; alpha-beta skips replies that cannot change
  the decision.
* **MCTS (UCT)** — selection by `W/N + c·sqrt(ln N_parent / N)`, progressive-widening
  expansion, short heuristic rollouts, backpropagation from each mover's perspective;
  the most-visited shot is played.
* **Fuzzy risk** — 6 fuzzified inputs, 14 Mamdani rules, max aggregation and centroid
  defuzzification give a 0–1 risk that is added as a penalty to risky shots.
* **History-Based Adaptive AI** — counts the opponent's aggressive/defensive shots,
  power and targets; aggressive opponents make the agent value edge safety more and
  protect the most-targeted marble.

## Troubleshooting

* **`No module named panda3d`** — activate the virtual environment and run
  `python -m pip install -r requirements.txt`.
* **Black window / shader errors on old GPUs** — update the graphics driver; the game
  needs OpenGL with basic shader support.
* **No sound** — sound is optional; enable it in Settings, or check the default audio device.
* **First AI move takes a few seconds** — the AI worker process starts on first use.
* **Fonts look different** — the UI uses Windows' Bahnschrift/Segoe UI fonts when
  present and falls back to Panda3D's built-in font otherwise.

## Known limitations

See [AI_ALGORITHMS.md §20](AI_ALGORITHMS.md#20-known-limitations-and-future-work).
In short: search beams can miss unusual shots, MCTS rollouts are short, the physics
is a 2-D model, and small tournaments are statistically noisy.
