# Marble Fight — Benchmark on saved states (normal budget)

Measured on 2026-10-23 20:05:05 on this machine. States: `data/benchmark_states.json` (9 positions).

## Decision cost per state

| State | MM time (s) | MM sims | MM nodes | MM pruned | MM depth | MCTS time (s) | MCTS sims | MCTS iters | MCTS visits |
|---|---|---|---|---|---|---|---|---|---|
| classic-start | 0.467 | 654 | 578 | 757 | 3 | 0.708 | 953 | 160 | 8 |
| classic-turn2 | 0.387 | 647 | 589 | 536 | 3 | 0.592 | 953 | 160 | 8 |
| classic-turn4 | 0.239 | 561 | 505 | 410 | 3 | 0.430 | 960 | 160 | 7 |
| neon-start | 0.605 | 530 | 432 | 453 | 3 | 1.049 | 949 | 160 | 11 |
| neon-turn2 | 0.394 | 515 | 414 | 461 | 3 | 0.703 | 904 | 160 | 18 |
| neon-turn4 | 0.450 | 556 | 447 | 438 | 3 | 0.739 | 925 | 160 | 11 |
| magnetic-start | 0.618 | 537 | 440 | 445 | 3 | 1.089 | 905 | 160 | 15 |
| magnetic-turn2 | 0.106 | 101 | 127 | 163 | 2 | 0.777 | 719 | 160 | 11 |
| magnetic-turn4 | 0.014 | 24 | 25 | 0 | 1 | 0.366 | 543 | 160 | 10 |

Mean decision time: Minimax 0.364 s (max 0.618), MCTS 0.717 s (max 1.089).

## Alpha-Beta vs plain Minimax (depth 2, root width 12, reply width 8)

| State | Value (plain) | Value (alpha-beta) | Nodes plain | Nodes alpha-beta | Saved | Branches pruned |
|---|---|---|---|---|---|---|
| classic-start | 4.6063 | 4.6063 | 109 | 32 | 70.6% | 77 |
| classic-turn2 | -21.8598 | -21.8598 | 109 | 64 | 41.3% | 45 |
| classic-turn4 | 8.3361 | 8.3361 | 109 | 60 | 45.0% | 49 |
| neon-start | 1.6783 | 1.6783 | 109 | 39 | 64.2% | 70 |
| neon-turn2 | 2.4534 | 2.4534 | 109 | 40 | 63.3% | 69 |
| neon-turn4 | 85.5319 | 85.5319 | 109 | 41 | 62.4% | 68 |
| magnetic-start | 6.3359 | 6.3359 | 109 | 52 | 52.3% | 57 |
| magnetic-turn2 | 10158.5305 | 10158.5305 | 109 | 46 | 57.8% | 63 |
| magnetic-turn4 | 10095.0 | 10095.0 | 93 | 28 | 69.9% | 65 |

Identical root value in every state: **True**.
