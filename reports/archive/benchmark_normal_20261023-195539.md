# Marble Fight — Benchmark on saved states (normal budget)

Measured on 2026-10-23 19:55:39 on this machine. States: `data/benchmark_states.json` (9 positions).

## Decision cost per state

| State | MM time (s) | MM sims | MM nodes | MM pruned | MM depth | MCTS time (s) | MCTS sims | MCTS iters | MCTS visits |
|---|---|---|---|---|---|---|---|---|---|
| classic-start | 0.386 | 521 | 423 | 462 | 3 | 0.701 | 954 | 160 | 9 |
| classic-turn2 | 0.280 | 547 | 469 | 546 | 3 | 0.462 | 898 | 160 | 18 |
| classic-turn4 | 0.008 | 24 | 25 | 0 | 1 | 0.238 | 637 | 160 | 29 |
| neon-start | 0.654 | 530 | 432 | 453 | 3 | 1.252 | 949 | 160 | 11 |
| neon-turn2 | 0.504 | 515 | 414 | 461 | 3 | 0.895 | 904 | 160 | 18 |
| neon-turn4 | 0.552 | 556 | 447 | 438 | 3 | 0.922 | 925 | 160 | 11 |
| magnetic-start | 0.729 | 537 | 440 | 445 | 3 | 1.372 | 905 | 160 | 15 |
| magnetic-turn2 | 0.132 | 101 | 127 | 163 | 2 | 0.948 | 719 | 160 | 11 |
| magnetic-turn4 | 0.018 | 24 | 25 | 0 | 1 | 0.476 | 543 | 160 | 10 |

Mean decision time: Minimax 0.363 s (max 0.729), MCTS 0.807 s (max 1.372).

## Alpha-Beta vs plain Minimax (depth 2, root width 12, reply width 8)

| State | Value (plain) | Value (alpha-beta) | Nodes plain | Nodes alpha-beta | Saved | Branches pruned |
|---|---|---|---|---|---|---|
| classic-start | 1.7767 | 1.7767 | 109 | 39 | 64.2% | 70 |
| classic-turn2 | 2.33 | 2.33 | 109 | 43 | 60.6% | 66 |
| classic-turn4 | 10045.0 | 10045.0 | 101 | 24 | 76.2% | 77 |
| neon-start | 1.6783 | 1.6783 | 109 | 39 | 64.2% | 70 |
| neon-turn2 | 2.4534 | 2.4534 | 109 | 40 | 63.3% | 69 |
| neon-turn4 | 85.5319 | 85.5319 | 109 | 41 | 62.4% | 68 |
| magnetic-start | 6.3359 | 6.3359 | 109 | 52 | 52.3% | 57 |
| magnetic-turn2 | 10158.5305 | 10158.5305 | 109 | 46 | 57.8% | 63 |
| magnetic-turn4 | 10095.0 | 10095.0 | 93 | 28 | 69.9% | 65 |

Identical root value in every state: **True**.
