# Marble Fight — Benchmark on saved states (normal budget)

Measured on 2026-10-23 21:56:05 on this machine. States: `data/benchmark_states.json` (9 positions).

## Decision cost per state

| State | MM time (s) | MM sims | MM nodes | MM pruned | MM depth | MCTS time (s) | MCTS sims | MCTS iters | MCTS visits |
|---|---|---|---|---|---|---|---|---|---|
| classic-start | 0.344 | 523 | 417 | 468 | 3 | 0.729 | 952 | 160 | 10 |
| classic-turn2 | 0.315 | 556 | 478 | 537 | 3 | 0.446 | 915 | 160 | 11 |
| classic-turn4 | 0.007 | 24 | 25 | 0 | 1 | 0.221 | 680 | 160 | 31 |
| neon-start | 0.731 | 639 | 554 | 601 | 3 | 1.359 | 932 | 160 | 12 |
| neon-turn2 | 0.704 | 584 | 496 | 519 | 3 | 0.982 | 844 | 160 | 20 |
| neon-turn4 | 0.066 | 76 | 102 | 188 | 2 | 0.432 | 479 | 160 | 16 |
| magnetic-start | 0.813 | 597 | 509 | 466 | 3 | 1.245 | 890 | 160 | 12 |
| magnetic-turn2 | 0.633 | 612 | 522 | 593 | 3 | 0.825 | 827 | 160 | 20 |
| magnetic-turn4 | 0.275 | 304 | 209 | 446 | 3 | 0.596 | 488 | 160 | 18 |

Mean decision time: Minimax 0.432 s (max 0.813), MCTS 0.759 s (max 1.359).

## Alpha-Beta vs plain Minimax (depth 2, root width 12, reply width 8)

| State | Value (plain) | Value (alpha-beta) | Nodes plain | Nodes alpha-beta | Saved | Branches pruned |
|---|---|---|---|---|---|---|
| classic-start | 1.6739 | 1.6739 | 109 | 39 | 64.2% | 70 |
| classic-turn2 | 2.3188 | 2.3188 | 109 | 47 | 56.9% | 62 |
| classic-turn4 | 10045.0 | 10045.0 | 101 | 24 | 76.2% | 77 |
| neon-start | 1.78 | 1.78 | 109 | 40 | 63.3% | 69 |
| neon-turn2 | 2.687 | 2.687 | 109 | 40 | 63.3% | 69 |
| neon-turn4 | -10053.3681 | -10053.3681 | 109 | 40 | 63.3% | 69 |
| magnetic-start | 92.4466 | 92.4466 | 109 | 62 | 43.1% | 47 |
| magnetic-turn2 | 2.2326 | 2.2326 | 109 | 34 | 68.8% | 75 |
| magnetic-turn4 | -10102.8676 | -10102.8676 | 109 | 51 | 53.2% | 58 |

Identical root value in every state: **True**.
