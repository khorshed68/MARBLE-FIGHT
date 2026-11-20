# Marble Fight — Tournament Report (TEMPLATE)

> This is a **template**. Real reports are generated automatically as
> `reports/tournament_<id>.md` after `python -m analytics.tournament ...` or Tournament Mode finishes.
> Do not copy numbers into this template by hand; quote the generated report instead.

## 1. Purpose
What question does this tournament answer? (e.g. "Does MCTS beat Minimax at equal budget?",
"Does the fuzzy risk evaluator change the win rate?")

## 2. Configuration
| Item | Value |
|---|---|
| Matches | _from report_ |
| Pairings | _from report_ |
| Arena rotation | _from report_ |
| Budget preset (`config.py DIFFICULTY_PRESETS`) | _from report_ |
| Starting-seat rotation | _from report_ |
| Base seed | _from report_ |
| Fuzzy risk / Adaptive AI | _from report_ |

## 3. Results (copy from the generated report)
- Head-to-head:
- Win rate per algorithm:
- Average / maximum decision time:
- Minimax nodes and Alpha-Beta prunes:
- MCTS iterations, rollout depth, selected-action visits:

## 4. Discussion
- Which agent won more, and is the sample large enough to be confident? (Report the number of matches.)
- First-mover effect: compare results when each algorithm sat in seat 1.
- Arena effect: compare the per-arena table.
- Cost: decision time and physics simulations per decision.

## 5. Threats to validity
- Small sample sizes; deterministic seeds reduce but do not remove variance between seeds.
- Budgets are not identical in "work": compare simulations per decision.
- Time caps can cut a search short on slow machines (`budget_exhausted` in decision stats).

## 6. Reproduction
```
python -m analytics.tournament --matches N --pairing minimax:mcts --arenas classic,neon,magnetic --difficulty normal --seed 42
```
