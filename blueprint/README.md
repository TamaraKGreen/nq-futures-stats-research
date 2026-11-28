# NQ RTH scalping blueprint

Pre-registered plan: [`PLAN.md`](PLAN.md). Nothing is tuned on OOT or the 2012-19 lockbox.

## Status
- [x] Phase 0 — engine: data (dev / oot / lockbox), levels, FVG / iFVG (1m / 5m / 15m), 1m fill simulator, unit tests, chart spot-checks
- [ ] Phase 1 — bias layer (B1–B3; base tables → logistic → KNN → HMM → GBM / MLP)
- [ ] Phase 2 — entry families E1–E4 vs random control
- [ ] Phase 3 — exits
- [ ] Phase 4 — meta-label filter
- [ ] Phase 5 — freeze → OOT → lockbox
- [ ] Phase 6 — rulebook + Pine v5 + NinjaScript indicators

## Layout
| Path | |
|---|---|
| `engine/data.py` | `load_split('dev'\|'oot'\|'lockbox')`, cost constants (5 MNQ, $6.50 RT = 0.65 pt, 1-tick slippage) |
| `engine/levels.py` | `build_levels(df)` → every key level as known at each bar's open |
| `engine/fvg.py` | `resample`, `detect` (FVGs + inversions, close or wick mode), `live_at` |
| `engine/sim.py` | `Bars`, `trade`, `run` — conservative 1m fills, 50-pt / 30-min caps |
| `tests/` | `python3 -m pytest -q blueprint/tests` |
| `scripts/spotcheck.py` | charts + level/iFVG tables for chosen days → `spotcheck/` (raw contract prices) |

Run from the repo root: `pip install -r requirements.txt pyarrow scikit-learn hmmlearn matplotlib`.
