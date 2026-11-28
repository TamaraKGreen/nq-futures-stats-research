# Phase 1 — bias layer: summary (Dev walk-forward only)

Test years 2022, 2023, 2024, 2025 H1; expanding training window, refitted yearly, 5-session purge.
Full tables: [`phase1_bias.txt`](phase1_bias.txt). Out-of-fold predictions are cached for phase 4.

| Question | Baseline | Best model | Skill vs baseline | AUC | Verdict |
|---|---|---|---|---|---|
| B1 09:30 — which level (above / below) is hit first by 11:00 | random walk: the nearer level | none beats it | all models **worse** (−0.6% to −7.6%) | 0.70 (geometry) | **No directional edge.** Distance alone explains it |
| B1 11:00 — same, to 16:00 | random walk | HMM table +0.09% (t 0.4) | noise | 0.73 (geometry) | **No directional edge** |
| B2 10:30 — trend vs rotation day (extension measured in IB widths) | base rate | logistic +14.9% (t 5.8, 4/4 yrs) | real but **mechanical**: IB width alone gives +13.0% | 0.76 | Narrow IB → extends ≥ 1 IB width. Useful for IB-relative targets only |
| B2-ATR — extension beyond IB in ATR units | base rate | logistic +1.0–1.3% (t ≈ 1) | not significant | 0.58 | How far the day really runs after 10:30 is **not predictable** with these features |
| B3 far — hour takes far extreme of prior hour | base rate 23% | logistic(+HMM) +9.6% (t 12.7, 4/4) | table (hour × open position × range) already +7.8% | 0.72 | **Usable, well calibrated** target-hit probability |
| B3 near — hour takes near extreme | base rate 73% | GBM +10.4%, logistic +9.5% (4/4) | table +6.6% | 0.71 | **Usable**, as above |

What didn't help anywhere
- **HMM regime** (2 or 3 states, filtered probabilities): ≤ 0.3 pts of skill added in any task.
- **KNN** (k 25/50/100): always behind logistic regression.
- **Order flow** (overnight / IB / last-hour delta): no gain on any label.
- **GBM / MLP** (B3): match logistic regression, never clearly beat it.

What this means for the blueprint
1. A **directional bias filter isn't supported** by the data. Price goes to the nearer level about as often as a
   random walk would. Entries have to carry their own directional edge.
2. Layer 1 becomes a **context / target-probability layer**:
   - P(near / far extreme of the previous hour is taken this hour), from logistic regression. Simple enough to put
     in Pine / NinjaScript as a formula.
   - IB width class → expected extension in IB widths (sets IB-based targets).
3. The simplest model was as good as the complex ones every time. Logistic regression can go straight into the
   indicators.
