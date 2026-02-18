# Objective bonus/penalty coefficient check (260216)

- Data: 260216-1, 260216-2, 260216-3 (non-GOx polymers only)
- U0 sign consistency across runs: 12/13 polymers
- Recommended fixed setting for now: `bonus_coef=0.35`, `bonus_deadband=0.05`, `bonus_cap=0.30`
  - Median bonus multiplier: 1.087
  - 90th percentile bonus multiplier: 1.105
  - Bonus-cap reached fraction: 0.31
  - Dynamic range (max/min score): 1.564
  - This keeps upside bonus moderate while preserving strong downside penalty, and suppresses tiny U0 gains (<5%) likely driven by measurement/pipetting variance.
