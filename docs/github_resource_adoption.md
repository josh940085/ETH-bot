# GitHub Resource Adoption Notes

This repo should use external trading projects as engineering references, not as strategy sources copied into live trading.

## Adopted First

- `freqtrade/freqtrade`: use the protection model: live/dry-run evidence, explicit blockers, and no leverage promotion while the active profile is underperforming.
- `freqtrade` leverage guidance: keep leverage capped at 5x and require stop/TP protection to remain Binance-confirmed.
- `vectorbt`: useful later for fast monthly5 parameter sweeps, but it is not on the live path.
- `hftbacktest`: useful later for tick/order-book replay when validating short samples such as 8h or 12h windows.

## Local Implementation

- `monthly5_risk_audit.py` audits recent Binance fills, monthly5 shadow performance, and promotion blockers.
- `monthly5_batch_validator.py` vectorizes the existing monthly artifact audit with pandas. It compares every candidate across train and holdout periods, checks the 5% monthly floor, flat time, missing months, leverage, suspicious exact-floor saturation, and trade-level cost evidence.
- `verify_monthly5_runtime.py` remains the runtime gate for the live service.
- Direct takeover must not bypass performance blockers such as:
  - `shadow_monthly_target`
  - `shadow_rolling_monthly_target`
  - `active_underperforming_plan`
  - `recovery_probe_probe_failed`

## Deferred

- Do not add large backtesting frameworks to the live service process. The current NumPy/pandas candidate matrices already provide the needed batch operations without adding `vectorbt` to production dependencies.
- Do not copy public GitHub trading strategies into production.
- Add `vectorbt` only if portfolio-order simulation is needed beyond the existing matrix engine. Evaluate `hftbacktest` separately when tick/order-book archives are available.

Run the current validator from the repository root:

```sh
./.venv/bin/python monthly5_batch_validator.py \
  --candidate postlock_scale0.15_floor_pdaystopNone \
  --start 2020-01 --complete-through 2026-07 --holdout-start 2024-01
```

`research_only` means the monthly metrics passed but trade-level fee and slippage evidence is still missing. It must not promote the live strategy.
