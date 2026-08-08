# GitHub Resource Adoption Notes

This repo should use external trading projects as engineering references, not as strategy sources copied into live trading.

## Adopted First

- `freqtrade/freqtrade`: use the protection model: live/dry-run evidence, explicit blockers, and no leverage promotion while the active profile is underperforming.
- `freqtrade` leverage guidance: keep leverage capped at 5x and require stop/TP protection to remain Binance-confirmed.
- `vectorbt`: useful later for fast monthly5 parameter sweeps, but it is not on the live path.
- `hftbacktest`: useful later for tick/order-book replay when validating short samples such as 8h or 12h windows.

## Local Implementation

- `monthly5_risk_audit.py` audits recent Binance fills, monthly5 shadow performance, and promotion blockers.
- `verify_monthly5_runtime.py` remains the runtime gate for the live service.
- Direct takeover must not bypass performance blockers such as:
  - `shadow_monthly_target`
  - `shadow_rolling_monthly_target`
  - `active_underperforming_plan`
  - `recovery_probe_probe_failed`

## Deferred

- Do not add large backtesting frameworks to the live service process.
- Do not copy public GitHub trading strategies into production.
- Evaluate `vectorbt` or `hftbacktest` only in separate research scripts after dependency compatibility is checked.
