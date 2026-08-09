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
- `binance_trade_history.py` downloads official USD-M `trades` or `aggTrades` archives and rejects files that fail Binance's published SHA256 checksum or timestamp/schema validation.
- `monthly5_execution_replay.py` converts timestamped strategy signals into auditable fills, fees, and slippage using official Binance ticks. Candidate identity and month coverage must match before the batch validator accepts the evidence.
- `monthly5_bias_audit.py` applies a Freqtrade-style promotion gate for negative shifts, prefix replay stability, and recursive indicator stability.
- `cryptofeed_binance_collector.py` records new Binance Futures trades and L2 updates for future queue/slippage replay.
- `hftbacktest_data_probe.py` prevents an hftbacktest run until both trade and L2 records are present.
- `research_tool_status.py` verifies all isolated research environments and records that Tardis is intentionally excluded.
- `verify_monthly5_runtime.py` remains the runtime gate for the live service.
- Direct takeover must not bypass performance blockers such as:
  - `shadow_monthly_target`
  - `shadow_rolling_monthly_target`
  - `active_underperforming_plan`
  - `recovery_probe_probe_failed`

## Isolated Environments

- `.runtime/research_envs/freqtrade`: Python 3.12, `freqtrade==2026.7`.
- `.runtime/research_envs/hftbacktest`: Python 3.12, `hftbacktest==2.4.4`.
- `.runtime/research_envs/cryptofeed`: Python 3.12, `cryptofeed==2.4.1`.
- `.runtime/research_envs/nautilus`: Python 3.12, `nautilus_trader==1.227.0`.
- Tardis is not installed.

These environments are research-only. They are absent from `requirements.txt` and are never imported by `eth.py`.

## Deferred

- Do not add large backtesting frameworks to the live service process. The current NumPy/pandas candidate matrices already provide the needed batch operations without adding `vectorbt` to production dependencies.
- Do not copy public GitHub trading strategies into production.
- Add `vectorbt` only if portfolio-order simulation is needed beyond the existing matrix engine. Evaluate `hftbacktest` separately when tick/order-book archives are available.

Check the installed research stack:

```sh
./.venv/bin/python research_tool_status.py
```

Run Freqtrade's upstream bias-analysis commands from its isolated environment:

```sh
.runtime/research_envs/freqtrade/bin/freqtrade lookahead-analysis --help
.runtime/research_envs/freqtrade/bin/freqtrade recursive-analysis --help
```

Capture five minutes of new Binance Futures microstructure data:

```sh
.runtime/research_envs/cryptofeed/bin/python cryptofeed_binance_collector.py \
  --symbol BTC-USDT-PERP --duration-sec 300
```

Probe the capture before hftbacktest conversion:

```sh
.runtime/research_envs/hftbacktest/bin/python hftbacktest_data_probe.py \
  --input .runtime/data/market_microstructure/binance_futures_btcusdt.jsonl
```

Run the current validator from the repository root:

```sh
./.venv/bin/python monthly5_batch_validator.py \
  --candidate postlock_scale0.15_floor_pdaystopNone \
  --start 2020-01 --complete-through 2026-07 --holdout-start 2024-01
```

`research_only` means the monthly metrics passed but trade-level fee and slippage evidence is still missing. It must not promote the live strategy.
