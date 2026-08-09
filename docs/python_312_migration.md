# Python 3.12 Runtime

The live service and every isolated research environment require CPython 3.12.x.
`runtime_python.py` rejects other minor versions before a service starts.

## Current Layout

- `.venv` points to `.venv-py312` and is used by Supervisor and maintenance.
- `.venv-py311-backup` preserves the previous environment for emergency rollback.
- `.python-version` records the validated interpreter version, `3.12.13`.
- Research packages remain under `.runtime/research_envs` and are never live dependencies.

## Rollback

Rollback is an emergency operation and requires all Supervisor programs to be stopped first:

```sh
./.venv/bin/supervisorctl -c supervisord.conf shutdown
rm .venv
mv .venv-py311-backup .venv
./.venv/bin/supervisord -c supervisord.conf
```

After rollback, verify fresh service logs, Binance Mark Price validation, and the exchange position before treating the stack as healthy.

## Upgrade Schedule

Daily maintenance remains a health and safety check. Compatible Python dependency,
Homebrew, and same-major n8n upgrades run only during the Saturday maintenance
window, defaulting to 04:30 Asia/Taipei. Tests and compilation run before updated
services restart; restarts wait while a live Binance position is active.
