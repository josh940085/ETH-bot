# ETH-bot n8n

This is a local-only, self-hosted Community installation. It does not use n8n Cloud,
OpenAI, or any paid AI service.

## Runtime

- Editor: `http://127.0.0.1:5678`
- Health: `http://127.0.0.1:5678/healthz`
- Workflow: `workflows/eth-bot-notifications.json`
- Persistent state: `.runtime/n8n-home`
- Installed package: `.runtime/n8n-package`
- Service owner: Supervisor program `n8n`

Only `TELEGRAM_TOKEN`, `DISCORD_WEBHOOK`, and `DISCORD_NEWS` are exposed to the
n8n process. Binance credentials are intentionally excluded. The webhook is bound
to localhost and ETH-bot falls back to its original direct notification path if n8n
is unavailable.

## Install or refresh the pinned version

```sh
brew install node@22
PATH=/opt/homebrew/opt/node@22/bin:/usr/bin:/bin \
  /opt/homebrew/opt/node@22/bin/npm install \
  --prefix .runtime/n8n-package n8n@2.31.5
```

## Import and publish the workflow

```sh
./.venv/bin/python n8n_service.py import:workflow \
  --input=n8n/workflows/eth-bot-notifications.json
./.venv/bin/python n8n_service.py publish:workflow --id=ethBotNotifications
./.venv/bin/supervisorctl -c supervisord.conf restart n8n
```
