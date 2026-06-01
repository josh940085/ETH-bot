# ETH-bot

這個 repo 現在可以直接拆成兩個雲端服務部署，不需要再只靠本地長駐：

- `bot worker`：執行 `program.py`，負責 Telegram、策略判斷、下單、同步面板資料
- `panel web service`：執行 `panel_realtime_server.py`，提供 Mini App 的 REST / WebSocket 即時資料

## Panel Realtime

Mini App 如果要在 Telegram 手機端即時更新，不能只靠 `docs/position.json`。
需要把最新倉位推到一個公開可連線的 API / WebSocket 後端。

本 repo 已附上 `panel_realtime_server.py`，提供：

- `POST /api/panel/publish`：bot 推送最新倉位
- `GET /api/panel/state`：Mini App 取得最新快照
- `WS /ws/panel`：Mini App 即時接收更新

這版預設會用 Telegram WebApp `initData` 驗證讀取端。
也就是說：

- Bot 推送仍用 `POSITION_PANEL_REALTIME_TOKEN`
- Mini App 讀取則要求合法 Telegram 身分
- 可再用 `POSITION_PANEL_ALLOWED_TELEGRAM_USER_IDS` 限制只允許特定帳號

### 啟動 server

```bash
pip install -r requirements-realtime.txt
python3 panel_realtime_server.py
```

### Bot 環境變數

```bash
POSITION_PANEL_REALTIME_BASE_URL=https://your-public-domain
POSITION_PANEL_REALTIME_TOKEN=change-me
POSITION_PANEL_ALLOWED_TELEGRAM_USER_IDS=123456789
DISCORD_WEBHOOK=https://discord.com/api/webhooks/xxx/yyy
DISCORD_NEWS=https://discord.com/api/webhooks/xxx/yyy
DISCORD_AUTO_DELETE_HOURS=24
```

`DISCORD_AUTO_DELETE_HOURS` 預設為 `24`，設為 `0` 可停用自動刪除。

### Telegram Mini App

`eth.py` 會自動把 `state_url` / `ws_url` 塞進 Mini App URL。
前端會優先用 Realtime API / WebSocket。
公開部署時不再回退到 `position.json`，避免敏感倉位被 GitHub Pages 直接公開。

## 雲端部署建議

### 服務 1：Bot Worker

啟動命令：

```bash
python program.py
```

如果你要在本機長駐執行，也可以直接用 repo 內附的 `supervisord.conf`：

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/supervisord -c supervisord.conf
.venv/bin/supervisorctl -c supervisord.conf status
```

常用指令：

```bash
.venv/bin/supervisorctl -c supervisord.conf restart eth-bot
.venv/bin/supervisorctl -c supervisord.conf stop eth-bot
.venv/bin/supervisorctl -c supervisord.conf start eth-bot
.venv/bin/supervisorctl -c supervisord.conf shutdown
```

必要條件：

- 只開 `1` 個實例，否則 Telegram `getUpdates` 會互相衝突
- 建議掛一個持久化磁碟，並設定 `BOT_DATA_DIR`
- 若你要把 AI 訓練資料和模型放到獨立磁碟，再加設 `BOT_AI_DATA_DIR`

### 服務 2：Panel Web Service

啟動命令：

```bash
python panel_realtime_server.py
```

這個服務現在支援雲端平台常見的 `PORT` 環境變數；如果沒給 `PORT`，才會退回 `POSITION_PANEL_REALTIME_PORT` 或預設 `8787`。

## SSD 自包含模式

如果你要把整套程式和資料都收斂在同一顆 SSD，建議直接用 repo 內建的 `.runtime/`：

```bash
BOT_DATA_DIR=.runtime/data
BOT_AI_DATA_DIR=.runtime/ai
```

這樣程式碼、模型、快照、log 都會跟著 repo 走，不再依賴 `/Users/...` 或 `/Volumes/...` 這類機器路徑。

## Docker

已附上 `Dockerfile` 和 [compose.yaml](compose.yaml)。
如果另一台機器有 Docker，直接在 SSD 內的 repo 執行：

```bash
docker compose up -d --build
```

這會啟動：

- `eth-bot`
- `panel-realtime`

兩個服務共用 repo 內的 `.runtime/`，所以資料仍然留在 SSD。

如果你繼續用 `.venv` + `supervisord` 啟動，仍然會依賴宿主機上的 Python。
要把 SSD 插到另一台機器直接跑，應改用 `docker compose`，或把整顆 SSD 做成可開機的 Linux 環境。

## 建議環境變數

### 共同

```bash
BOT_DATA_DIR=.runtime/data
BOT_AI_DATA_DIR=.runtime/ai
```

`BOT_DATA_DIR` 會接手這些原本偏本地化的檔案：

- `.telegram_state.json`
- `pending_training_sample.json`
- `docs/position.json`
- `news_model*.json/pkl`
- `news_predictions.jsonl`
- `learning_buffer.pkl`
- `sl_followup_reviews.json`

`BOT_AI_DATA_DIR` 會接手 AI 訓練資料與模型檔：

- `model.pkl`
- `ai_data.csv`
- `online_model.pkl`
- `online_scaler.pkl`
- `online_model_meta.json`
- `ai_learning_meta.json`

如果你沒設 `BOT_AI_DATA_DIR`，程式預設會使用 repo 內的 `.runtime/ai`。

### Bot Worker

```bash
TELEGRAM_TOKEN=...
TELEGRAM_CHAT_ID=...
DISCORD_WEBHOOK=...
DISCORD_NEWS=...
DISCORD_AUTO_DELETE_HOURS=24
POSITION_PANEL_REALTIME_BASE_URL=https://your-panel-domain
POSITION_PANEL_REALTIME_TOKEN=change-me
```

### Panel Web Service

```bash
POSITION_PANEL_REALTIME_TOKEN=change-me
TELEGRAM_TOKEN=123456:ABCDEF
POSITION_PANEL_ALLOWED_TELEGRAM_USER_IDS=123456789
POSITION_PANEL_ALLOWED_ORIGINS=https://your-mini-app-domain
POSITION_PANEL_REALTIME_HOST=0.0.0.0
POSITION_PANEL_REALTIME_PORT=8787
```

## requirements

完整依賴現在放在 [requirements.txt](requirements.txt)。

如果只想單獨跑面板 API，也可以只裝 [requirements-realtime.txt](requirements-realtime.txt)。

## 前端面板

`docs/index.html` 會優先連線 Realtime API / WebSocket。
如果不是從 Telegram Mini App 打開，或沒有通過伺服器端驗證，頁面只會顯示鎖定提示。

因此雲端部署後建議使用：

- 靜態頁面：GitHub Pages 或任意靜態主機
- 即時資料：`panel_realtime_server.py`
- 交易邏輯：`program.py`
