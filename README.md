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

### 啟動 server

```bash
pip install -r requirements-realtime.txt
python3 panel_realtime_server.py
```

### Bot 環境變數

```bash
POSITION_PANEL_REALTIME_BASE_URL=https://your-public-domain
POSITION_PANEL_REALTIME_TOKEN=change-me
DISCORD_WEBHOOK=https://discord.com/api/webhooks/xxx/yyy
DISCORD_NEWS=https://discord.com/api/webhooks/xxx/yyy
DISCORD_AUTO_DELETE_HOURS=24
```

`DISCORD_AUTO_DELETE_HOURS` 預設為 `24`，設為 `0` 可停用自動刪除。

### Telegram Mini App

`eth.py` 會自動把 `state_url` / `ws_url` 塞進 Mini App URL。
前端會優先用 Realtime API / WebSocket，失敗時才退回 `position.json` 輪詢。

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

## Docker

已附上 `Dockerfile`，預設啟動 bot worker：

```bash
docker build -t eth-bot .
docker run --env-file .env eth-bot
```

如果要跑 panel 服務，覆寫啟動命令即可：

```bash
docker run -p 8787:8787 --env-file .env eth-bot python panel_realtime_server.py
```

## 建議環境變數

### 共同

```bash
BOT_DATA_DIR=/data
BOT_AI_DATA_DIR=/data/ai
```

`BOT_DATA_DIR` 會接手這些原本偏本地化的檔案：

- `.telegram_state.json`
- `pending_training_sample.json`
- `docs/position.json`
- `news_model*.json/pkl`
- `news_predictions.jsonl`
- `learning_buffer.pkl`
- `sl_followup_reviews.json`

`BOT_AI_DATA_DIR` 會接手原本寫死在 `/Volumes/SSD/trading` 的 AI 檔案：

- `model.pkl`
- `ai_data.csv`
- `online_model.pkl`
- `online_scaler.pkl`
- `online_model_meta.json`
- `ai_learning_meta.json`

如果你沒設 `BOT_AI_DATA_DIR`，程式會優先沿用舊的 `/Volumes/SSD/trading`；找不到時才退回 `BOT_DATA_DIR`。這樣本地既有資料不會被直接打斷。

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
POSITION_PANEL_ALLOWED_ORIGINS=https://your-mini-app-domain
POSITION_PANEL_REALTIME_HOST=0.0.0.0
POSITION_PANEL_REALTIME_PORT=8787
```

## requirements

完整依賴現在放在 [requirements.txt](/Users/ju-kuangchang/ETH-bot/requirements.txt)。

如果只想單獨跑面板 API，也可以只裝 [requirements-realtime.txt](/Users/ju-kuangchang/ETH-bot/requirements-realtime.txt)。

## 前端面板

`docs/index.html` 會優先連線 Realtime API / WebSocket；失敗時才退回 `position.json` 輪詢。

因此雲端部署後建議使用：

- 靜態頁面：GitHub Pages 或任意靜態主機
- 即時資料：`panel_realtime_server.py`
- 交易邏輯：`program.py`
