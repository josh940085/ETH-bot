# MLX Mac App

原生 SwiftUI macOS App，連接 ETH-bot 在 `http://127.0.0.1:8080/v1` 提供的 MLX OpenAI-compatible API。
App 也會內嵌 `http://127.0.0.1:8787/` 的 Telegram Mini App 交易面板，直接顯示本機即時倉位資料。

## 使用方式

1. 用 Xcode 開啟 `MLXMacApp.xcodeproj`。
2. 選擇 `My Mac`，按 Run。

App 啟動時會從 `/Volumes/SSD/ETH-bot` 自動啟動 supervisord，並確保 `eth-bot`、`mlx-agent`、`panel-realtime` 與 `panel-tunnel` 全部運行。服務就緒後，App 會載入交易面板、讀取 `/v1/models`，並用 `/v1/chat/completions` 傳送對話。
第一次啟動時，macOS 會詢問是否允許存取外接 SSD；按一次「允許」後即會記住設定。

這是本機專用 App，需要啟動專案內的監控程式，因此不使用 macOS App Sandbox。
