# MLX Mac App

原生 SwiftUI macOS App，連接 ETH-bot 在 `http://127.0.0.1:8080/v1` 提供的 MLX OpenAI-compatible API。
App 也會內嵌 `http://127.0.0.1:8787/` 的 Telegram Mini App 交易面板，直接顯示本機即時倉位資料。

## 使用方式

1. 確認 `mlx-agent` 與 `panel-realtime` 正在執行。
2. 用 Xcode 開啟 `MLXMacApp.xcodeproj`。
3. 選擇 `My Mac`，按 Run。

App 啟動時會讀取 `/v1/models`，並用 `/v1/chat/completions` 傳送對話。
