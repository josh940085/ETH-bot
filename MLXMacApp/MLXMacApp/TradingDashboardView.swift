import SwiftUI
import WebKit

struct TradingDashboardView: View {
    @State private var reloadID = UUID()

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("ETH 交易面板")
                        .font(.title2.bold())
                    Text("Mini App · 本機即時資料")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    reloadID = UUID()
                } label: {
                    Label("重新整理", systemImage: "arrow.clockwise")
                }
            }
            .padding(.horizontal, 22)
            .padding(.vertical, 14)

            Divider()

            MiniAppWebView(url: URL(string: "http://127.0.0.1:8787/")!)
                .id(reloadID)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

private struct MiniAppWebView: NSViewRepresentable {
    let url: URL

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = true

        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        webView.setValue(false, forKey: "drawsBackground")
        webView.load(URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData))
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        guard webView.url == nil else { return }
        webView.load(URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData))
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
        ) {
            guard let destination = navigationAction.request.url else {
                decisionHandler(.cancel)
                return
            }

            let isLocalPanel = destination.host == "127.0.0.1" || destination.host == "localhost"
            if navigationAction.navigationType == .linkActivated, !isLocalPanel {
                NSWorkspace.shared.open(destination)
                decisionHandler(.cancel)
                return
            }
            decisionHandler(.allow)
        }
    }
}
