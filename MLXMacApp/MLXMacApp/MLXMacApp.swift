import SwiftUI

@main
struct MLXMacApp: App {
    @StateObject private var runtime = ETHBotRuntime()

    var body: some Scene {
        WindowGroup {
            ContentView(runtime: runtime)
                .frame(minWidth: 760, minHeight: 560)
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 960, height: 700)
    }
}

@MainActor
final class ETHBotRuntime: ObservableObject {
    enum State: Equatable {
        case starting(String)
        case ready
        case failed(String)
    }

    @Published private(set) var state: State = .starting("準備啟動 ETH-bot…")

    private let repositoryURL = URL(fileURLWithPath: "/Volumes/SSD/ETH-bot", isDirectory: true)
    private var started = false

    var isReady: Bool { state == .ready }

    func startIfNeeded() async {
        guard !started else { return }
        started = true

        state = .starting("檢查監控程式…")
        let firstStatus = await run(".venv/bin/supervisorctl", ["-c", "supervisord.conf", "status"])

        if firstStatus.exitCode != 0 {
            state = .starting("啟動 ETH-bot 程式…")
            let launch = await run(".venv/bin/supervisord", ["-c", "supervisord.conf"])
            guard launch.exitCode == 0 else {
                state = .failed(cleanMessage(launch.output, fallback: "無法啟動 supervisord"))
                return
            }

            for _ in 0..<20 {
                try? await Task.sleep(for: .milliseconds(250))
                let status = await run(".venv/bin/supervisorctl", ["-c", "supervisord.conf", "status"])
                if status.exitCode == 0 { break }
            }
        }

        state = .starting("啟動交易與 UI 服務…")
        _ = await run(".venv/bin/supervisorctl", ["-c", "supervisord.conf", "start", "all"])

        for _ in 0..<40 {
            let status = await run(".venv/bin/supervisorctl", ["-c", "supervisord.conf", "status"])
            if status.exitCode == 0,
               ["eth-bot", "mlx-agent", "panel-realtime", "panel-tunnel"].allSatisfy({ service in
                   status.output.contains(service) && status.output
                       .split(separator: "\n")
                       .contains(where: { $0.contains(service) && $0.contains("RUNNING") })
               }) {
                state = .ready
                return
            }
            try? await Task.sleep(for: .milliseconds(500))
        }

        let finalStatus = await run(".venv/bin/supervisorctl", ["-c", "supervisord.conf", "status"])
        state = .failed(cleanMessage(finalStatus.output, fallback: "ETH-bot 服務未能全部啟動"))
    }

    func retry() async {
        started = false
        await startIfNeeded()
    }

    private func run(_ relativeExecutable: String, _ arguments: [String]) async -> (exitCode: Int32, output: String) {
        await Task.detached(priority: .userInitiated) { [repositoryURL] in
            let process = Process()
            let pipe = Pipe()
            process.currentDirectoryURL = repositoryURL
            process.executableURL = repositoryURL.appendingPathComponent(relativeExecutable)
            process.arguments = arguments
            process.standardOutput = pipe
            process.standardError = pipe

            do {
                try process.run()
                process.waitUntilExit()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                return (process.terminationStatus, String(decoding: data, as: UTF8.self))
            } catch {
                return (-1, error.localizedDescription)
            }
        }.value
    }

    private func cleanMessage(_ value: String, fallback: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? fallback : trimmed
    }
}
