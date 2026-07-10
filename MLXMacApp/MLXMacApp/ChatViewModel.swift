import Foundation

@MainActor
final class ChatViewModel: ObservableObject {
    enum ConnectionState: Equatable {
        case checking
        case connected
        case disconnected(String)
    }

    @Published var baseURL = "http://127.0.0.1:8080/v1"
    @Published var models: [String] = []
    @Published var selectedModel = "Qwen/Qwen3-4B-MLX-4bit"
    @Published var messages: [ChatMessage] = []
    @Published var input = ""
    @Published var isSending = false
    @Published var connectionState: ConnectionState = .checking

    var canSend: Bool {
        !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !selectedModel.isEmpty
            && !isSending
    }

    func checkConnection() async {
        connectionState = .checking
        do {
            let availableModels = try await MLXService(baseURL: baseURL).fetchModels()
            models = availableModels
            if !availableModels.contains(selectedModel), let first = availableModels.first {
                selectedModel = first
            }
            connectionState = .connected
        } catch {
            connectionState = .disconnected(error.localizedDescription)
        }
    }

    func send() async {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isSending else { return }

        input = ""
        messages.append(ChatMessage(role: .user, content: text))
        isSending = true
        defer { isSending = false }

        do {
            let reply = try await MLXService(baseURL: baseURL)
                .complete(messages: messages, model: selectedModel)
            messages.append(ChatMessage(role: .assistant, content: reply))
            connectionState = .connected
        } catch {
            messages.append(ChatMessage(role: .assistant, content: "⚠️ \(error.localizedDescription)"))
            connectionState = .disconnected(error.localizedDescription)
        }
    }

    func clear() {
        messages.removeAll()
    }
}
