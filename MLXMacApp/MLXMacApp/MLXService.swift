import Foundation

enum MLXServiceError: LocalizedError {
    case invalidURL
    case invalidResponse
    case server(status: Int, message: String)
    case emptyResponse

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "MLX API 位址無效"
        case .invalidResponse:
            return "MLX 回傳了無法辨識的回應"
        case let .server(status, message):
            return "MLX 連線失敗（HTTP \(status)）：\(message)"
        case .emptyResponse:
            return "MLX 沒有回傳內容"
        }
    }
}

struct MLXService {
    var baseURL: String

    private var normalizedBaseURL: String {
        baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    func fetchModels() async throws -> [String] {
        let data = try await request(path: "models", method: "GET", body: nil)
        let response = try JSONDecoder().decode(ModelListResponse.self, from: data)
        return response.data.map(\.id)
    }

    func complete(messages: [ChatMessage], model: String) async throws -> String {
        let payload = ChatCompletionRequest(
            model: model,
            messages: messages.map { .init(role: $0.role.rawValue, content: $0.content) },
            temperature: 0.7,
            max_tokens: 1200
        )
        let body = try JSONEncoder().encode(payload)
        let data = try await request(path: "chat/completions", method: "POST", body: body)
        let response = try JSONDecoder().decode(ChatCompletionResponse.self, from: data)
        guard let content = response.choices.first?.message.content,
              !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MLXServiceError.emptyResponse
        }
        return content
    }

    private func request(path: String, method: String, body: Data?) async throws -> Data {
        guard let url = URL(string: "\(normalizedBaseURL)/\(path)") else {
            throw MLXServiceError.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = 120
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw MLXServiceError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            let message = String(data: data, encoding: .utf8) ?? "未知錯誤"
            throw MLXServiceError.server(status: http.statusCode, message: message)
        }
        return data
    }
}
