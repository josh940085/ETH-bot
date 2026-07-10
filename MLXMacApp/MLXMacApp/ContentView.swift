import SwiftUI

struct ContentView: View {
    private enum AppSection: String, CaseIterable, Identifiable {
        case dashboard = "交易面板"
        case chat = "MLX 對話"

        var id: Self { self }

        var icon: String {
            switch self {
            case .dashboard: return "chart.xyaxis.line"
            case .chat: return "bubble.left.and.bubble.right"
            }
        }
    }

    @StateObject private var viewModel = ChatViewModel()
    @State private var selectedSection: AppSection = .dashboard
    @FocusState private var inputFocused: Bool

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 230, ideal: 270, max: 330)
        } detail: {
            switch selectedSection {
            case .dashboard:
                TradingDashboardView()
            case .chat:
                VStack(spacing: 0) {
                    header
                    Divider()
                    conversation
                    Divider()
                    composer
                }
                .background(Color(nsColor: .windowBackgroundColor))
            }
        }
        .task {
            await viewModel.checkConnection()
        }
    }

    private var sidebar: some View {
        Form {
            Section("功能") {
                ForEach(AppSection.allCases) { section in
                    Button {
                        selectedSection = section
                    } label: {
                        HStack {
                            Label(section.rawValue, systemImage: section.icon)
                            Spacer()
                            if selectedSection == section {
                                Image(systemName: "checkmark")
                                    .foregroundStyle(.tint)
                            }
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }

            Section("MLX 連線") {
                TextField("API 位址", text: $viewModel.baseURL)
                    .textFieldStyle(.roundedBorder)

                Button {
                    Task { await viewModel.checkConnection() }
                } label: {
                    Label("重新連線", systemImage: "arrow.clockwise")
                }
                .disabled(viewModel.connectionState == .checking)
            }

            Section("模型") {
                if viewModel.models.isEmpty {
                    TextField("模型名稱", text: $viewModel.selectedModel)
                        .textFieldStyle(.roundedBorder)
                } else {
                    Picker("模型", selection: $viewModel.selectedModel) {
                        ForEach(viewModel.models, id: \.self) { model in
                            Text(model).tag(model)
                        }
                    }
                    .labelsHidden()
                }
            }

            Section("狀態") {
                HStack(spacing: 8) {
                    Circle()
                        .fill(statusColor)
                        .frame(width: 9, height: 9)
                    Text(statusTitle)
                }
                if case let .disconnected(message) = viewModel.connectionState {
                    Text(message)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }

            Spacer()

            Button(role: .destructive) {
                viewModel.clear()
            } label: {
                Label("清除對話", systemImage: "trash")
            }
            .disabled(viewModel.messages.isEmpty)
        }
        .formStyle(.grouped)
        .navigationTitle("MLX 控制台")
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text("本機 MLX 對話")
                    .font(.title2.bold())
                Text(viewModel.selectedModel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer()
            if viewModel.isSending {
                ProgressView()
                    .controlSize(.small)
                Text("MLX 思考中…")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 22)
        .padding(.vertical, 14)
    }

    private var conversation: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 14) {
                    if viewModel.messages.isEmpty {
                        ContentUnavailableView(
                            "開始與本機 MLX 對話",
                            systemImage: "apple.intelligence",
                            description: Text("輸入訊息後按 Return 送出；Shift + Return 可換行。")
                        )
                        .frame(minHeight: 360)
                    }
                    ForEach(viewModel.messages) { message in
                        MessageBubble(message: message)
                            .id(message.id)
                    }
                }
                .padding(22)
            }
            .onChange(of: viewModel.messages) { _, messages in
                guard let last = messages.last else { return }
                withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
            }
        }
    }

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 12) {
            TextField("輸入給 MLX 的訊息…", text: $viewModel.input, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...7)
                .padding(11)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
                .focused($inputFocused)
                .onSubmit {
                    guard viewModel.canSend else { return }
                    Task { await viewModel.send() }
                }

            Button {
                Task { await viewModel.send() }
            } label: {
                Image(systemName: "arrow.up")
                    .font(.headline.bold())
                    .frame(width: 36, height: 36)
            }
            .buttonStyle(.borderedProminent)
            .buttonBorderShape(.circle)
            .disabled(!viewModel.canSend)
            .keyboardShortcut(.return, modifiers: .command)
        }
        .padding(16)
    }

    private var statusColor: Color {
        switch viewModel.connectionState {
        case .checking: return .orange
        case .connected: return .green
        case .disconnected: return .red
        }
    }

    private var statusTitle: String {
        switch viewModel.connectionState {
        case .checking: return "檢查中"
        case .connected: return "已連線"
        case .disconnected: return "未連線"
        }
    }
}

private struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 90) }
            Text(message.content)
                .textSelection(.enabled)
                .padding(.horizontal, 14)
                .padding(.vertical, 11)
                .background(background, in: RoundedRectangle(cornerRadius: 15))
                .foregroundStyle(message.role == .user ? Color.white : Color.primary)
            if message.role != .user { Spacer(minLength: 90) }
        }
    }

    private var background: Color {
        message.role == .user ? .accentColor : Color(nsColor: .controlBackgroundColor)
    }
}

#Preview {
    ContentView()
        .frame(width: 960, height: 700)
}
