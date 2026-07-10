import SwiftUI

@main
struct MLXMacApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 760, minHeight: 560)
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 960, height: 700)
    }
}
