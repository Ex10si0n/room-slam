//
//  ContentView.swift
//  SLAM World Sender
//
//  Created by Ex10si0n Yan on 10/9/25.
//

import SwiftUI
import RealityKit
import ARKit
import Network
import simd

// MARK: - UDP Sender using NWConnection
final class UdpSender {
    private var connection: NWConnection?
    private let queue = DispatchQueue(label: "udp.sender.queue")

    init(host: String, port: UInt16) {
        reconnect(host: host, port: port)
    }

    deinit {
        self.connection?.cancel()
    }

    func reconnect(host: String, port: UInt16) {
        connection?.cancel()
        let params = NWParameters.udp
        self.connection = NWConnection(host: NWEndpoint.Host(host), port: NWEndpoint.Port(rawValue: port)!, using: params)
        self.connection?.stateUpdateHandler = { state in
            // Debug print if needed
        }
        self.connection?.start(queue: queue)
    }

    func send(_ text: String) {
        let data = text.data(using: .utf8) ?? Data()
        connection?.send(content: data, completion: .contentProcessed { _ in })
    }
}

// MARK: - AR View container for SwiftUI
struct ARViewContainer: UIViewRepresentable {
    let sender: UdpSender
    let targetHz: Double
    let resetTrigger: Bool

    func makeCoordinator() -> Coordinator {
        Coordinator(sender: sender, targetHz: targetHz)
    }

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        context.coordinator.arView = arView
        
        // Configure ARKit session (back camera, world tracking)
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.environmentTexturing = .none
        config.planeDetection = []
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])

        // Delegate to receive AR frame updates
        arView.session.delegate = context.coordinator

        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        // Update target Hz
        context.coordinator.updateTargetHz(targetHz)
        
        // Reset AR session if triggered
        if resetTrigger != context.coordinator.lastResetState {
            context.coordinator.lastResetState = resetTrigger
            context.coordinator.resetARSession()
        }
    }
}

// MARK: - ARSession Delegate Coordinator
final class Coordinator: NSObject, ARSessionDelegate {
    private let sender: UdpSender
    private var targetInterval: Double
    private var lastSentTime: CFTimeInterval = 0
    var lastResetState: Bool = false
    weak var arView: ARView?

    init(sender: UdpSender, targetHz: Double) {
        self.sender = sender
        self.targetInterval = 1.0 / max(1.0, targetHz)
    }

    func updateTargetHz(_ targetHz: Double) {
        self.targetInterval = 1.0 / max(1.0, targetHz)
    }
    
    func resetARSession() {
        guard let arView = arView else { return }
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.environmentTexturing = .none
        config.planeDetection = []
        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Control sending frequency
        let now = CACurrentMediaTime()
        if now - lastSentTime < targetInterval { return }
        lastSentTime = now

        // Extract camera transform
        let transform: simd_float4x4 = frame.camera.transform

        // Position in AR world space (meters)
        let position = SIMD3<Float>(transform.columns.3.x,
                                    transform.columns.3.y,
                                    transform.columns.3.z)

        // Rotation as quaternion
        let quat = simd_quatf(transform)

        // Build JSON string
        let json = String(format:
            #"{"x":%.6f,"y":%.6f,"z":%.6f,"qx":%.6f,"qy":%.6f,"qz":%.6f,"qw":%.6f}"#,
            position.x, position.y, position.z,
            quat.imag.x, quat.imag.y, quat.imag.z, quat.real
        )

        sender.send(json)
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("ARSession error: \(error.localizedDescription)")
    }
    func sessionWasInterrupted(_ session: ARSession) {}
    func sessionInterruptionEnded(_ session: ARSession) {}
}

// MARK: - Settings View
struct SettingsView: View {
    @AppStorage("targetHost") private var targetHost = "192.168.0.53"
    @AppStorage("targetPort") private var targetPort = 4399
    @AppStorage("targetHz") private var targetHz = 30.0
    
    var onSettingsChanged: () -> Void
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Network Configuration")) {
                    TextField("IP Address", text: $targetHost)
                        .keyboardType(.decimalPad)
                        .autocapitalization(.none)
                    
                    HStack {
                        Text("Port")
                        TextField("Port", value: $targetPort, format: .number)
                            .keyboardType(.numberPad)
                            .multilineTextAlignment(.trailing)
                    }
                }
                
                Section(header: Text("Frequency")) {
                    HStack {
                        Text("Target Hz")
                        Spacer()
                        Text("\(Int(targetHz)) Hz")
                            .foregroundColor(.secondary)
                    }
                    Slider(value: $targetHz, in: 1...60, step: 1)
                }
                
                Section {
                    Button("Apply Settings") {
                        onSettingsChanged()
                    }
                    .frame(maxWidth: .infinity)
                }
            }
            .navigationTitle("Settings")
        }
    }
}

// MARK: - AR View with HUD
struct ARCameraView: View {
    let sender: UdpSender
    @AppStorage("targetHost") private var targetHost = "192.168.0.53"
    @AppStorage("targetPort") private var targetPort = 4399
    @AppStorage("targetHz") private var targetHz = 30.0
    @State private var resetTrigger = false
    
    var body: some View {
        ZStack {
            ARViewContainer(sender: sender, targetHz: targetHz, resetTrigger: resetTrigger)
                .ignoresSafeArea()
            
            VStack {
                // Top HUD
                HStack {
                    Text("Sending to \(targetHost):\(targetPort) @ \(Int(targetHz))Hz")
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .padding(8)
                        .background(.thinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    
                    Spacer()
                }
                .padding()
                
                Spacer()
                
                // Bottom reset button
                Button(action: {
                    resetTrigger.toggle()
                }) {
                    Label("Reset Position", systemImage: "arrow.counterclockwise")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 12)
                        .background(.ultraThinMaterial)
                        .clipShape(Capsule())
                }
                .padding(.bottom, 40)
            }
        }
    }
}

// MARK: - SwiftUI App Entry
struct ContentView: View {
    @AppStorage("targetHost") private var targetHost = "192.168.0.53"
    @AppStorage("targetPort") private var targetPort = 4399
    @State private var sender: UdpSender?
    
    init() {
        let host = UserDefaults.standard.string(forKey: "targetHost") ?? "192.168.0.53"
        let port = UserDefaults.standard.integer(forKey: "targetPort")
        _sender = State(initialValue: UdpSender(host: host, port: UInt16(port == 0 ? 4399 : port)))
    }
    
    var body: some View {
        TabView {
            if let sender = sender {
                ARCameraView(sender: sender)
                    .tabItem {
                        Label("AR Camera", systemImage: "camera.fill")
                    }
            }
            
            SettingsView {
                // Reconnect with new settings
                sender?.reconnect(host: targetHost, port: UInt16(targetPort))
            }
            .tabItem {
                Label("Settings", systemImage: "gear")
            }
        }
    }
}

#Preview {
    ContentView()
}
