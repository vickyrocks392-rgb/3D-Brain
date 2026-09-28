# 3D Visualization Concept & Telemetry Data Contract

## 1. Co-Design Principle & Visualization Objective

In **3D-Brain**, the visualization is not an artistic mockup; it is a **high-fidelity spatial oscilloscope and causal debugging console**.

Every pixel rendered in the 3D world corresponds to an actual state variable in the computational substrate.

For **Experiment 001 (Delayed Comparison)**, the visualizer allows the researcher to directly observe:
1. The cue stimulus entering the sensory input cluster.
2. Action signals propagating along physical 3D connections into the recurrent memory population.
3. The maintenance of working memory during the silent delay period: observing whether the population maintains a stable attractor state or disperses under distractor noise.
4. The probe stimulus arriving at time $T_{\text{probe}}$ and triggering a match/mismatch comparison cascade.
5. The final decision readout glowing with positive (match) or negative (mismatch) polarity.

---

## 2. Spatial Scene Architecture for Experiment 001

The network for Experiment 001 is mapped into 3D Euclidean coordinates $(X, Y, Z)$ inside a normalized bounding volume $[-10.0, 10.0]^3$:

```text
       SENSORY INPUT                  RECURRENT MEMORY                     DECISION READOUT
         CLUSTER                         POPULATION                            CLUSTER
      (Z = -8.0)                     (Z = -2.0 to +2.0)                       (Z = +8.0)

    ┌──────────────┐                 ┌─────────────────┐                  ┌──────────────┐
    │ Input Node 1 │ ──────────────> │                 │ ───────────────> │ Output Unit  │
    │  (Cue/Probe) │                 │  N Recurrent    │                  │  (Match /    │
    ├──────────────┤ ──────────────> │  Memory Neurons │ ───────────────> │   Mismatch)  │
    │ Input Node 2 │                 │  (3D Ellipsoid  │                  └──────────────┘
    │  (Gate Flag) │                 │   or Cylinder)  │
    └──────────────┘                 │                 │
                                     └─────────────────┘
                                       ↺ Recurrent
                                         Interconnects
```

### Node Spatial Placement
* **Input Cluster (2 nodes)**: Positioned at $\mathbf{p}_{\text{in1}} = (-2.0, 0.0, -8.0)$, $\mathbf{p}_{\text{in2}} = (2.0, 0.0, -8.0)$.
* **Memory Population ($N$ nodes, e.g. $N=16$ to $64$)**: Arranged in a 3D prolate ellipsoid around the origin $(0, 0, 0)$ using a deterministic Fibonacci sphere or layered cylinder algorithm.
* **Output Readout (1 node)**: Positioned at $\mathbf{p}_{\text{out}} = (0.0, 0.0, +8.0)$.
* **Edges ($E$ connections)**: Rendered as 3D quadratic Bézier splines connecting source and destination nodes.

---

## 3. Visual Primitives & Material Aesthetics

1. **Neuron Mesh**:
   * Geometry: Icosahedron / smooth sphere.
   * Resting State: Translucent dark slate gray with faint core glow ($r=0.25$).
   * Active State: Emissive luminescence proportional to $|a_i(t)|$.
   * Polarity Hue:
     * Positive Activation ($a_i > 0$): Electric Cyan / Blue (`#00E5FF`).
     * Negative Activation ($a_i < 0$): Deep Amber / Coral (`#FF6D00`).
2. **Connections (Synaptic Splines)**:
   * Geometry: Tube or ribbon geometry.
   * Thickness: Proportional to absolute synaptic weight $|w_{ij}|$.
   * Base Opacity: Low opacity ($0.15$) for idle connections; brightens when signals traverse.
   * Sign: Excitatory = cool tint; Inhibitory = warm tint.
3. **Signal Packets in Transit**:
   * Geometry: Luminous spherical particle or glowing comet tail traversing spline path $\mathbf{p}(s) = (1-s)^2 \mathbf{p}_j + 2(1-s)s \mathbf{p}_{\text{mid}} + s^2 \mathbf{p}_i$.
   * Speed: Defined by finite propagation velocity $v_{\text{prop}}$, creating observable waves of propagation.
4. **Prediction Error Burst**:
   * Geometry: Concentric shockwave ring expanding outward from the error node.
   * Color: Intense Magenta (`#E040FB`).

---

## 4. Telemetry Data Contract (JSON & Binary Stream)

To decouple compute from rendering, the computational engine emits structured frame telemetry.

### Frame Packet Schema (`BrainFrameTelemetry`)

```json
{
  "protocol_version": "1.0",
  "step": 42,
  "time_ms": 420.0,
  "nodes": {
    "count": 18,
    "activations": [0.85, 1.0, 0.12, -0.45, 0.0, ...],
    "states": [0.92, 1.0, 0.14, -0.52, 0.0, ...]
  },
  "signals": [
    {
      "src": 0,
      "dst": 4,
      "progress": 0.65,
      "magnitude": 0.85
    }
  ],
  "global_metrics": {
    "output_value": 0.02,
    "target_value": 0.0,
    "prediction_error": 0.02,
    "neuromodulator_level": 0.05
  },
  "events": [
    {
      "type": "CUE_RECEIVED",
      "node_id": 0,
      "payload": {"value": 1.0}
    }
  ]
}
```

### Static Topology Schema (`BrainTopology`)
Emitted once at initialization or on structural reorganization:

```json
{
  "topology_id": "exp001_li_16units",
  "node_count": 19,
  "nodes": [
    {"id": 0, "type": "INPUT", "label": "CueInput", "pos": [-2.0, 0.0, -8.0]},
    {"id": 1, "type": "INPUT", "label": "GateFlag", "pos": [2.0, 0.0, -8.0]},
    {"id": 2, "type": "RECURRENT", "label": "Mem_00", "pos": [-1.2, 0.8, -1.0]},
    ...
    {"id": 18, "type": "OUTPUT", "label": "Decision", "pos": [0.0, 0.0, 8.0]}
  ],
  "edges": [
    {"src": 0, "dst": 2, "weight": 0.452, "latency_steps": 2},
    {"src": 2, "dst": 3, "weight": -0.781, "latency_steps": 1},
    ...
  ]
}
```

---

## 5. Interactive macOS Experience & Timeline Controls

The user interface in the desktop application must provide:
1. **Interactive Orbit Camera**:
   * Pan, zoom, orbit around the 3D neural cluster.
   * Region focus: double-click a population cluster to frame the camera.
2. **Deterministic Timeline Scrubber**:
   * **Play / Pause / Slow-Motion** ($0.1\times, 0.5\times, 1.0\times, 2.0\times$).
   * **Step Forward ($t+1$) & Step Backward ($t-1$)**: Allows inspecting exact pre-synaptic values before an error burst.
   * **Event Marker Bookmarks**: Visual notches along the scrubber indicating `CUE`, `PROBE`, `DECISION`, and high `ERROR` events.
3. **Inspector Drawer (Click-to-Inspect)**:
   * Clicking any neuron displays an inspection HUD:
     * Node ID, Type, 3D Coordinates.
     * Real-time rolling waveform of activation $a_i(t)$ and state $h_i(t)$.
     * Incoming connections list with current weights and live contributions $w_{ij} a_j$.
     * Outgoing connections and downstream targets.

---

## 6. Rendering Technology Evaluation for macOS

To fulfill the vision of a native-feeling macOS application without premature framework lock-in, we evaluate three potential rendering stacks:

| Technology Stack | Performance & FPS | Apple Silicon Integration | Development Velocity | Portability & Decoupling |
| :--- | :--- | :--- | :--- | :--- |
| **Swift + Metal / RealityKit** | Maximum (Native Metal shaders, zero overhead) | Best (Direct unified memory access) | Moderate (macOS specific) | macOS only |
| **Rust + WGPU (Metal backend)** | High (Native binary, compiles to Metal) | High (Direct Metal API mapping) | Moderate | Cross-platform |
| **Modern WebGPU + Desktop Container (Tauri/Electron)** | High (Hardware-accelerated WebGPU, 60+ FPS) | High (Modern WebGPU on Apple Silicon) | **Highest** (Rapid 3D UI, shaders, canvas) | **Highest** (Identical telemetry, zero lock-in) |

### Engineering Decision for Initial Phase:
1. Maintain strict separation via the **Telemetry Data Contract**.
2. The computational engine emits standard telemetry files and streaming web sockets/shared memory.
3. This allows building the initial interactive visualizer using the fastest, highest-fidelity prototype container (e.g. WebGPU / Three.js / Canvas in Tauri or native Webview) while retaining the exact same telemetry stream for a pure native Swift/Metal port when final performance optimization is required.

---

## 7. Phase 9 Neural Observatory Console Architecture

The Phase 9 Remediation transformed the research player into a high-fidelity **Neural Observatory Console** adhering to strict scientific dark instrumentation aesthetics.

```text
┌─────────────────────────────────────────────────────────────┐
│  TOP BAR — Brain Identity / Mode / Time / System Status    │
├───────────────┬───────────────────────────────┬─────────────┤
│               │                               │             │
│  LEFT         │                               │   RIGHT     │
│  INSPECTOR    │         3D BRAIN              │   TELEMETRY │
│               │                               │             │
│  selected     │      live computational       │   metrics   │
│  neuron       │           substrate           │   events    │
│  community    │                               │   learning  │
│  synapse      │                               │             │
│               │                               │             │
├───────────────┴───────────────────────────────┴─────────────┤
│ TIMELINE / REPLAY / EVENT SCRUBBER                          │
└─────────────────────────────────────────────────────────────┘
```

### 7.1 Visual Design System & Palette
- **Void & Surfaces**: Near-black void (`#05070b`), glassmorphic panels (`rgba(11, 15, 23, 0.85)` with blur), thin high-contrast borders (`rgba(255, 255, 255, 0.08)`).
- **Excitatory**: Warm Coral / Orange family (`#ff6b4a`).
- **Inhibitory**: Electric Cyan / Blue family (`#00e5ff`).
- **Reward**: Emerald Green (`#10b981`).
- **Plasticity / Credit**: Amber (`#f59e0b`).
- **Prediction / Neuromodulation**: Violet / Purple (`#a855f7`).
- **Selected Entity**: Bright Gold (`#ffd700`).

### 7.2 Panel Hierarchy & Interaction Model
1. **Top Bar**: Real-time status (`● LIVE` vs `◉ REPLAY`), master simulation time ($t$ in ms), step count, model selector (defaulting to Phase 9 `EXP009_three_factor_rpe_modular.brainlog`), regime indicators (E/I ratio, Dale law, AI regime status, Kuramoto $R$, active plasticity rule), and camera controls.
2. **Left Inspector**:
   - **Empty State**: Displays `NO SELECTION` with interactive community list (`C00`, `C01`, `C02`, `C03`).
   - **Neuron Inspector**: Cell type (`EXCITATORY` vs `INHIBITORY`), module cluster, 3D Euclidean coordinates, instantaneous membrane potential $V_m$ with progress bar to spike threshold ($-50.0\text{ mV}$), firing status (`⚡ ACTIVE SPIKE` vs `Resting`), eligibility trace, and connected synapses list.
   - **Synapse Inspector**: Directional pair $u \to v$, class (`E → E`, `E → I`, `I → E`, `I → I`), Dale constraint sign, 3D Euclidean distance, propagation latency, current weight $W_t$, eligibility $e_{ij}$, and Three-Factor weight update $\Delta w$.
   - **Community Inspector**: Module ID, member count, E/I ratio, mean population activation, and member unit chips.
3. **Right Telemetry & Event Feed Panel**:
   - **Telemetry Tab**: Network size, synchrony metrics (Global $R$, Local $R$, PLV, Pair correlation, CV-ISI, Fano Factor — displaying `NOT RECORDED` when absent), modularity ($Q_{\text{struct}}$, $Q_{\text{func}}$), Three-Factor learning variables ($R(t)$, $V(t)$, $\text{RPE } \delta$, $M(t)$, mean/max eligibility, plastic updates count, $\sum |\Delta w|$), and predictive state ($MSE$, context accuracy).
   - **Events Tab**: Chronological feed of verifiable computational events (`REWARD_RECEIVED`, `RPE_COMPUTED`, `PLASTICITY_APPLIED`, `CREDIT_ASSIGNED`, `INHIBITORY_SPIKE`, `TRIAL_STARTED`). Clicking an event seeks the master timeline to that step, highlights the entity, and focuses the camera.
4. **Bottom Timeline / Scrubber**:
   - Color-coded event marker rail.
   - Scrub slider synchronized with 3D nodes, telemetry, inspector, and events feed.
   - Playback controls: `⏮ First`, `◀ Step`, `▶ Play / ⏸ Pause`, `Step ▶`, `⏭ End`, speed cycle (`0.25×`, `0.5×`, `1.0×`, `2.0×`, `5.0×`, `10.0×`).
   - Phase badge (`CONTEXT`, `DELAY`, `PROBE / REWARD`).

### 7.3 Structural vs Functional Connectivity
- **Structural Mode**: Renders actual anatomical synaptic wiring with opacity proportional to $|w_{ij}|$.
- **Functional Mode**: Dims quiescent connections to 3% opacity and illuminates active traveling signal pulses and co-firing synchrony pathways.

### 7.4 Keyboard Shortcuts
- `Space`: Play / Pause
- `← / →`: Step Backward / Step Forward
- `Home / End`: Seek to Start / End
- `L`: Jump to LIVE edge-of-time
- `R`: Restart REPLAY
- `F`: Focus camera on selected object
- `O`: Reset camera position
- `Esc`: Clear selection
- `1`: Structural connectivity mode
- `2`: Functional connectivity mode
- `?`: Toggle keyboard shortcuts modal

