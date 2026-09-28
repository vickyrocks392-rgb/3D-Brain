# Phase 1 Implementation Specification: Minimal Adaptive Substrate & Benchmark Engine

## 1. Scope & Objective

Phase 1 implements the minimal closed-loop scientific pipeline:
$$\text{Task Generation} \longrightarrow \text{Model Computation} \longrightarrow \text{Evaluation} \longrightarrow \text{Telemetry} \longrightarrow \text{.brainlog} \longrightarrow \text{3D Visualization}$$

The implementation strictly satisfies the requirements of:
* [PROJECT_CHARTER.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/PROJECT_CHARTER.md)
* [SYSTEM_PRINCIPLES.md](file:///Users/kalachanddash/Documents/3D-Brain/docs/SYSTEM_PRINCIPLES.md)
* [EXPERIMENT_001_SPECIFICATION.md](file:///Users/kalachanddash/Documents/3D-Brain/experiments/EXPERIMENT_001_SPECIFICATION.md)
* [VISUALIZATION_CONCEPT_AND_TELEMETRY.md](file:///Users/kalachanddash/Documents/3D-Brain/visualization/VISUALIZATION_CONCEPT_AND_TELEMETRY.md)

---

## 2. Directory Layout & Module Structure

```text
3D-Brain/
├── brain/
│   ├── __init__.py
│   ├── base.py                   # Common BrainModel interface & Abstract Base Class
│   ├── rnn.py                    # BASE-RNN: Standard Elman recurrent substrate
│   ├── gru.py                    # BASE-GRU: Standard Gated Recurrent Unit substrate
│   └── liquid.py                 # CAND-LI: Continuous-time Leaky Integrator / Liquid substrate
├── benchmarks/
│   ├── __init__.py
│   └── delayed_match.py          # Continuous Delayed Match-to-Sample task generator
├── experiments/
│   ├── __init__.py
│   ├── runner.py                 # Reproducible experiment runner & metric logger
│   └── run_exp001.py             # CLI entrypoint for Experiment 001 matrix execution
├── visualization/
│   ├── __init__.py
│   ├── telemetry.py              # Telemetry emitter, topology generator, & .brainlog serializer
│   ├── player.html               # Interactive 3D WebGL/Canvas visualizer & timeline scrubber
│   └── serve.py                  # Lightweight local desktop server / launcher
└── tests/
    ├── test_benchmarks.py        # Deterministic generation & task correctness tests
    ├── test_models.py            # Parameter counts, shapes, numerical sanity tests
    ├── test_telemetry.py         # Schema compliance & .brainlog round-trip tests
    └── test_runner.py            # Reproducibility & headless execution tests
```

---

## 3. Candidate Model Interface Contract

All three candidate models inherit from `brain.base.BrainModel` and expose an identical API:

```python
class BrainModel(ABC):
    @abstractmethod
    def reset(self, batch_size: int = 1) -> None:
        """Reset internal hidden state/traces to resting initial state."""
        ...

    @abstractmethod
    def step(self, x: np.ndarray) -> np.ndarray:
        """Execute a single time step given input x [batch_size, input_dim].
        Returns output y [batch_size, output_dim]."""
        ...

    @abstractmethod
    def forward_sequence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Execute complete sequence X [batch_size, seq_len, input_dim].
        Returns outputs [batch_size, seq_len, output_dim] and hidden states [batch_size, seq_len, hidden_dim]."""
        ...

    @abstractmethod
    def get_state(self) -> np.ndarray:
        """Return current hidden state vector h [batch_size, hidden_dim]."""
        ...

    @abstractmethod
    def get_activation(self) -> np.ndarray:
        """Return current non-linear activation vector a [batch_size, hidden_dim]."""
        ...

    @abstractmethod
    def parameter_count(self) -> int:
        """Return total trainable parameters."""
        ...

    @abstractmethod
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return dictionary of weight matrices and biases."""
        ...
```

---

## 4. Models Mathematical Formulation

### 4.1 BASE-RNN (Standard Elman Recurrent Unit)
* **Parameters**: $W_{xh} \in \mathbb{R}^{d_{\text{in}} \times d_h}$, $W_{hh} \in \mathbb{R}^{d_h \times d_h}$, $\mathbf{b}_h \in \mathbb{R}^{d_h}$, $W_{hy} \in \mathbb{R}^{d_h \times d_{\text{out}}}$, $b_y \in \mathbb{R}^{d_{\text{out}}}$.
* **State Update**:
  $$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + \mathbf{b}_h)$$
  $$a_t = h_t$$
  $$\hat{y}_t = W_{hy} a_t + b_y$$
* **Parameter Count Formula**:
  $$N_{\text{RNN}} = d_h^2 + d_h(d_{\text{in}} + d_{\text{out}} + 1) + d_{\text{out}}$$

### 4.2 BASE-GRU (Gated Recurrent Unit)
* **Parameters**: Reset gate $(W_r, U_r, b_r)$, Update gate $(W_z, U_z, b_z)$, Candidate state $(W_h, U_h, b_h)$, Readout $(W_{hy}, b_y)$.
* **State Update**:
  $$r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$$
  $$z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$$
  $$\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)$$
  $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
  $$a_t = h_t, \quad \hat{y}_t = W_{hy} a_t + b_y$$
* **Parameter Count Formula**:
  $$N_{\text{GRU}} = 3(d_h^2 + d_h \cdot d_{\text{in}} + d_h) + d_h \cdot d_{\text{out}} + d_{\text{out}}$$

### 4.3 CAND-LI (Continuous-Time Leaky-Integrator / Liquid Unit)
* **Parameters**: $W_{xh} \in \mathbb{R}^{d_{\text{in}} \times d_h}$, $W_{hh} \in \mathbb{R}^{d_h \times d_h}$, $\mathbf{b}_h \in \mathbb{R}^{d_h}$, learnable time-constants $\boldsymbol{\tau} \in \mathbb{R}^{d_h}$, Readout $(W_{hy}, b_y)$.
* **State ODE Update (Forward Euler with $\Delta t = 1.0$)**:
  $$\frac{d h_i}{dt} = -\frac{1}{\tau_i} h_i(t) + \frac{1}{\tau_i} \tanh\left(\sum_j W_{xh}^{ij} x_j(t) + \sum_k W_{hh}^{ik} a_k(t) + b_i\right)$$
  $$h_{t} = h_{t-1} + \Delta t \cdot \frac{dh}{dt} = \left(1 - \frac{\Delta t}{\tau}\right) \odot h_{t-1} + \frac{\Delta t}{\tau} \odot \tanh(W_{xh} x_t + W_{hh} a_{t-1} + \mathbf{b}_h)$$
  $$a_t = \tanh(h_t)$$
  $$\hat{y}_t = W_{hy} a_t + b_y$$
* **Parameter Count Formula**:
  $$N_{\text{LI}} = d_h^2 + d_h(d_{\text{in}} + d_{\text{out}} + 2) + d_{\text{out}} \quad (\text{includes } \boldsymbol{\tau})$$

---

## 5. Parameter Budget Allocation

Given $d_{\text{in}} = 2$ and $d_{\text{out}} = 1$:

| Target Budget | BASE-RNN ($d_h$) | BASE-RNN Actual Params | BASE-GRU ($d_h$) | BASE-GRU Actual Params | CAND-LI ($d_h$) | CAND-LI Actual Params |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **~300 params** | $d_h = 16$ | $16^2 + 16(4) + 1 = \mathbf{321}$ | $d_h = 9$ | $3(81 + 18 + 9) + 9 + 1 = \mathbf{334}$ | $d_h = 15$ | $15^2 + 15(5) + 1 = \mathbf{301}$ |
| **~1000 params**| $d_h = 30$ | $30^2 + 30(4) + 1 = \mathbf{1021}$| $d_h = 16$ | $3(256 + 32 + 16) + 16 + 1 = \mathbf{931}$| $d_h = 29$ | $29^2 + 29(5) + 1 = \mathbf{987}$ |
| **~3000 params**| $d_h = 52$ | $52^2 + 52(4) + 1 = \mathbf{2913}$| $d_h = 30$ | $3(900 + 60 + 30) + 30 + 1 = \mathbf{2971}$| $d_h = 51$ | $51^2 + 51(5) + 1 = \mathbf{2857}$ |

All candidates match target budgets within $\pm 7\%$.

---

## 6. The .brainlog File Format Specification

A `.brainlog` file is a single JSON document (or NDJSON/streaming chunks) structured as:

```json
{
  "brainlog_version": "1.0",
  "metadata": {
    "experiment_id": "EXP001",
    "model_id": "CAND-LI",
    "budget": 300,
    "seed": 42,
    "delay": 25,
    "total_steps": 40,
    "dt_ms": 50.0,
    "timestamp": "2026-09-28T16:00:00Z"
  },
  "topology": {
    "nodes": [
      {"id": 0, "type": "INPUT", "label": "Cue", "pos": [-2.0, 0.0, -8.0]},
      {"id": 1, "type": "INPUT", "label": "Flag", "pos": [2.0, 0.0, -8.0]},
      {"id": 2, "type": "RECURRENT", "label": "H_0", "pos": [-1.2, 0.5, -0.5]},
      ...
      {"id": 17, "type": "OUTPUT", "label": "Decision", "pos": [0.0, 0.0, 8.0]}
    ],
    "edges": [
      {"src": 0, "dst": 2, "weight": 0.42, "latency_steps": 1},
      ...
    ]
  },
  "frames": [
    {
      "step": 0,
      "time_ms": 0.0,
      "inputs": [1.0, 1.0],
      "activations": [1.0, 1.0, 0.0, 0.0, ...],
      "states": [1.0, 1.0, 0.0, 0.0, ...],
      "output": 0.001,
      "target": 0.0,
      "error": 0.001,
      "signals": [
        {"src": 0, "dst": 2, "progress": 1.0, "mag": 0.42}
      ],
      "events": [{"type": "CUE_RECEIVED", "node_id": 0}]
    },
    ...
  ]
}
```

This format is fully readable by headless validation tools, replay scripts, and the 3D visualizer.
