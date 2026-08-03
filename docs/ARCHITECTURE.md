# FedHealth: System Architecture & Design Specification

## 1. Architectural Philosophy
FedHealth is engineered as a modular, extensible, research-grade federated learning platform specifically tailored to clinical healthcare environments. It adheres to strict software engineering standards:

- **SOLID Design Principles**: Single responsibility across data partitioning, local model training, aggregation protocols, and privacy accounting.
- **Factory & Registry Pattern**: Decoupled algorithm and model registrations allow seamless extension without modifying core orchestration loops.
- **Strict Pydantic V2 Schemas**: Complete type validation, default hyperparameter guards, and bidirectional JSON/YAML serialization.
- **Privacy-by-Design**: Integrated Rényi Differential Privacy (RDP) and DP-SGD gradient perturbation.

---

## 2. Core Subsystems

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FedHealth Central Server                        │
│  ┌───────────────────────┐  ┌────────────────────┐  ┌────────────────┐ │
│  │   Algorithm Registry   │  │   RDP Accountant   │  │   AI Copilot   │ │
│  │ (FedAvg/Prox/SCAFFOLD) │  │  (Exact Analytical)│  │ (Telemetry ML) │ │
│  └───────────────────────┘  └────────────────────┘  └────────────────┘ │
│  ┌───────────────────────┐  ┌────────────────────┐  ┌────────────────┐ │
│  │    Training Replay    │  │  XAI Explainer     │  │ FastApi Engine │ │
│  │ (Time-Travel History) │  │ (Biomarker Saliency)│ │  (WebSockets)  │ │
│  └───────────────────────┘  └────────────────────┘  └────────────────┘ │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Communication Network (gRPC / WS)
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│ Hospital Node │           │ Hospital Node │           │ Hospital Node │
│ (Mayo Clinic) │           │(Johns Hopkins)│           │(Stanford Med) │
│ - NVIDIA A100 │           │ - RTX 4090    │           │ - NVIDIA H100 │
│ - DP-SGD Clip │           │ - DP-SGD Clip │           │ - DP-SGD Clip │
│ - Local Data  │           │ - Local Data  │           │ - Local Data  │
└───────────────┘           └───────────────┘           └───────────────┘
```

### A. Algorithm Registry (`fedpro.algorithms`)
Implements standard and advanced federated optimization algorithms:
- `FedAvg`: Federated Averaging (McMahan et al., 2017)
- `FedProx`: Proximal Regularization for non-IID data (Li et al., 2020)
- `SCAFFOLD`: Stochastic Controlled Averaging for Client Drift (Karimireddy et al., 2020)
- `FedNova`: Normalized Averaging under heterogeneous local steps (Wang et al., 2020)
- `FedOpt`: Adaptive server optimization including `FedAdam` and `FedYogi` (Reddi et al., 2021)

### B. Privacy Engine (`fedpro.privacy`)
- Exact analytical **Rényi Differential Privacy (RDP)** accounting over Gaussian mechanisms.
- Converts accumulated orders $\alpha \in [1.5, 128]$ into strict $(\epsilon, \delta)$-DP guarantees:
  $$\epsilon(\delta) = \min_{\alpha > 1} \left( \mathcal{R}_\alpha + \frac{\log(1/\delta)}{\alpha - 1} \right)$$
- Local gradient clipping with dynamic $L_2$ norm enforcement and calibrated noise injection.

### C. Digital Twin Simulation Engine (`fedpro.core.hospital`)
Simulates realistic hospital compute hardware, geographical latency, and network bandwidth:
- Hardware profiles: NVIDIA A100, RTX 4090, H100, Intel Xeon Platinum.
- Straggler injection and synthetic packet jitter.
- Trust scores and institutional privacy policies.

### D. AI Federated Copilot (`fedpro.copilot`)
Continuously monitors telemetry metrics across communication rounds:
- Detects client drift and non-IID label divergence.
- Predicts privacy budget exhaustion.
- Recommends hyperparameter adjustments ($\mu$, learning rate, local epochs).

### E. Explainable AI Subsystem (`fedpro.xai`)
- Integrated Gradients and feature saliency attribution for clinical biomarkers.
- Generates clinician-friendly diagnostic rationales and patient-specific risk factor polarity.

---

## 3. Real-Time Telemetry & WebSockets
The central server broadcasts structured event packets across WebSocket connections:
- `round_start`: Commencing round metadata.
- `round_end`: Global loss, accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and hospital node states.
- `simulation_complete`: Final benchmark convergence summary.
