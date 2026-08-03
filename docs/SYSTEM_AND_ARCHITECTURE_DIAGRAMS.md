# FedHealth System Architecture & Engineering Diagrams

This document provides visual and technical diagrams of the **FedHealth v1.0.0** architecture, communication workflows, privacy pipelines, and Explainable AI subsystems.

---

## 1. High-Level Layered System Architecture

```mermaid
graph TD
    subgraph UI_Layer ["Presentation & Telemetry Layer"]
        UI["React 19 Glassmorphic Dashboard"]
        CLI["FedHealth Unified CLI (`fedhealth`)"]
        API["FastAPI REST & WebSocket Server"]
    end

    subgraph Intelligence_Layer ["Intelligence & Diagnostics Layer"]
        Copilot["AI Federated Diagnostic Copilot"]
        XAI["Clinical Explainer (Cohort-Centroid IG)"]
        MIA["Empirical MIA Privacy Evaluator"]
        Tracker["Automated Experiment Tracker"]
    end

    subgraph Core_FL_Layer ["Federated Orchestration Layer"]
        Server["FLServer (Global Coordinator)"]
        Aggregators["Algorithm Registry (FedAvg, FedProx, SCAFFOLD, FedNova, FedAdam)"]
        RDP["RDP Privacy Accountant (26 Orders)"]
    end

    subgraph Simulation_Layer ["Digital Twin Hospital Simulation Layer"]
        NetSim["WAN Physics & Latency Simulator"]
        H1["Hospital Node 1 (Tier-1 GPU)"]
        H2["Hospital Node 2 (Tier-2 Workstation)"]
        H3["Hospital Node 3 (Tier-3 Edge Node)"]
    end

    CLI --> Server
    UI <-->|WebSocket & REST| API
    API <--> Server
    Server --> Aggregators
    Server --> Copilot
    Server --> Tracker
    Server --> RDP
    Aggregators <--> NetSim
    NetSim <--> H1
    NetSim <--> H2
    NetSim <--> H3
    H1 --> XAI
    Server --> MIA
```

---

## 2. Federated Communication & Aggregation Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant S as FLServer (Global Coordinator)
    participant N as Digital Twin Network Simulator
    participant C as Hospital Node (Client k)
    participant DP as DP-SGD & RDP Engine

    Note over S: Initialize Global Model w_0 & RDP Accountant
    loop Each Communication Round t = 1, ..., T
        S->>N: Broadcast Global Weights w_t
        N->>C: Inject WAN Latency (RTT + Bytes/BW + Jitter)
        Note over C: Load Local Non-IID Dataset D_k
        loop Local Epochs e = 1, ..., E
            Note over C: Compute Gradients g_i = ∇L(w, (x_i, y_i))
            C->>DP: Clip Gradient Norm ||g_i||_2 ≤ C
            DP->>C: Add Calibrated Noise N(0, σ²C²I)
            Note over C: Local Optimizer Step (FedAvg/FedProx/SCAFFOLD)
        end
        C->>N: Transmit Local Parameter Delta Δw_k
        N->>S: Deliver Client Updates
        Note over S: Server Aggregates Updates w_{t+1} = Σ (n_k/N) w_k
        S->>DP: Update Analytical RDP Orders α ∈ [1.25, 128.0]
        Note over S: Compute Drift Matrix S_ij & Gradient SNR
        S->>S: Global Evaluation (Acc, Loss, ROC-AUC, ε)
    end
    Note over S: Finalize Checkpoints & Publish Telemetry
```

---

## 3. Differential Privacy (DP-SGD & RDP) Pipeline

```mermaid
flowchart LR
    subgraph Local_Hospital ["Hospital Local Training"]
        RawBatch["Mini-Batch (x, y)"] --> Loss["Compute Loss ℓ(f(x), y)"]
        Loss --> Backprop["Backpropagation ∇_w ℓ"]
        Backprop --> PerSampleGrad["Per-Sample Gradient g_i"]
        PerSampleGrad --> Clip["L2 Norm Clipping: g_i / max(1, ||g_i||_2 / C)"]
        Clip --> Noise["Add Gaussian Noise: + N(0, σ² C² I)"]
        Noise --> LocalUpdate["Perturbed Gradient g̃"]
    end

    subgraph Privacy_Accounting ["Analytical & Empirical Accounting"]
        LocalUpdate --> RDPOrders["RDP Order Accumulation: R_α = α / (2σ²)"]
        RDPOrders --> PoissonSubsample["Subsampled Amplification (q = |B|/N)"]
        PoissonSubsample --> EpsilonSolver["Optimal Convex Minimization: min_α (R_α + ln(1/δ)/(α-1))"]
        EpsilonSolver --> ExactDP["Guaranteed (ε, δ)-DP Bound"]
        LocalUpdate --> MIAAttack["Empirical MIA Evaluator: Loss-Threshold Attack ROC-AUC"]
    end
```

---

## 4. Clinically Grounded Explainable AI (XAI) Pipeline

```mermaid
flowchart TD
    subgraph Baseline_Construction ["Physiological Baseline Selection"]
        RefData["Healthy / Benign Cohort (y = 0)"] --> CentroidCalc["Empirical Mean Vector: μ_benign = E[x | y=0]"]
        CentroidCalc --> Baseline["Cohort-Centroid Baseline x'"]
    end

    subgraph Path_Integration ["Path-Integrated Gradients (Sundararajan et al., 2017)"]
        PatientInput["Patient Diagnostic Vector x"] --> Interpolation["Generate Path: x(α) = x' + α(x - x'), α ∈ [0, 1]"]
        Baseline --> Interpolation
        Interpolation --> Forward["Forward Pass through Global Neural Network"]
        Forward --> Gradients["Compute Path Gradients: ∂F(x(α)) / ∂x"]
        Gradients --> RiemannSum["Riemann Sum Integration: ∫_0^1 (∂F/∂x) dα"]
        RiemannSum --> Attributions["Feature Attribution Vector: IG_i = (x_i - x_i') × AvgGrad_i"]
    end

    subgraph Clinical_Delivery ["Clinical Decision Support"]
        Attributions --> Completeness["Verify Axiom of Completeness: Σ IG_i ≈ F(x) - F(x')"]
        Completeness --> Rationale["Natural Language Clinical Diagnostic Rationale"]
        Rationale --> Output["Clinician Diagnostic Report (Risk Escalators vs Protective Indicators)"]
    end
```
