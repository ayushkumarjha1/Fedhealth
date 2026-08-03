# Changelog

All notable changes to **FedHealth** are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-08-02

### 🌟 Major Release Highlights
- **Production Open-Source Architecture**: Transitioned codebase to strict SOLID modularity with explicit type safety, Pydantic V2 schemas, and zero unhandled platform exceptions.
- **Unified Command-Line Interface (`fedhealth`)**: End-to-end CLI with subcommands: `run`, `benchmark`, `explain`, `dashboard`, `audit`.
- **Empirical Differential Privacy Validation**: Built-in Membership Inference Attack (`fedhealth audit`) computing loss-threshold ROC-AUC, empirical privacy advantage, and generalization gap compression.
- **Clinically Grounded Explainable AI (XAI)**: Integrated Gradients supporting both Zero and **Cohort-Centroid Baselines** ($\mu_{\text{benign}}$), provably satisfying the Axiom of Completeness across both references.
- **Scientific Differential Privacy Engine**: Subsampled Gaussian DP-SGD mechanism with analytical Rényi Differential Privacy (RDP) accounting ($\alpha \in [1.25, 128]$) and exact $(\epsilon, \delta)$ conversion.
- **Grounded AI Federated Copilot**: Telemetry-driven diagnostic engine computing pairwise client update cosine similarity $S_{ij}$, gradient SNR, and privacy budget velocity $\frac{\partial \epsilon}{\partial t}$, categorizing outputs into verified facts, statistical observations, and heuristic recommendations.
- **Automated Experiment Tracker**: Structured run persistence producing timestamped PyTorch checkpoints (`model_best.pt`, `model_final.pt`), vector diagnostic plots (ROC/PR curves, confusion matrices, client drift heatmaps), raw JSON logs, and clinical Markdown research reports.
- **Digital Twin Hospital Simulation**: Physics-informed compute, networking ($T = \text{RTT} + \frac{\text{Bytes}}{\text{BW}} + \text{Jitter}$), and straggler node emulation across heterogeneous hospital tiers.
- **Real-Time Analytics Dashboard**: FastAPI + WebSocket backend connected to a dark-mode glassmorphic React 19 UI with live telemetry, time-travel history replay, and interactive patient diagnostics.

### 🔬 Algorithms Implemented & Mathematically Verified
- **FedAvg** (*McMahan et al., 2017*): Exact sample-weighted parameter aggregation.
- **FedProx** (*Li et al., 2020*): Proximal regularization $\frac{\mu}{2}\|w - w^t\|_2^2$ tackling system and statistical heterogeneity.
- **SCAFFOLD** (*Karimireddy et al., 2020*): Option II control variate formulation $c_i^+ = c_i - c + \frac{1}{K\eta_l}(x - y_i)$ with zero-drift gradient correction.
- **FedNova** (*Wang et al., 2020*): Normalized gradient aggregation for variable local epoch regimes.
- **FedOpt / FedAdam / FedYogi / FedAvgM** (*Reddi et al., 2021*): Server-side adaptive optimization with second moment stabilization.

### 🛡️ Testing & Quality Assurance
- 25 comprehensive unit, integration, and mathematical invariant tests achieving 100% pass rate in 1.25s.
- Cross-platform verified across Linux, macOS, and Windows.
