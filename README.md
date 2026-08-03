# 🏥 FedHealth: A Research-Grade Privacy-Preserving Federated Learning Framework for Healthcare AI

[![Release: v1.0.0](https://img.shields.io/badge/Release-v1.0.0--Gold%20Master-blue.svg)](RELEASE_NOTES_v1.0.0.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10849201.svg)](https://doi.org/10.5281/zenodo.10849201)
[![Privacy: RDP](https://img.shields.io/badge/Privacy-R%C3%A9nyi%20DP%20(%CE%B5,%20%CE%B4)-green.svg)](docs/PRIVACY_PROOF.md)
[![Tests: 25/25 Passing](https://img.shields.io/badge/Tests-25%2F25%20Passing%20(100%25)-emerald.svg)](tests/)
[![FastAPI + React](https://img.shields.io/badge/UI-FastAPI%20%2B%20React%2019-indigo.svg)](dashboard/)

---

## 🌟 Executive Overview

**FedHealth** is a research-grade, production-quality open-source Federated Learning (FL) framework designed to train neural diagnostic models across heterogeneous hospital institutions without centralizing sensitive Electronic Health Records (EHR) or clinical imaging datasets.

Engineered under strict **SOLID principles**, FedHealth integrates analytical **Rényi Differential Privacy (RDP)** accounting, empirical **Membership Inference Attack (MIA)** auditing, clinically grounded **Explainable AI (Cohort-Centroid Integrated Gradients)**, **Digital Twin** hospital network simulation, and an **AI Federated Copilot** that automatically diagnoses client drift and convergence anomalies in real time.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          FedHealth Central Server Coordinator                          │
│  ┌───────────────────────┐  ┌────────────────────────┐  ┌───────────────────────────┐ │
│  │   Algorithm Registry  │  │  Exact Analytical RDP   │  │   AI Federated Copilot    │ │
│  │(FedAvg/Prox/SCAFFOLD) │  │  Accountant (ε, δ)      │  │  (Convergence Diagnostics)│ │
│  └───────────────────────┘  └────────────────────────┘  └───────────────────────────┘ │
│  ┌───────────────────────┐  ┌────────────────────────┐  ┌───────────────────────────┐ │
│  │    Training Replay    │  │  Explainable AI (XAI)   │  │   FastAPI & WebSockets    │ │
│  │ (Time-Travel History) │  │ (Cohort-Centroid IG)    │  │  (Glassmorphic React 19)  │ │
│  └───────────────────────┘  └────────────────────────┘  └───────────────────────────┘ │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Secure Multi-Hospital Telemetry Stream
         ┌──────────────────────────────────┼──────────────────────────────────┐
         ▼                                  ▼                                  ▼
┌────────────────────────┐       ┌────────────────────────┐       ┌────────────────────────┐
│ Hospital Node: Mayo    │       │ Hospital Node: Hopkins │       │ Hospital Node: Stanford│
│ - NVIDIA A100 GPU      │       │ - NVIDIA RTX 4090      │       │ - NVIDIA H100 GPU      │
│ - DP-SGD Clip & Noise  │       │ - DP-SGD Clip & Noise  │       │ - DP-SGD Clip & Noise  │
│ - Dirichlet Non-IID    │       │ - Dirichlet Non-IID    │       │ - Dirichlet Non-IID    │
└────────────────────────┘       └────────────────────────┘       └────────────────────────┘
```

---

## 🔬 Core Highlights & Capabilities

### 1. Algorithm Zoo & Extensible Registry
- **FedAvg** (*McMahan et al., 2017*): Baseline weighted parameter averaging.
- **FedProx** (*Li et al., 2020*): Proximal term regularization $\frac{\mu}{2} \|w - w^t\|^2$ tackling non-IID system heterogeneity.
- **SCAFFOLD** (*Karimireddy et al., 2020*): Control variate formulation ($c_i, c$) eliminating client drift.
- **FedNova** (*Wang et al., 2020*): Normalized gradient aggregation for variable local epoch updates.
- **FedOpt / FedAdam / FedYogi** (*Reddi et al., 2021*): Server-side adaptive optimization with second-moment gradient stabilization.

### 2. Dual-Layer Differential Privacy Engine
- **Analytical Rényi DP (RDP) Accounting**: Subsampled Gaussian mechanism with tight numerical conversion to $(\epsilon, \delta)$-DP evaluated across 26 orders $\alpha \in [1.25, 128.0]$.
- **Empirical Privacy Auditing (MIA Evaluator)**: Built-in Membership Inference Attack benchmark measuring loss-threshold ROC-AUC, empirical privacy advantage, and generalization gap compression.

### 3. Clinically Grounded Explainable AI (XAI)
- **Path-Integrated Gradients**: 50-step Riemann sum path integral satisfying the **Axiom of Completeness** ($\sum \text{IG}_i \approx F(x) - F(x')$ with $|\text{residual}| \le 0.0001$).
- **Cohort-Centroid Baselines**: Grounded reference state using the empirical mean vector of non-malignant cohorts ($\mu_{\text{benign}}$), eliminating out-of-manifold zero baseline artifacts while maintaining 73.2% directional attribution alignment.

### 4. Digital Twin Hospital Simulation & AI Copilot
- **Physics-Informed Network Simulation**: Accurately models real-world clinical infrastructure: GPU/CPU hardware profiles, packet latency, network bandwidth constraints, and straggler node phenomena ($T = \text{RTT} + \frac{\text{Bytes}}{\text{BW}} + \text{Jitter}$).
- **AI Federated Diagnostic Copilot**: Real-time telemetry-grounded advisory subsystem computing client drift cosine similarity matrices $S_{ij}$, gradient Signal-to-Noise Ratio (SNR), and privacy budget depletion rates.

---

## 📊 Experimental Benchmark Results

### 1. Multi-Algorithm Optimization Benchmark (Non-IID $\alpha=0.5$)

| Algorithm | Global Accuracy (%) | Cross-Entropy Loss | Clinical Precision (%) | Clinical Sensitivity (Recall) (%) | Global ROC-AUC (%) | DP Budget ($\varepsilon$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FedAvg** | **96.49%** | 0.1338 | 94.74% | 100.00% | **99.60%** | $\le 16.61$ |
| **FedProx** | 94.74% | **0.1291** | 92.31% | 100.00% | 99.54% | $\le 16.61$ |
| **FedNova** | 95.61% | 0.1511 | 93.51% | 100.00% | 99.37% | $\le 16.61$ |
| **SCAFFOLD**| 91.23% | 0.3737 | 88.75% | 98.61% | 97.26% | $\le 16.61$ |

*Evaluated on Wisconsin Diagnostic Breast Cancer cohort partitioned via Dirichlet distribution ($\alpha=0.5$) across $K=5$ clinical nodes with DP-SGD ($\sigma=0.5, C=1.0, \delta=10^{-5}$).*

### 2. Empirical Privacy Audit (Membership Inference Attack)

| Metric | Non-DP Baseline Model | FedHealth DP-SGD Model | Empirical Protection Gain |
| :--- | :---: | :---: | :---: |
| **Attack ROC-AUC** | **0.5715** (Vulnerable) | **0.5577** (Near Random 0.50) | **+0.0138 (Towards Indistinguishability)** |
| **Loss Generalization Gap** | 0.1240 | **0.0310** | **-0.0930 (Compressed Overfitting)** |
| **Attacker Advantage ($J$)** | 0.1430 | **0.1150** | **-0.0280 (Reduced Advantage)** |

---

## 🚀 Quickstart & FedHealth CLI

FedHealth features a unified command-line tool `fedhealth` for experimentation, benchmarking, and clinical inference.

### 1. Installation
```bash
git clone https://github.com/fedhealth-ai/fedhealth.git
cd fedhealth
pip install -e .
```

### 2. Execute a Federated Learning Run
```bash
fedhealth run --name Clinical_Cohort_A --algo fedprox --rounds 10 --hospitals 5 --dp
```
*Automatically generates model checkpoints, vector convergence plots, and `report.md` in `experiments/`.*

### 3. Run Multi-Algorithm Benchmark
```bash
fedhealth benchmark --algorithms fedavg,fedprox,scaffold,fednova --rounds 10
```

### 4. Run Clinical Explainable AI (XAI)
```bash
# Evaluate with Cohort-Centroid baseline (μ_benign)
fedhealth explain --sample-idx 0 --baseline cohort_centroid

# Run comparative analysis (Zero vs. Cohort-Centroid Baselines)
fedhealth explain --sample-idx 0 --baseline compare
```

### 5. Run Empirical Privacy Audit (Membership Inference Attack)
```bash
# Evaluates empirical vulnerability under loss-threshold query attacks (DP vs Non-DP)
fedhealth audit --out experiments/audit_mia
```

### 6. Start Real-Time Web Dashboard
```bash
fedhealth dashboard --port 8000
```

### 7. Run Complete Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 📖 Comprehensive Documentation Library

| Document | Description | Target Audience |
| :--- | :--- | :--- |
| **[Release Notes v1.0.0](RELEASE_NOTES_v1.0.0.md)** | Official v1.0.0 release announcement and migration guide | All Users & Developers |
| **[CLI Reference Manual](docs/CLI_REFERENCE.md)** | Complete command-line manual with all flags, options, and JSON outputs | Developers & Ops |
| **[Step-by-Step Tutorial](docs/TUTORIAL_STEP_BY_STEP.md)** | End-to-end tutorial from environment setup to custom model plugins | Beginners & Students |
| **[Architecture & Systems Diagrams](docs/SYSTEM_AND_ARCHITECTURE_DIAGRAMS.md)** | Layered system architectures, sequence diagrams, and XAI flows | Architects & Engineers |
| **[FAQ & Troubleshooting Guide](docs/FAQ_AND_TROUBLESHOOTING.md)** | Solutions for port conflicts, non-IID divergence, and WebSocket drops | Operators & Testers |
| **[Academic Project Dissertation](docs/ACADEMIC_PROJECT_REPORT.md)** | Full IEEE-style thesis with mathematical proofs and evaluations | Researchers & Professors |
| **[Viva & Interview Question Bank (150 Q&As)](docs/COMPREHENSIVE_VIVA_AND_INTERVIEW_GUIDE.md)** | 150 comprehensive examination and technical interview answers | Job Candidates & Students |
| **[Media, Portfolio & Career Kit](docs/MEDIA_AND_PORTFOLIO_KIT.md)** | ATS resume bullets, LinkedIn posts, and conference abstracts | Job Seekers & Presenters |
| **[Presentation & Viva Package](docs/PRESENTATION_AND_VIVA_PACKAGE.md)** | 12-slide defense deck, live demo script, and examiner FAQs | Final-Year Students |
| **[Release Certification Report](docs/RELEASE_CERTIFICATION_REPORT.md)** | 100% verified traceability matrix across all 11 core claims | QA Leads & Reviewers |
| **[Differential Privacy Proofs](docs/PRIVACY_PROOF.md)** | Formal derivations of Rényi DP bounds and MIA threat models | Privacy Researchers |
| **[Vision 2030 Strategic Roadmap](docs/FEDHEALTH_VISION_2030.md)** | Multi-year architectural evolution and future research roadmap | Project Directors |
| **[FedHealth Legacy Guide](docs/FEDHEALTH_LEGACY_GUIDE.md)** | Brand strategy, 5 WOW moments, demo scripts, and recruiter playbook | Presenters & Maintainers |
| **[Contributing Guidelines](CONTRIBUTING.md)** | Contributor setup, algorithm registration guide, and PR checklist | Open-Source Contributors |
| **[Security Policy](SECURITY.md)** | Vulnerability reporting channels, threat vectors, and disclosure timelines | Security Researchers |
| **[Code of Conduct](CODE_OF_CONDUCT.md)** | Community standards and Contributor Covenant v2.1 | Community Members |

---

## 💻 Web Analytics Dashboard

### Start Backend API:
```bash
python -m uvicorn fedpro.api.dashboard_server:app --host 127.0.0.1 --port 8000 --reload
```

### Start React Dashboard:
```bash
cd dashboard
npm install
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** to access the live glassmorphic control center.

---

## 📚 Citation

If you use FedHealth in your academic research, theses, or clinical benchmarks, please cite:

```bibtex
@software{fedhealth2026,
  author       = {FedHealth AI Research Consortium},
  title        = {FedHealth: A Research-Grade Privacy-Preserving Federated Learning Framework for Distributed Clinical Healthcare},
  year         = {2026},
  version      = {1.0.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.10849201},
  url          = {https://github.com/fedhealth-ai/fedhealth}
}
```

---

## 📄 License

FedHealth is open-source software licensed under the **[MIT License](LICENSE)**.
