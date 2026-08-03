# FedHealth Quickstart & Developer Guide

Welcome to **FedHealth**, a production-grade, research-ready Federated Learning framework for privacy-preserving clinical diagnostics.

---

## 1. Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- Node.js 18+ (for Web Dashboard)

### Install via pip / editable mode
```bash
git clone https://github.com/your-org/fedhealth.git
cd fedhealth
pip install -e .
```

---

## 2. Using the `fedhealth` CLI

FedHealth comes with a unified command-line tool `fedhealth`:

### A. Run a Federated Simulation Experiment
```bash
fedhealth run --name Clinical_Cohort_A --algo fedprox --rounds 10 --hospitals 5 --dp
```
*This command executes the federated training loop, logs metrics, generates publication-ready vector plots (`convergence_curves.png`, `privacy_frontier.png`, `confusion_matrix.png`), and saves model weights (`model_best.pt`, `model_final.pt`) in `experiments/<run_id>/`.*

### B. Run Multi-Algorithm Benchmark
```bash
fedhealth benchmark --algorithms fedavg,fedprox,scaffold,fednova,fedadam --rounds 10
```

### C. Run Clinical Explainable AI (XAI)
```bash
fedhealth explain --sample-idx 0
```

### D. Launch Real-Time Dashboard Server
```bash
fedhealth dashboard --port 8000
```

---

## 3. Launching the Real-Time Analytics Dashboard

### Step 1: Start the FastAPI Backend
```bash
python -m uvicorn fedpro.api.dashboard_server:app --host 127.0.0.1 --port 8000 --reload
```

### Step 2: Start the React Dashboard
```bash
cd dashboard
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 4. Custom Configuration & Python API

```python
from fedpro.configs.base_config import FedHealthConfig
from fedpro.core.server import FLServer
from fedpro.models.mlp import HealthcareMLP
from fedpro.data.medical_datasets import load_breast_cancer_data
from fedpro.data.partitioner import dirichlet_non_iid_partition
from fedpro.algorithms.registry import get_algorithm_client_class

# Configure Experiment
config = FedHealthConfig()
config.algorithm.name = "fedprox"
config.algorithm.mu = 0.01
config.privacy.enabled = True
config.privacy.noise_multiplier = 0.5
config.privacy.clip_norm = 1.0

# Load Dataset & Non-IID Dirichlet Skew
train_data, test_data, in_dim, num_classes, feature_names = load_breast_cancer_data()
subsets = dirichlet_non_iid_partition(train_data, num_clients=5, alpha=0.5)

# Initialize Neural Architecture & Orchestrator
global_model = HealthcareMLP(input_dim=in_dim, num_classes=num_classes)
server = FLServer(global_model=global_model, config=config, feature_names=feature_names)
```
