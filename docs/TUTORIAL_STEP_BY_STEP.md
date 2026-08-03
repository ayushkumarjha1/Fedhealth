# FedHealth Step-by-Step Developer Tutorial

This tutorial provides a complete walkthrough of **FedHealth**, guiding you from initial environment setup through advanced federated optimization, differential privacy configuration, Explainable AI diagnostics, and custom model integration.

---

## Prerequisites
- Python 3.10+
- PyTorch 2.0+
- Virtual environment tool (`venv` or `conda`)

---

## Step 1: Installation & Self-Verification

First, clone the repository and install FedHealth in editable development mode:

```bash
# Clone the repository
git clone https://github.com/fedhealth-ai/fedhealth.git
cd fedhealth

# Create a virtual environment
python -m venv .venv
# Activate on Windows:
.venv\Scripts\Activate.ps1
# Activate on Linux/macOS:
source .venv/bin/activate

# Install FedHealth
pip install -e .

# Run the 25-test verification suite
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected output: `Ran 25 tests ... OK` (completed in ~1.25s).

---

## Step 2: Running Your First Federated Experiment

Run a baseline 5-round federated training session across 5 simulated hospital nodes using **FedAvg**:

```bash
fedhealth run --name Quickstart_Tutorial --algo fedavg --rounds 5 --hospitals 5
```

### What Happens Behind the Scenes:
1. Loads the Wisconsin Diagnostic Breast Cancer cohort (569 patient records, 30 biomarkers).
2. Partitions data across 5 hospital nodes using a non-IID Dirichlet distribution ($\alpha=0.5$).
3. Simulates local client training across local epochs.
4. Aggregates parameter updates on the global server using sample weighting:
   $$w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_{t+1}^k$$
5. Computes global test evaluation metrics and saves artifacts in `experiments/Quickstart_Tutorial_<timestamp>/`.

---

## Step 3: Enabling Differential Privacy (DP-SGD)

To guarantee patient-level differential privacy, enable DP-SGD with calibrated Gaussian noise:

```bash
fedhealth run --name Privacy_Trial --algo fedprox --rounds 10 --dp --noise 0.8 --clip 1.0
```

### How Privacy Accounting Works:
- Each hospital node clips per-sample gradients to $L_2$ threshold $C=1.0$:
  $$g_i \gets \frac{g_i}{\max(1, \|g_i\|_2 / C)}$$
- Calibrated Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$ is injected prior to parameter transmission.
- The `RDPAccountant` tracks Rényi divergence across 26 orders $\alpha \in [1.25, 128.0]$ and outputs the exact $(\epsilon, \delta)$-DP bound in `metrics.json`.

---

## Step 4: Multi-Algorithm Comparative Benchmarking

Evaluate how different optimization algorithms handle non-IID statistical heterogeneity:

```bash
fedhealth benchmark --algorithms fedavg,fedprox,scaffold,fednova --rounds 10
```

### Interpretation of Results:
- **FedProx** adds proximal penalty $\frac{\mu}{2}\|w - w^t\|_2^2$ to stabilize local gradient drift.
- **SCAFFOLD** uses client control variates ($c_i, c$) to directly compensate for client variance.
- **FedNova** normalizes aggregation weights by local step counts $\tau_i$, preventing objective inconsistency.

---

## Step 5: Clinical Explainable AI (XAI) Diagnostics

Produce feature attributions using path-integrated gradients:

```bash
# Explain Patient Record #0 with Cohort-Centroid Baseline
fedhealth explain --sample-idx 0 --baseline cohort_centroid

# Compare Zero vs. Cohort-Centroid Baselines side-by-side
fedhealth explain --sample-idx 0 --baseline compare
```

### Why Cohort-Centroid Matters:
A zero baseline ($x'=0$) represents a non-physical clinical state (e.g. zero blood pressure). FedHealth computes $\mu_{\text{benign}} = \mathbb{E}[x \mid y=0]$ as the reference point, ensuring attributions represent true physiological deviations from healthy tissue.

---

## Step 6: Empirical Privacy Auditing (MIA)

Evaluate your model against Membership Inference Attacks:

```bash
fedhealth audit --out experiments/audit_mia
```

This generates `experiments/audit_mia/mia_evaluation.png` (ROC curves) and `experiments/audit_mia/mia_report.md`, comparing the attack vulnerability of Non-DP vs. DP-SGD models.

---

## Step 7: Launching the Web Dashboard

Start the FastAPI backend:

```bash
fedhealth dashboard --port 8000
```

In a separate terminal, launch the React frontend:

```bash
cd dashboard
npm install
npm run dev
```

Open `http://localhost:5173` in your browser to inspect live telemetry, client drift heatmaps, and time-travel replay.

---

## Step 8: Integrating a Custom PyTorch Model

You can easily integrate your own custom neural network into FedHealth:

```python
import torch
import torch.nn as nn
from fedpro.core.server import FLServer
from fedpro.configs.base_config import FedHealthConfig

# 1. Define custom model
class CustomOncologyCNN(nn.Module):
    def __init__(self, input_dim=30, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# 2. Instantiate and launch FL Server
config = FedHealthConfig()
config.training.num_rounds = 10
model = CustomOncologyCNN()
server = FLServer(global_model=model, config=config)
print("Custom model successfully registered with FedHealth FLServer!")
```
