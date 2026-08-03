# FedHealth CLI Complete Reference Manual

The `fedhealth` command-line interface provides a single unified entry point for federated learning orchestration, multi-algorithm benchmarking, Explainable AI diagnostics, empirical privacy auditing, and dashboard hosting.

---

## Command Syntax Overview

```bash
fedhealth <subcommand> [options]
```

### Available Subcommands

| Subcommand | Description | Primary Output |
| :--- | :--- | :--- |
| `run` | Executes a complete federated training experiment | Checkpoints (`.pt`), metrics (`.json`), plots (`.png`), report (`.md`) |
| `benchmark` | Executes a comparative benchmark across multiple FL algorithms | Terminal comparative summary table |
| `explain` | Generates clinical Integrated Gradients attribution diagnostics | Terminal feature attribution report & rationale |
| `audit` | Runs empirical Membership Inference Attack (MIA) privacy evaluation | ROC curve comparison plot & Markdown audit report |
| `dashboard` | Launches FastAPI backend & WebSocket telemetry broadcaster | Real-time REST and WebSocket server (`http://127.0.0.1:8000`) |

---

## 1. `fedhealth run`

Executes a multi-round federated learning simulation across distributed hospital nodes.

```bash
fedhealth run [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--name` | `string` | `"Clinical_Experiment"` | Name identifier for the run and its output directory |
| `--algo` | `choice` | `"fedavg"` | Federated algorithm (`fedavg`, `fedprox`, `scaffold`, `fednova`, `fedadam`) |
| `--rounds` | `int` | `5` | Total number of global federated communication rounds |
| `--hospitals` | `int` | `5` | Number of simulated clinical hospital nodes ($K$) |
| `--epochs` | `int` | `2` | Number of local training epochs ($E$) per hospital per round |
| `--alpha` | `float` | `0.5` | Dirichlet non-IID concentration parameter ($\alpha \to 0$ = extreme heterogeneity) |
| `--dp` | `flag` | `True` | Enable Differential Privacy via DP-SGD gradient perturbation |
| `--noise` | `float` | `0.5` | Calibrated Gaussian noise multiplier ($\sigma$) |
| `--clip` | `float` | `1.0` | Maximum $L_2$ gradient clipping threshold ($C$) |

### Example
```bash
fedhealth run --name Oncology_Trial_01 --algo fedprox --rounds 10 --hospitals 5 --epochs 3 --alpha 0.3 --dp --noise 0.6 --clip 1.0
```

### Generated Artifacts
Output directory: `experiments/<run_name>_<timestamp>/`
- `model_best.pt`: PyTorch weights for highest-accuracy global round
- `model_final.pt`: Final round global model weights
- `metrics.json`: Per-round training loss, validation accuracy, precision, recall, and $\epsilon$ expenditure
- `convergence.png`: High-resolution dual-axis plot of accuracy vs. privacy loss
- `report.md`: Structured clinical research summary

---

## 2. `fedhealth benchmark`

Runs a standardized comparative benchmark across multiple federated optimization algorithms on identical non-IID partitions.

```bash
fedhealth benchmark [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--algorithms` | `string` | `"fedavg,fedprox,scaffold,fednova"` | Comma-separated list of algorithms to evaluate |
| `--rounds` | `int` | `5` | Number of communication rounds per algorithm |

### Example
```bash
fedhealth benchmark --algorithms fedavg,fedprox,scaffold,fednova --rounds 10
```

### Sample Output
```
================================================================================
 FEDHEALTH ALGORITHM COMPARATIVE BENCHMARK RESULTS
================================================================================
algorithm  test_accuracy  test_loss  precision    recall   roc_auc  epsilon
   fedavg        96.49%     0.1338     94.74%   100.00%    99.60%    16.61
  fedprox        94.74%     0.1291     92.31%   100.00%    99.54%    16.61
  fednova        95.61%     0.1511     93.51%   100.00%    99.37%    16.61
 scaffold        91.23%     0.3737     88.75%    98.61%    97.26%    16.61
================================================================================
```

---

## 3. `fedhealth explain`

Computes mathematically grounded Integrated Gradients feature attributions and generates natural-language clinical rationales for an individual patient record.

```bash
fedhealth explain [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--sample-idx` | `int` | `0` | Index of the patient record within the test cohort |
| `--baseline` | `choice` | `"cohort_centroid"` | Attribution reference (`cohort_centroid`, `zeros`, `compare`) |

### Example (Standard Inference)
```bash
fedhealth explain --sample-idx 0 --baseline cohort_centroid
```

### Example (Comparative Baselines)
```bash
fedhealth explain --sample-idx 0 --baseline compare
```

### Sample Output
```
======================================================================
 CLINICAL XAI: ZERO VS. COHORT-CENTROID BASELINE COMPARISON
======================================================================
Attribution Directional Alignment (Cosine Similarity): 73.2%
Zero-Baseline Completeness Residual         : -0.00001
Cohort-Centroid Completeness Residual        : -0.00009

Clinical Takeaway:
  Attribution vectors share 73.2% directional alignment. Cohort-centroid baseline eliminates non-physical zero-reference artifacts.
======================================================================
```

---

## 4. `fedhealth audit`

Conducts an empirical privacy audit using loss-threshold and confidence query Membership Inference Attacks (MIA), comparing Non-DP vs. DP-SGD trained models.

```bash
fedhealth audit [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--out` | `string` | `"experiments/audit_mia"` | Output directory for audit figures and markdown report |

### Example
```bash
fedhealth audit --out experiments/audit_mia
```

### Sample Output
```
===========================================================================
 EMPIRICAL MEMBERSHIP INFERENCE ATTACK (MIA) AUDIT RESULTS
===========================================================================
Non-DP Baseline Attack ROC-AUC  : 0.5715 (Susceptible)
FedHealth DP-SGD Attack ROC-AUC : 0.5577 (Near Random Guess 0.50)
Empirical Attack AUC Reduction  : +0.0138
Max Privacy Advantage Reduction : -0.0089
Audit Artifacts Generated in    : experiments/audit_mia
===========================================================================
```

---

## 5. `fedhealth dashboard`

Starts the production FastAPI backend and WebSocket telemetry service for real-time visualization and remote client orchestration.

```bash
fedhealth dashboard [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--host` | `string` | `"127.0.0.1"` | Server host IP address |
| `--port` | `int` | `8000` | Server listening port |

### Example
```bash
fedhealth dashboard --host 0.0.0.0 --port 8000
```
