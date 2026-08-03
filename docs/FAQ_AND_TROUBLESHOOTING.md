# Frequently Asked Questions (FAQ) & Troubleshooting Guide

---

## 💡 Frequently Asked Questions (FAQ)

### 1. What makes FedHealth different from general-purpose frameworks like Flower or PySyft?
While general-purpose frameworks provide basic communication abstractions, **FedHealth** is tailored specifically for regulated clinical healthcare environments. It includes:
- **Exact Rényi Differential Privacy (RDP) Accounting** evaluated across 26 orders with Poisson subsampling.
- **Empirical Membership Inference Attack (MIA) Auditing** to empirically validate privacy defenses.
- **Clinically Grounded Explainable AI (XAI)** utilizing Cohort-Centroid baselines ($\mu_{\text{benign}}$) rather than non-physical zeros.
- **Digital Twin Hospital Simulation** modeling WAN packet latency, hardware tiers, and straggler node phenomena.
- **Telemetry-Grounded AI Copilot** delivering automated convergence diagnostics and parameter guidance.

---

### 2. Does FedHealth transmit raw patient data to the central server?
**No.** FedHealth strictly adheres to the principle of data minimization and zero-data egress. Only mathematical parameter weights ($w$) or clipped, noisy gradients ($\tilde{g}$) are transmitted across the network. Raw patient medical records never leave local hospital nodes.

---

### 3. Does FedHealth support GPU acceleration?
**Yes.** All PyTorch computations automatically detect CUDA or Apple Silicon (MPS) if available, falling back gracefully to CPU if no discrete GPU is present.

---

### 4. How is the Differential Privacy budget $\epsilon$ determined?
FedHealth tracks privacy loss using **Rényi Differential Privacy (RDP)**:
$$\mathcal{R}_\alpha(\mathcal{M}) = \frac{\alpha}{2\sigma^2}$$
For subsampled Gaussian mechanisms with sampling ratio $q = |B|/N$, RDP orders $\alpha \in [1.25, 128.0]$ are tracked across all communication rounds and converted to $(\epsilon, \delta)$-DP via convex optimization:
$$\epsilon(\delta) = \min_{\alpha > 1} \left( \sum_{t=1}^T \mathcal{R}_\alpha^{(t)} + \frac{\ln(1/\delta)}{\alpha - 1} \right)$$

---

### 5. Why use a Cohort-Centroid baseline for Integrated Gradients in healthcare?
In standard computer vision, a black image ($x'=0$) is an intuitive baseline. In clinical healthcare, setting physiological biomarkers (such as blood pressure, glucose, or cell radius) to zero represents a non-physical, dead reference state. FedHealth computes $\mu_{\text{benign}} = \mathbb{E}[x \mid y=0]$ as the baseline, ensuring attributions represent true pathological deviations from healthy patients.

---

## 🛠️ Troubleshooting & Common Issues

### 1. Port Conflict on Dashboard Launch (`Address already in use: 8000`)
**Symptom**: `uvicorn.run` fails with `[Errno 98] Address already in use: 8000`.  
**Solution**: Specify an alternative listening port via the CLI:
```bash
fedhealth dashboard --port 8080
```

---

### 2. Training Divergence under Severe Non-IID Skew ($\alpha \le 0.1$)
**Symptom**: Global test accuracy plateaus or oscillates wildly under extreme label distribution skew.  
**Solution**: Switch from baseline `fedavg` to **`fedprox`** or **`scaffold`**:
```bash
# Enable proximal regularization (mu=0.01 to 0.1)
fedhealth run --algo fedprox --rounds 15 --hospitals 5
```

---

### 3. Integrated Gradients Completeness Residual Exceeds Threshold
**Symptom**: `completeness_delta` is larger than $0.05$.  
**Solution**: Increase the Riemann sum path integration steps from 50 to 100 in the explainer:
```python
explainer.compute_integrated_gradients(sample_tensor, steps=100)
```

---

### 4. High Privacy Budget Depletion ($\epsilon > 20$)
**Symptom**: Privacy expenditure $\epsilon$ grows too quickly after multiple rounds.  
**Solution**:
1. Increase Gaussian noise multiplier: `--noise 0.8` or `1.0`.
2. Tighten $L_2$ gradient clipping threshold: `--clip 0.5`.
3. Increase local hospital dataset size or batch size to reduce sampling ratio $q = |B|/N$.

---

### 5. WebSocket Reconnection Dropouts in High-Latency Simulations
**Symptom**: Dashboard intermittently disconnects during rounds with simulated high WAN jitter.  
**Solution**: The React 19 dashboard includes built-in exponential backoff reconnection logic (reconnects every 3s). Ensure the FastAPI server is running on `127.0.0.1:8000`.
