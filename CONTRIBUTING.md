# Contributing to FedHealth

Welcome to the **FedHealth Open-Source Project**! We are building the next-generation, research-grade federated learning framework for privacy-preserving clinical healthcare. We welcome contributions from machine learning engineers, distributed systems researchers, privacy scientists, clinicians, and software developers worldwide.

---

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Areas Where You Can Contribute](#areas-where-you-can-contribute)
3. [Development Environment Setup](#development-environment-setup)
4. [Repository Architecture & Layout](#repository-architecture--layout)
5. [How to Add a New Federated Algorithm](#how-to-add-a-new-federated-algorithm)
6. [Coding Standards & Mathematical Verification](#coding-standards--mathematical-verification)
7. [Commit Conventions & Git Workflow](#commit-conventions--git-workflow)
8. [Pull Request Review Checklist](#pull-request-review-checklist)

---

## 1. Code of Conduct
All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat all community members with empathy, respect, and professional rigor.

---

## 2. Areas Where You Can Contribute
- **Optimization Algorithms**: Implement novel aggregation schemes or local regularization methods (e.g. MOON, FedDyn, FedProto).
- **Privacy & Security**: Expand privacy accounting (e.g. Gaussian Differential Privacy, f-DP) or develop new empirical privacy attacks (e.g. reconstruction/inversion attacks).
- **Explainable AI (XAI)**: Implement counterfactual explanations, Concept Activation Vectors (TCAV), or SHAP integration for clinical pipelines.
- **Clinical Benchmarks**: Add support for standard medical datasets (e.g. MIMIC-III, CheXpert, ISIC Skin Cancer).
- **Telemetry & UI**: Enhance the React 19 glassmorphic dashboard, add time-series visualizations, or optimize WebSocket streaming.
- **Documentation & Tutorials**: Improve educational examples, write Viva preparation guides, or translate documentation.

---

## 3. Development Environment Setup

### Prerequisites
- Python 3.10+ (Python 3.11/3.12 recommended)
- Git
- Node.js 18+ & npm (optional, for dashboard development)

### Quick Setup Steps
```bash
# 1. Clone your fork
git clone https://github.com/<your-username>/fedhealth.git
cd fedhealth

# 2. Create and activate a clean virtual environment
python -m venv .venv
# On Linux / macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# 3. Install in editable mode with development dependencies
pip install -e .

# 4. Verify test suite passes
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 4. Repository Architecture & Layout

```
FedPro/
├── src/fedpro/
│   ├── algorithms/     # Federated algorithms (FedAvg, FedProx, SCAFFOLD, FedNova, FedOpt)
│   ├── api/            # FastAPI backend & WebSocket telemetry broadcaster
│   ├── cli/            # FedHealth unified CLI entrypoint (`fedhealth`)
│   ├── configs/        # Pydantic V2 validated typed configuration schemas
│   ├── copilot/        # Telemetry-grounded AI Federated Diagnostic Copilot
│   ├── core/           # FLServer, BaseFLClient, Communication Protocol, Tracker
│   ├── data/           # Medical datasets (Breast Cancer, Synthetics) & Dirichlet partitioners
│   ├── models/         # PyTorch neural architectures (HealthcareMLP, CNNs)
│   ├── privacy/        # DP-SGD mechanism, Rényi DP accountant, MIA Evaluator
│   ├── simulation/     # Digital Twin hospital node infrastructure & network physics
│   ├── utils/          # Logging, math utilities, serialization
│   └── xai/            # Clinical Integrated Gradients & Cohort-Centroid Explainer
├── tests/              # 25 automated mathematical invariant & property tests
├── dashboard/          # React 19 + TypeScript + Vite live analytics frontend
├── docs/               # Full research reports, mathematical proofs, viva & portfolio kits
└── pyproject.toml      # Modern PEP 621 packaging specification
```

---

## 5. How to Add a New Federated Algorithm

1. **Create the Client Implementation**:
   Create a new file in `src/fedpro/algorithms/<algo_name>.py` and subclass `BaseFLClient`:
   ```python
   from fedpro.core.client import BaseFLClient
   from fedpro.algorithms.registry import register_algorithm

   @register_algorithm("<algo_name>")
   class CustomFLClient(BaseFLClient):
       def train_epoch(self, dataloader, optimizer, criterion):
           # Custom local training logic (e.g. proximal loss, drift correction)
           pass
   ```

2. **Register Algorithm in Registry**:
   Ensure your algorithm is mapped in `src/fedpro/algorithms/registry.py`.

3. **Add Mathematical Invariant Test**:
   Create a test in `tests/test_algorithms_math.py` asserting the theoretical property of your algorithm (e.g. convergence under variance, loss penalty correctness).

4. **Verify All Tests**:
   ```bash
   python -m unittest tests/test_algorithms_math.py -v
   ```

---

## 6. Coding Standards & Mathematical Verification

- **Strict Type Annotations**: All public functions and class methods must use standard Python type hints (`Optional`, `Union`, `List`, `Dict`, `Tuple`).
- **Mathematical Accuracy**: All formulas implemented in code must correspond directly to peer-reviewed literature and be documented in both docstrings and `docs/`.
- **Axiomatic Soundness**: When contributing XAI or Privacy features, you must write automated assertions verifying formal invariants (e.g. Axiom of Completeness, RDP monotonicity).
- **Clean Code**: Zero warnings on execution; adhere to PEP 8 standards with descriptive variable naming.

---

## 7. Commit Conventions & Git Workflow

We adhere to the [Conventional Commits](https://www.conventionalcommits.org/) specification:

| Prefix | Usage | Example |
| :--- | :--- | :--- |
| `feat:` | A new user-facing feature or CLI subcommand | `feat(xai): add cohort-centroid baseline support` |
| `fix:` | A bug fix in algorithms, privacy, or API | `fix(dp): resolve device mapping in gradient noise injection` |
| `math:` | Mathematical refinement or analytical proof update | `math(rdp): tighten order bounds for subsampled Gaussian` |
| `docs:` | Documentation, tutorials, or academic reports | `docs(viva): add 15 distributed systems examination questions` |
| `test:` | Adding or upgrading unit and regression tests | `test(mia): add loss-threshold attack advantage test` |
| `refactor:` | Code restructuring without altering external behavior | `refactor(core): streamline server aggregation dispatch` |

---

## 8. Pull Request Review Checklist

Before submitting your Pull Request, ensure that:
- [ ] Your code passes all existing tests: `python -m unittest discover -s tests -p "test_*.py" -v`.
- [ ] You have added automated tests covering any new functionality or edge cases.
- [ ] You have added or updated relevant documentation in `docs/` and docstrings.
- [ ] Your branch is rebased onto the latest `main` branch.
- [ ] You have verified that `fedhealth --help` runs without errors.
