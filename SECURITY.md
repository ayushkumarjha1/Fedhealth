# Security Policy & Vulnerability Disclosure

## Supported Versions

The following versions of FedHealth are currently supported with security updates:

| Version | Supported          |
| :------ | :----------------- |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## Threat Model & Security Scope

FedHealth is designed for collaborative, privacy-preserving machine learning in clinical and institutional research environments.

### In-Scope Security Vectors
1. **Differential Privacy Budget Accounting**: Flaws in the subsampled Gaussian Rényi Differential Privacy (RDP) numerical compositions that could lead to privacy budget underestimation.
2. **Local Gradient & Parameter Sanitization**: Bypasses in per-sample clipping ($L_2$ norm enforcement) or Gaussian noise calibration.
3. **Empirical Privacy Leakage**: Regressions causing unexpected susceptibility to Membership Inference Attacks (MIA) under standard query bounds.
4. **API & Telemetry Security**: Remote code execution or unauthorized state mutation through FastAPI endpoints or WebSocket message handlers.
5. **Safe Deserialization**: Insecure checkpoint loading (enforcing PyTorch `weights_only=True` parameter serialization).

### Out-of-Scope Security Vectors
1. **Physical / Root Node Compromise**: Attacks requiring root access to an individual hospital's local compute hardware.
2. **Network Transport Security without TLS**: FedHealth core relies on mTLS/TLS 1.3 at the transport layer for production WAN deployments.

---

## Reporting a Vulnerability

We take the security of healthcare AI systems seriously. If you discover a potential vulnerability, please do **NOT** open a public GitHub issue.

Instead, please report security issues through one of the following channels:
1. **Security Email**: `security@fedhealth-project.org`
2. **Encrypted Submission**: Open a private advisory on GitHub via the **Security** tab -> **Report a vulnerability**.

### Disclosure Timeline
- **Initial Acknowledgment**: Within 24 hours.
- **Triage & Severity Assessment**: Within 72 hours.
- **Patch & Public Advisory Release**: Typically within 14 calendar days depending on vulnerability complexity.
