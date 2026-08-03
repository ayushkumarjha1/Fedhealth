"""
AI Federated Copilot Engine for FedHealth.
Analyzes live training telemetry using rigorous quantitative metrics:
1. Inter-client parameter drift via pairwise cosine similarity matrices
2. Gradient Signal-to-Noise Ratio (SNR) across hospital cohorts
3. Differential privacy budget expenditure velocity (d_eps / dt)
4. Hardware latency coefficient of variation (CV) for straggler detection

All diagnostic insights and recommendations are grounded in statistical telemetry.
"""

import math
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple

class AIFederatedCopilot:
    """
    Scientifically grounded telemetry analyzer providing verifiable diagnostics 
    and hyperparameter optimization recommendations.
    """
    
    def __init__(self, target_privacy_epsilon: float = 10.0):
        self.target_privacy_epsilon = float(target_privacy_epsilon)
        self.history: List[Dict[str, Any]] = []

    def get_latest_insight(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent copilot evaluation report."""
        return self.history[-1] if self.history else None

    @property
    def insights_history(self) -> List[Dict[str, Any]]:
        """Retrieve complete chronological list of copilot insights."""
        return self.history

    def compute_client_drift_matrix(
        self, 
        client_deltas: List[Dict[str, torch.Tensor]]
    ) -> Tuple[np.ndarray, float]:
        """
        Computes pairwise cosine similarity matrix S_ij between hospital parameter updates:
        S_ij = <delta_w_i, delta_w_j> / (||delta_w_i|| * ||delta_w_j||)
        
        Returns:
            Tuple of (similarity_matrix, mean_off_diagonal_cosine_similarity)
        """
        n = len(client_deltas)
        if n <= 1:
            return np.ones((1, 1), dtype=np.float32), 1.0
            
        # Flatten each client update into a single 1D vector
        flat_vectors = []
        for delta in client_deltas:
            tensors = [v.detach().cpu().float().view(-1) for v in delta.values()]
            flat_vectors.append(torch.cat(tensors))
            
        stacked = torch.stack(flat_vectors) # Shape: (n, D)
        norms = torch.norm(stacked, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        normalized = stacked / norms
        
        # Cosine similarity matrix S = normalized * normalized^T
        similarity_matrix = torch.mm(normalized, normalized.t()).numpy()
        
        # Compute mean of off-diagonal elements
        mask = ~np.eye(n, dtype=bool)
        mean_similarity = float(similarity_matrix[mask].mean()) if mask.sum() > 0 else 1.0
        
        return similarity_matrix, mean_similarity

    def compute_gradient_snr(
        self, 
        client_deltas: List[Dict[str, torch.Tensor]]
    ) -> float:
        """
        Calculates empirical Signal-to-Noise Ratio (SNR) across client updates:
        SNR = ||mean(delta_w)||_2 / sqrt(var(delta_w) + eps)
        """
        if not client_deltas:
            return 0.0
            
        flat_vectors = []
        for delta in client_deltas:
            tensors = [v.detach().cpu().float().view(-1) for v in delta.values()]
            flat_vectors.append(torch.cat(tensors))
            
        stacked = torch.stack(flat_vectors) # (n, D)
        mean_vector = torch.mean(stacked, dim=0)
        signal_norm = float(torch.norm(mean_vector, p=2).item())
        
        # Compute variance across clients
        variance = torch.var(stacked, dim=0, unbiased=True).sum().item()
        noise_norm = math.sqrt(max(1e-12, variance))
        
        return signal_norm / max(1e-6, noise_norm)

    def analyze_round(
        self,
        round_num: int,
        metrics_history: List[Dict[str, Any]],
        hospital_updates: List[Dict[str, Any]],
        privacy_metrics: Dict[str, Any],
        algorithm_name: str,
        drift_similarity: Optional[float] = None,
        gradient_snr: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates round telemetry and produces grounded clinical & engineering insights.
        """
        if not metrics_history:
            return {"round": round_num, "status": "INITIALIZING", "insights": [], "recommendations": []}
            
        current = metrics_history[-1]
        prev = metrics_history[-2] if len(metrics_history) >= 2 else None
        
        curr_loss = current.get("loss", 1.0)
        curr_acc = current.get("accuracy", 50.0)
        curr_eps = privacy_metrics.get("epsilon", 0.0)
        is_dp = privacy_metrics.get("enabled", False)
        
        insights = []
        recommendations = []
        status_category = "OPTIMAL"
        
        # 1. Convergence Trajectory
        if prev:
            prev_acc = prev.get("accuracy", 50.0)
            prev_loss = prev.get("loss", 1.0)
            acc_delta = curr_acc - prev_acc
            loss_delta = curr_loss - prev_loss
            
            if acc_delta > 1.0:
                insights.append(
                    f"Positive generalization observed: Global accuracy improved by +{acc_delta:.2f}% "
                    f"with loss reduction of {abs(loss_delta):.4f} under {algorithm_name.upper()}."
                )
            elif acc_delta < -1.5:
                status_category = "WARNING"
                insights.append(
                    f"Performance variance detected: Global accuracy decreased by {abs(acc_delta):.2f}% "
                    f"(loss shift: {loss_delta:+.4f})."
                )
                if is_dp:
                    insights.append(
                        f"[Privacy Impact]: DP-SGD Gaussian perturbation (sigma={privacy_metrics.get('noise_multiplier', 0.5)}) "
                        f"contributes to empirical stochastic gradient variance."
                    )
                else:
                    insights.append(
                        f"[Optimization Drift]: Local gradient updates appear to diverge across non-IID hospital cohorts."
                    )

        # 2. Quantitative Client Drift Analysis (Cosine Similarity)
        if drift_similarity is not None:
            if drift_similarity >= 0.70:
                insights.append(
                    f"Inter-hospital gradient alignment is strong (mean cosine similarity S = {drift_similarity:.3f} >= 0.70). "
                    f"Cohorts exhibit compatible optimization directions."
                )
            elif 0.30 <= drift_similarity < 0.70:
                insights.append(
                    f"Moderate client drift observed (mean cosine similarity S = {drift_similarity:.3f}). "
                    f"Heterogeneous patient distributions are causing directional divergence in local weights."
                )
                if algorithm_name.lower() == "fedavg":
                    recommendations.append(
                        "Consider switching to FedProx (mu in [0.001, 0.01]) or SCAFFOLD to counteract client drift."
                    )
            else:
                status_category = "CRITICAL"
                insights.append(
                    f"Severe client drift detected (mean cosine similarity S = {drift_similarity:.3f} < 0.30). "
                    f"Local hospital updates are orthogonal or conflicting."
                )
                recommendations.append(
                    "Switch optimizer to SCAFFOLD (control variates) or reduce local training epochs to prevent catastrophic client drift."
                )

        # 3. Gradient Signal-to-Noise Ratio
        if gradient_snr is not None:
            if gradient_snr < 0.5 and is_dp:
                status_category = "WARNING"
                insights.append(
                    f"Low Gradient Signal-to-Noise Ratio (SNR = {gradient_snr:.3f} < 0.50). "
                    f"DP noise dominates the aggregated update magnitude."
                )
                recommendations.append(
                    "Increase local batch size or decrease DP noise multiplier to improve gradient SNR."
                )

        # 4. Differential Privacy Budget Expenditure
        if is_dp and curr_eps > 0.0:
            spent_ratio = curr_eps / max(0.1, self.target_privacy_epsilon)
            
            # Compute privacy velocity
            prev_eps = metrics_history[-2].get("epsilon", 0.0) if prev and "epsilon" in prev else 0.0
            eps_velocity = max(0.0, curr_eps - prev_eps)
            
            insights.append(
                f"Privacy budget expenditure: eps = {curr_eps:.2f} (Target cap: {self.target_privacy_epsilon}, "
                f"{spent_ratio*100:.1f}% consumed). Velocity: +{eps_velocity:.2f} eps/round."
            )
            
            if spent_ratio >= 0.85:
                status_category = "CRITICAL"
                insights.append("Privacy budget threshold approaching exhaustion (>85%).")
                recommendations.append(
                    "Prepare for early stopping to satisfy Institutional Review Board (IRB) differential privacy bounds."
                )

        # 5. Straggler Latency Coefficient of Variation
        latencies = [h.get("latency_ms", 0.0) for h in hospital_updates if "latency_ms" in h]
        if len(latencies) > 1:
            mean_lat = np.mean(latencies)
            std_lat = np.std(latencies)
            cv_lat = std_lat / max(1e-6, mean_lat)
            if cv_lat > 0.50:
                slowest = max(hospital_updates, key=lambda x: x.get("latency_ms", 0.0))
                insights.append(
                    f"System latency asymmetry detected (CV = {cv_lat:.2f} > 0.50). "
                    f"Straggler node: '{slowest.get('name', 'Unknown')}' with {slowest.get('latency_ms', 0.0):.1f} ms."
                )
                recommendations.append(
                    f"Configure asynchronous aggregation or lower local epochs on straggler node '{slowest.get('name', '')}'."
                )

        summary_text = insights[0] if insights else "Round telemetry processed within normal bounds."
        
        report = {
            "round": round_num,
            "category": status_category,
            "summary": summary_text,
            "all_insights": insights,
            "recommendations": recommendations,
            "metrics": {
                "drift_similarity": drift_similarity,
                "gradient_snr": gradient_snr,
                "loss": curr_loss,
                "accuracy": curr_acc,
                "epsilon": curr_eps
            }
        }
        self.history.append(report)
        return report
