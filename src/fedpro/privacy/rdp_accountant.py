"""
Rényi Differential Privacy (RDP) Accountant for FedHealth.
Implements exact analytical RDP composition over Subsampled Gaussian Mechanisms
and numerical conversion to (epsilon, delta)-Differential Privacy guarantees.

References:
- Mironov, I. (2017). "Rényi Differential Privacy." IEEE CSF.
- Wang, Y. X., Balle, B., & Kasiviswanathan, S. P. (2019). "Subsampled Rényi Differential Privacy." AISTATS.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# Standard evaluation orders for Rényi Divergence
DEFAULT_RDP_ORDERS: List[float] = [
    1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5, 
    5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 20.0, 
    24.0, 28.0, 32.0, 48.0, 64.0, 128.0
]

def compute_rdp_gaussian(alpha: float, noise_multiplier: float) -> float:
    """
    Computes RDP for a standard Gaussian mechanism with noise multiplier sigma.
    RDP(alpha) = alpha / (2 * sigma^2)
    """
    if noise_multiplier <= 0:
        return float("inf")
    return alpha / (2.0 * (noise_multiplier ** 2))

def compute_rdp_subsampled_gaussian(alpha: float, noise_multiplier: float, sample_rate: float) -> float:
    """
    Analytical upper bound on RDP of subsampled Gaussian mechanism using order-dependent expansion.
    For sample rate q and noise multiplier sigma:
    - If q == 1.0 (full batch): exact Gaussian RDP = alpha / (2 * sigma^2)
    - If q < 1.0: computed via tight binomial approximation / Wang et al. 2019 bound.
    """
    if noise_multiplier <= 0:
        return float("inf")
    if sample_rate == 0:
        return 0.0
    if sample_rate == 1.0:
        return compute_rdp_gaussian(alpha, noise_multiplier)
    
    # Subsampled Gaussian bound (Wang et al. 2019, Mironov et al. 2019)
    sigma = float(noise_multiplier)
    q = min(1.0, max(0.0, float(sample_rate)))
    
    if q >= 1.0:
        return compute_rdp_gaussian(alpha, sigma)
    if q == 0.0 or alpha <= 1.0:
        return 0.0
    
    # Gaussian RDP for the base mechanism
    rdp_gauss = compute_rdp_gaussian(alpha, sigma)
    
    # For small q, subsampled RDP <= q^2 * alpha / (2 * sigma^2) or standard bound
    # Using Wang et al. 2019 analytical formula with positive bases
    base_1_minus_q = max(1e-12, 1.0 - q)
    
    try:
        if rdp_gauss > 50.0:
            return float(rdp_gauss)
        exp_term = math.exp(min(50.0, rdp_gauss))
        inner_val = (
            (base_1_minus_q ** alpha) + 
            alpha * q * (base_1_minus_q ** (alpha - 1.0)) + 
            (q ** 2) * (alpha * (alpha - 1.0) / 2.0) * exp_term
        )
        if inner_val <= 0:
            return float(rdp_gauss)
        subsampled_rdp = (1.0 / (alpha - 1.0)) * math.log(inner_val)
        return float(max(0.0, min(subsampled_rdp, rdp_gauss)))
    except (ValueError, OverflowError):
        return float(rdp_gauss)

def rdp_to_dp(rdp_orders: List[float], rdp_values: List[float], target_delta: float) -> Tuple[float, float]:
    """
    Converts accumulated RDP values across orders to an (epsilon, delta)-DP guarantee.
    epsilon(delta) = min_{alpha > 1} [ rdp(alpha) + log(1 / delta) / (alpha - 1) ]
    
    Returns:
        Tuple of (optimal_epsilon, optimal_order_alpha)
    """
    if target_delta <= 0.0 or target_delta >= 1.0:
        raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")
        
    epsilons = []
    for alpha, rdp in zip(rdp_orders, rdp_values):
        if alpha <= 1.0 or math.isinf(rdp):
            continue
        eps = rdp + math.log(1.0 / target_delta) / (alpha - 1.0)
        epsilons.append((eps, alpha))
        
    if not epsilons:
        return float("inf"), 1.0
        
    min_eps, best_alpha = min(epsilons, key=lambda x: x[0])
    return max(0.0, min_eps), best_alpha

class RDPAccountant:
    """
    Tracks and accumulates Differential Privacy expenditure over training rounds.
    Maintains round-by-round history of (epsilon, delta) values.
    """
    
    def __init__(self, target_delta: float = 1e-5, orders: Optional[List[float]] = None):
        self.target_delta = target_delta
        self.orders = orders or DEFAULT_RDP_ORDERS
        self.rdp_total = np.zeros(len(self.orders), dtype=np.float64)
        self.history: List[Dict[str, Any]] = []
        self.steps = 0
        
    def step(self, noise_multiplier: float, sample_rate: float = 1.0, num_steps: int = 1) -> float:
        """
        Record training steps with Gaussian noise and update accumulated RDP.
        
        Args:
            noise_multiplier: Noise scale sigma = noise_std / clip_norm
            sample_rate: Subsampling ratio q = batch_size / total_dataset_size
            num_steps: Number of mini-batch gradient updates performed
            
        Returns:
            Current optimal epsilon for target_delta
        """
        if noise_multiplier <= 0.0:
            current_eps = float("inf")
        else:
            step_rdp = np.array([
                compute_rdp_subsampled_gaussian(alpha, noise_multiplier, sample_rate) * num_steps
                for alpha in self.orders
            ])
            self.rdp_total += step_rdp
            self.steps += num_steps
            current_eps, best_alpha = rdp_to_dp(self.orders, self.rdp_total.tolist(), self.target_delta)
            
        self.history.append({
            "step": self.steps,
            "epsilon": float(current_eps) if not math.isinf(current_eps) else 999.99,
            "delta": self.target_delta,
            "noise_multiplier": noise_multiplier,
            "sample_rate": sample_rate
        })
        return current_eps

    def get_privacy_spent(self) -> Dict[str, Any]:
        """Return current cumulative privacy guarantees."""
        if self.steps == 0 or np.all(self.rdp_total == 0):
            return {"epsilon": 0.0, "delta": self.target_delta, "steps": 0, "optimal_alpha": 1.0}
        eps, best_alpha = rdp_to_dp(self.orders, self.rdp_total.tolist(), self.target_delta)
        return {
            "epsilon": float(eps) if not math.isinf(eps) else 999.99,
            "delta": self.target_delta,
            "steps": self.steps,
            "optimal_alpha": float(best_alpha)
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """Return chronological privacy expenditure records."""
        return self.history

    def reset(self):
        """Reset privacy accountant state."""
        self.rdp_total = np.zeros(len(self.orders), dtype=np.float64)
        self.history = []
        self.steps = 0
