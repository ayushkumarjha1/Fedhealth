"""
FedOpt: Server-Side Adaptive Optimizers (FedAdam, FedYogi, FedAvgM).

Reference:
Reddi, S., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., Kumar, S., & McMahan, H. B. (2021).
"Adaptive Federated Optimization." ICLR.
"""

import torch
from typing import List, Dict, Literal
from fedpro.core.base import ClientUpdate

class FedOptServerOptimizer:
    """
    Server-side optimizer updating global model parameters using aggregated pseudo-gradients.
    Supports FedAdam, FedYogi, and FedAvgM.
    """
    
    def __init__(
        self,
        mode: Literal["fedadam", "fedyogi", "fedavgm"] = "fedadam",
        lr: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3
    ):
        self.mode = mode
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        
        # State buffers for first and second moments
        self.m: Dict[str, torch.Tensor] = {}
        self.v: Dict[str, torch.Tensor] = {}

    def step(
        self, 
        global_params: Dict[str, torch.Tensor], 
        client_updates: List[ClientUpdate]
    ) -> Dict[str, torch.Tensor]:
        """
        Execute adaptive server update step.
        
        Args:
            global_params: Current global model parameters w_t
            client_updates: Participating client updates w_{t,i}
            
        Returns:
            Updated global model parameters w_{t+1}
        """
        total_samples = sum(u.num_samples for u in client_updates)
        weights = [u.num_samples / total_samples for u in client_updates]
        
        # 1. Compute weighted average of client parameters: \bar{w} = \sum p_i w_{t,i}
        bar_w: Dict[str, torch.Tensor] = {}
        for key, tensor in global_params.items():
            if not tensor.is_floating_point():
                bar_w[key] = tensor.clone()
                continue
            bar_w[key] = torch.zeros_like(tensor, dtype=torch.float32)
            for update, p_i in zip(client_updates, weights):
                bar_w[key] += p_i * update.parameters[key].float()
                
        # 2. Compute Pseudo-Gradient: \Delta_t = w_t - \bar{w}
        updated_params: Dict[str, torch.Tensor] = {}
        
        for key, current_val in global_params.items():
            if not current_val.is_floating_point():
                updated_params[key] = current_val.clone()
                continue
                
            delta = current_val.float() - bar_w[key]
            
            # Initialize momentum buffers if first step
            if key not in self.m:
                self.m[key] = torch.zeros_like(delta)
                self.v[key] = torch.full_like(delta, fill_value=self.tau ** 2)
                
            # Update 1st moment: m_t = beta1 * m_{t-1} + (1 - beta1) * delta
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * delta
            
            if self.mode == "fedadam":
                # FedAdam 2nd moment: v_t = beta2 * v_{t-1} + (1 - beta2) * delta^2
                self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (delta ** 2)
                step_size = self.lr * self.m[key] / (torch.sqrt(self.v[key]) + self.tau)
            elif self.mode == "fedyogi":
                # FedYogi 2nd moment
                diff = delta ** 2 - self.v[key]
                self.v[key] = self.v[key] + (1.0 - self.beta2) * (delta ** 2) * torch.sign(diff)
                step_size = self.lr * self.m[key] / (torch.sqrt(self.v[key]) + self.tau)
            else:
                # FedAvgM (Momentum only)
                step_size = self.lr * self.m[key]
                
            # Apply update: w_{t+1} = w_t - step_size
            updated_params[key] = (current_val.float() - step_size).type(current_val.dtype)
            
        return updated_params
