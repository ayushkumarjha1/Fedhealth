"""
FedNova: Normalized Averaging for Heterogeneous Federated Learning.

Reference:
Wang, J., Liu, Q., Liang, H., Joshi, G., & Poor, H. V. (2020).
"Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization." NeurIPS.
"""

import torch
from typing import List, Dict
from fedpro.core.base import ClientUpdate

def aggregate_fednova(
    global_params: Dict[str, torch.Tensor],
    client_updates: List[ClientUpdate],
    local_steps: List[int]
) -> Dict[str, torch.Tensor]:
    """
    Computes FedNova normalized averaging over heterogeneous client updates.
    
    Args:
        global_params: Initial global model parameters at start of round
        client_updates: List of client update packets
        local_steps: List of number of local optimization steps taken per client
        
    Returns:
        Aggregated global model parameters dictionary
    """
    if not client_updates:
        raise ValueError("Cannot aggregate empty client updates list.")
        
    total_samples = sum(u.num_samples for u in client_updates)
    weights = [u.num_samples / total_samples for u in client_updates]
    
    # Calculate tau_eff (effective number of steps)
    tau_eff = sum(w * tau for w, tau in zip(weights, local_steps))
    
    # Compute normalized gradient update per client: d_i = (w_0 - w_i) / tau_i
    aggregated_delta: Dict[str, torch.Tensor] = {}
    
    for key, initial_tensor in global_params.items():
        if not initial_tensor.is_floating_point():
            aggregated_delta[key] = initial_tensor.clone()
            continue
            
        accumulated_grad = torch.zeros_like(initial_tensor, dtype=torch.float32)
        for update, p_i, tau_i in zip(client_updates, weights, local_steps):
            tau_i_safe = max(1, tau_i)
            # Local update delta
            client_tensor = update.parameters[key].to(initial_tensor.device)
            local_grad = (initial_tensor.float() - client_tensor.float()) / tau_i_safe
            accumulated_grad += p_i * local_grad
            
        # Global update: w_{t+1} = w_t - tau_eff * accumulated_grad
        aggregated_delta[key] = (initial_tensor.float() - tau_eff * accumulated_grad).type(initial_tensor.dtype)
        
    return aggregated_delta
