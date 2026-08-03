"""
Federated Averaging (FedAvg) Algorithm.

Reference:
McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).
"Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS.
"""

import torch
from typing import List, Tuple, Dict
from fedpro.core.base import ClientUpdate

def aggregate_fedavg(client_updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
    """
    Computes sample-weighted average of client model parameters:
    w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_k^{t+1}
    
    Args:
        client_updates: List of ClientUpdate objects from participating clients
        
    Returns:
        Aggregated global model parameters dictionary
    """
    if not client_updates:
        raise ValueError("Cannot aggregate empty client updates list.")
        
    total_samples = sum(update.num_samples for update in client_updates)
    if total_samples <= 0:
        raise ValueError("Total sample count across clients must be greater than 0.")
        
    first_weights = client_updates[0].parameters
    first_weighting = client_updates[0].num_samples / total_samples
    
    averaged_weights: Dict[str, torch.Tensor] = {}
    for key, tensor in first_weights.items():
        averaged_weights[key] = tensor.float().clone() * first_weighting
        
    for update in client_updates[1:]:
        weighting = update.num_samples / total_samples
        for key, tensor in update.parameters.items():
            averaged_weights[key] += tensor.float() * weighting
            
    return averaged_weights
