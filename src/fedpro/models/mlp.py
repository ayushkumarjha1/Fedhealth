"""
Healthcare Multi-Layer Perceptron (MLP) for Tabular Clinical Data.
Supports configurable hidden dimensions, Batch Normalization, and Dropout.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from fedpro.models.base import BaseMedicalModel

class HealthcareMLP(BaseMedicalModel):
    """Deep Feed-Forward Network designed for clinical biomarker diagnostics."""
    
    def __init__(
        self, 
        input_dim: int = 30, 
        hidden_dims: Optional[List[int]] = None, 
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        dims = hidden_dims or [64, 32, 16]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
