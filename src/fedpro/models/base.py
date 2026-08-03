"""
Base Neural Network Model interface and model registry for FedHealth.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Type

class BaseMedicalModel(nn.Module, ABC):
    """Abstract base class for all healthcare diagnostic models."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def get_num_parameters(self) -> int:
        """Count total trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """Estimate model weights size in megabytes."""
        bytes_total = sum(p.numel() * p.element_size() for p in self.parameters())
        return bytes_total / (1024 * 1024)
