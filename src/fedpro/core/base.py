"""
Core Base Interfaces and Data Transfer Objects (DTOs) for FedHealth.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torch.utils.data import DataLoader

@dataclass
class ClientUpdate:
    """Standardized result packet sent from a local hospital client to the central server."""
    client_id: str
    parameters: Dict[str, torch.Tensor]
    num_samples: int
    metrics: Dict[str, float] = field(default_factory=dict)
    privacy: Dict[str, Any] = field(default_factory=dict)
    grad_norms: Dict[str, float] = field(default_factory=dict)
    computation_time_sec: float = 0.0
    bytes_transferred: int = 0
    simulated_latency_ms: float = 0.0
    is_straggler: bool = False
    control_variate_delta: Optional[Dict[str, torch.Tensor]] = None

@dataclass
class ServerEvaluation:
    """Standardized global model evaluation metrics across validation/test partitions."""
    round_num: int
    loss: float
    accuracy: float
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    specificity: float = 0.0
    confusion_matrix: Optional[List[List[int]]] = None
    evaluation_time_sec: float = 0.0

class BaseFLClient(ABC):
    """Abstract Base Class for Federated Learning Clients / Hospital Nodes."""
    
    def __init__(self, client_id: str, model: nn.Module, device: str = "cpu"):
        self.client_id = client_id
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return model parameters on CPU."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Load global parameters into the local model."""
        self.model.load_state_dict({k: v.to(self.device) for k, v in parameters.items()})

    @abstractmethod
    def fit(self, parameters: Dict[str, torch.Tensor], train_loader: DataLoader, config: Dict[str, Any]) -> ClientUpdate:
        """Execute local training and return a ClientUpdate DTO."""
        pass

    @abstractmethod
    def evaluate(self, parameters: Dict[str, torch.Tensor], test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate local model on validation data."""
        pass

class BaseFLServer(ABC):
    """Abstract Base Class for Central Federated Learning Aggregator."""
    
    def __init__(self, global_model: nn.Module):
        self.global_model = global_model

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}

    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        self.global_model.load_state_dict(parameters)

    @abstractmethod
    def fit_round(
        self, 
        round_num: int, 
        clients: List[BaseFLClient], 
        dataloaders: List[DataLoader], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Orchestrate a single communication round of federated training."""
        pass

    @abstractmethod
    def evaluate_round(
        self, 
        round_num: int, 
        evaluator_client: BaseFLClient, 
        test_loader: DataLoader
    ) -> ServerEvaluation:
        """Evaluate global model performance against test dataset."""
        pass
