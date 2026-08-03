"""
Federated Proximal (FedProx) Algorithm.

Reference:
Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
"Federated Optimization in Heterogeneous Networks." MLSys.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
import time

from fedpro.core.base import BaseFLClient, ClientUpdate
from fedpro.privacy.privacy_engine import PrivacyEngine
from fedpro.configs.base_config import DPConfig

class FedProxClient(BaseFLClient):
    """
    Hospital client executing FedProx local optimization with proximal regularization term:
    \min_w F_k(w) + \frac{\mu}{2} \|w - w^t\|^2
    """
    
    def __init__(self, client_id: str, model: nn.Module, device: str = "cpu", dp_config: Optional[DPConfig] = None):
        super().__init__(client_id, model, device)
        self.dp_config = dp_config or DPConfig(enabled=False)
        self.privacy_engine: Optional[PrivacyEngine] = None

    def fit(self, parameters: Dict[str, torch.Tensor], train_loader: DataLoader, config: Dict[str, Any]) -> ClientUpdate:
        start_time = time.time()
        self.set_parameters(parameters)
        self.model.train()
        
        # Keep disconnected copy of global parameters for proximal regularization
        global_params = [p.detach().clone().to(self.device) for p in self.model.parameters()]
        
        epochs = config.get("epochs", 2)
        lr = config.get("lr", 0.01)
        mu = config.get("mu", 0.01) # Proximal parameter
        
        # Initialize Privacy Engine if needed
        if self.privacy_engine is None:
            self.privacy_engine = PrivacyEngine(
                config=self.dp_config,
                dataset_size=len(train_loader.dataset),
                batch_size=train_loader.batch_size or 16
            )
            
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=config.get("momentum", 0.9))
        
        total_loss = 0.0
        num_samples = 0
        total_steps = 0
        last_grad_norms = {"pre_clip_norm": 0.0, "post_noise_norm": 0.0}
        
        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                
                # Standard Empirical Loss
                loss = criterion(output, target)
                
                # Proximal Penalty Term: (mu / 2) * \sum ||w - w^t||^2
                if mu > 0.0:
                    proximal_term = 0.0
                    for local_p, global_p in zip(self.model.parameters(), global_params):
                        proximal_term += (local_p - global_p).norm(2) ** 2
                    loss += (mu / 2.0) * proximal_term
                    
                loss.backward()
                
                # Apply Differential Privacy if enabled
                if self.privacy_engine.is_enabled:
                    last_grad_norms = self.privacy_engine.sanitize_gradients(self.model, self.device)
                    
                optimizer.step()
                total_steps += 1
                
                total_loss += loss.item() * data.size(0)
                num_samples += data.size(0)
                
        # Record DP steps in RDP accountant
        if self.privacy_engine.is_enabled:
            self.privacy_engine.record_step(num_steps=total_steps)
            
        training_time = time.time() - start_time
        avg_loss = total_loss / max(1, num_samples)
        
        # Calculate approximate model payload bytes
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        
        return ClientUpdate(
            client_id=self.client_id,
            parameters=self.get_parameters(),
            num_samples=num_samples,
            metrics={"loss": avg_loss},
            privacy=self.privacy_engine.get_privacy_metrics(),
            grad_norms=last_grad_norms,
            computation_time_sec=training_time,
            bytes_transferred=param_bytes
        )

    def evaluate(self, parameters: Dict[str, torch.Tensor], test_loader: DataLoader) -> Dict[str, float]:
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        num_samples = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += criterion(output, target).item() * data.size(0)
                preds = output.argmax(dim=1)
                correct += preds.eq(target).sum().item()
                num_samples += data.size(0)
                
        return {
            "loss": total_loss / max(1, num_samples),
            "accuracy": correct / max(1, num_samples)
        }
