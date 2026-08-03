"""
Concrete Healthcare FL Client implementation with DP-SGD and local metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import time

from fedpro.core.base import BaseFLClient, ClientUpdate
from fedpro.privacy.privacy_engine import PrivacyEngine
from fedpro.configs.base_config import DPConfig

class FLClient(BaseFLClient):
    """Standard Healthcare Federated Learning Client with PrivacyEngine integration."""
    
    def __init__(self, client_id: str, model: nn.Module, device: str = "cpu", dp_config: Optional[DPConfig] = None):
        super().__init__(client_id, model, device)
        self.dp_config = dp_config or DPConfig(enabled=False)
        self.privacy_engine: Optional[PrivacyEngine] = None

    def fit(self, parameters: Dict[str, torch.Tensor], train_loader: DataLoader, config: Dict[str, Any]) -> ClientUpdate:
        start_time = time.time()
        self.set_parameters(parameters)
        self.model.train()
        
        epochs = config.get("epochs", 2)
        lr = config.get("lr", 0.01)
        momentum = config.get("momentum", 0.9)
        weight_decay = config.get("weight_decay", 1e-4)
        
        if self.privacy_engine is None:
            self.privacy_engine = PrivacyEngine(
                config=self.dp_config,
                dataset_size=len(train_loader.dataset),
                batch_size=train_loader.batch_size or 16
            )
            
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        total_loss = 0.0
        num_samples = 0
        total_steps = 0
        last_grad_norms = {"pre_clip_norm": 0.0, "post_noise_norm": 0.0}
        
        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Apply Differential Privacy
                if self.privacy_engine.is_enabled:
                    last_grad_norms = self.privacy_engine.sanitize_gradients(self.model, self.device)
                    
                optimizer.step()
                total_steps += 1
                total_loss += loss.item() * data.size(0)
                num_samples += data.size(0)
                
        if self.privacy_engine.is_enabled:
            self.privacy_engine.record_step(num_steps=total_steps)
            
        training_time = time.time() - start_time
        avg_loss = total_loss / max(1, num_samples)
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
