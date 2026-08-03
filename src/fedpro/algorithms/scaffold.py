"""
SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.

Reference:
Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S., & Suresh, A. T. (2020).
"SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." ICML.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
import time

from fedpro.core.base import BaseFLClient, ClientUpdate
from fedpro.privacy.privacy_engine import PrivacyEngine
from fedpro.configs.base_config import DPConfig

class SCAFFOLDClient(BaseFLClient):
    """
    Hospital client with local control variate c_i correcting for client drift.
    """
    
    def __init__(self, client_id: str, model: nn.Module, device: str = "cpu", dp_config: Optional[DPConfig] = None):
        super().__init__(client_id, model, device)
        self.dp_config = dp_config or DPConfig(enabled=False)
        self.privacy_engine: Optional[PrivacyEngine] = None
        
        # Local control variate c_i initialized to zero
        self.control_variate: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.model.named_parameters()
        }

    def fit(self, parameters: Dict[str, torch.Tensor], train_loader: DataLoader, config: Dict[str, Any]) -> ClientUpdate:
        start_time = time.time()
        self.set_parameters(parameters)
        self.model.train()
        
        server_c = config.get("server_control_variate", {})
        # Map server control variates to local device
        server_control = {
            k: v.to(self.device) for k, v in server_c.items()
        } if server_c else {
            k: torch.zeros_like(v) for k, v in self.control_variate.items()
        }
        
        initial_params = {k: v.clone() for k, v in self.model.named_parameters()}
        
        epochs = config.get("epochs", 2)
        lr = config.get("lr", 0.01)
        
        if self.privacy_engine is None:
            self.privacy_engine = PrivacyEngine(
                config=self.dp_config,
                dataset_size=len(train_loader.dataset),
                batch_size=train_loader.batch_size or 16
            )
            
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        
        total_loss = 0.0
        num_samples = 0
        num_steps = 0
        last_grad_norms = {"pre_clip_norm": 0.0, "post_noise_norm": 0.0}
        
        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Apply Differential Privacy if enabled
                if self.privacy_engine.is_enabled:
                    last_grad_norms = self.privacy_engine.sanitize_gradients(self.model, self.device)
                    
                # Apply SCAFFOLD drift correction: grad = grad - c_i + c
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in self.control_variate and name in server_control:
                        correction = -self.control_variate[name] + server_control[name]
                        param.grad.add_(correction)
                        
                optimizer.step()
                num_steps += 1
                total_loss += loss.item() * data.size(0)
                num_samples += data.size(0)
                
        if self.privacy_engine.is_enabled:
            self.privacy_engine.record_step(num_steps=num_steps)
            
        # Compute new local control variate c_i^+ and delta_c_i
        # c_i^+ = c_i - c + 1 / (K * eta_l) * (x - y_i)
        new_control_variate = {}
        delta_c = {}
        effective_lr = lr * max(1, num_steps)
        
        for name, param in self.model.named_parameters():
            if name in initial_params:
                diff = initial_params[name] - param.data
                c_i_plus = self.control_variate[name] - server_control.get(name, 0.0) + (diff / effective_lr)
                delta_c[name] = (c_i_plus - self.control_variate[name]).cpu()
                new_control_variate[name] = c_i_plus
                
        self.control_variate = new_control_variate
        training_time = time.time() - start_time
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        
        return ClientUpdate(
            client_id=self.client_id,
            parameters=self.get_parameters(),
            num_samples=num_samples,
            metrics={"loss": total_loss / max(1, num_samples)},
            privacy=self.privacy_engine.get_privacy_metrics(),
            grad_norms=last_grad_norms,
            computation_time_sec=training_time,
            bytes_transferred=param_bytes * 2, # weights + control variates
            control_variate_delta=delta_c
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
