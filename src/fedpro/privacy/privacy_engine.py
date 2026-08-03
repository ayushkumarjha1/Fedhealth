"""
Client-side Privacy Engine for managing DP-SGD execution and local privacy accounting.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from fedpro.privacy.dp_sgd import clip_and_add_noise
from fedpro.privacy.rdp_accountant import RDPAccountant
from fedpro.configs.base_config import DPConfig

class PrivacyEngine:
    """
    Manages Differential Privacy enforcement for a local client/hospital.
    Coordinates gradient sanitization and tracks local epsilon expenditure.
    """
    
    def __init__(self, config: DPConfig, dataset_size: int, batch_size: int):
        self.config = config
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.sample_rate = batch_size / float(dataset_size) if dataset_size > 0 else 1.0
        self.accountant = RDPAccountant(target_delta=config.target_delta)
        
    @property
    def is_enabled(self) -> bool:
        return self.config.enabled

    def sanitize_gradients(self, model: nn.Module, device: torch.device) -> Dict[str, float]:
        """
        Clips gradients and injects calibrated Gaussian noise if DP is enabled.
        
        Returns:
            Dictionary with diagnostic norms: {'pre_clip_norm': float, 'post_noise_norm': float}
        """
        if not self.config.enabled:
            return {"pre_clip_norm": 0.0, "post_noise_norm": 0.0}
            
        pre_norm, post_norm = clip_and_add_noise(
            model=model,
            clip_norm=self.config.clip_norm,
            noise_multiplier=self.config.noise_multiplier,
            batch_size=self.batch_size,
            device=device
        )
        return {"pre_clip_norm": pre_norm, "post_noise_norm": post_norm}

    def record_step(self, num_steps: int = 1) -> float:
        """Record mini-batch optimization steps into the RDP accountant."""
        if not self.config.enabled:
            return 0.0
        return self.accountant.step(
            noise_multiplier=self.config.noise_multiplier,
            sample_rate=self.sample_rate,
            num_steps=num_steps
        )

    def get_privacy_metrics(self) -> Dict[str, Any]:
        """Return current cumulative privacy budget metrics."""
        if not self.config.enabled:
            return {
                "enabled": False,
                "epsilon": 0.0,
                "delta": self.config.target_delta,
                "clip_norm": self.config.clip_norm,
                "noise_multiplier": self.config.noise_multiplier,
                "status": "Disabled"
            }
            
        spent = self.accountant.get_privacy_spent()
        return {
            "enabled": True,
            "epsilon": spent["epsilon"],
            "delta": spent["delta"],
            "optimal_alpha": spent.get("optimal_alpha", 1.0),
            "clip_norm": self.config.clip_norm,
            "noise_multiplier": self.config.noise_multiplier,
            "steps": spent["steps"],
            "status": "Exhausted" if self.config.max_epsilon and spent["epsilon"] > self.config.max_epsilon else "Active"
        }
