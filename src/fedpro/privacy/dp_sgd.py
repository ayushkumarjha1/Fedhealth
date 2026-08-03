r"""
DP-SGD Gradient Clipping and Calibrated Gaussian Noise Injection for PyTorch.

Mathematical formulation:
1. Clip gradients: g_i = g_i / max(1, ||g_i||_2 / C)
2. Add Gaussian noise: \tilde{g} = \frac{1}{B} (\sum_{i=1}^B g_i + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I}))
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

def clip_and_add_noise(
    model: nn.Module, 
    clip_norm: float, 
    noise_multiplier: float, 
    batch_size: int,
    device: Optional[torch.device] = None
) -> Tuple[float, float]:
    """
    Clips model gradient norm to `clip_norm` and injects calibrated Gaussian noise.
    
    Args:
        model: PyTorch model with computed parameter gradients
        clip_norm: Maximum L2 norm threshold C
        noise_multiplier: Noise scale sigma
        batch_size: Mini-batch size B
        device: Target compute device (optional, defaults to model parameter device or CPU)
        
    Returns:
        Tuple of (pre_clip_grad_norm, post_noise_grad_norm)
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    # 1. Compute total gradient norm before clipping
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
    pre_clip_norm = float(total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm)
    
    # 2. Add calibrated Gaussian noise to each parameter tensor
    if noise_multiplier > 0.0:
        noise_std = (noise_multiplier * clip_norm) / float(batch_size)
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0.0, 
                    std=noise_std, 
                    size=param.grad.shape, 
                    device=device,
                    dtype=param.grad.dtype
                )
                param.grad.add_(noise)
                
    # 3. Measure post-noise gradient norm for diagnostics
    post_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            post_norm_sq += param.grad.detach().norm(2).item() ** 2
    post_noise_norm = post_norm_sq ** 0.5
    
    return pre_clip_norm, post_noise_norm
