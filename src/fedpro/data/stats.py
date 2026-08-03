"""
Statistical Analysis & Heterogeneity Measurement for Federated Datasets.
Computes Class Distributions, Earth Mover's Distance, and KL-Divergence across hospital nodes.
"""

import numpy as np
from torch.utils.data import Subset
from typing import List, Dict, Any

def compute_partition_distribution(subsets: List[Subset], num_classes: int = 2) -> List[Dict[str, Any]]:
    """
    Computes class counts and frequency distributions for each hospital subset.
    """
    distributions = []
    
    for idx, subset in enumerate(subsets):
        # Extract labels
        if hasattr(subset.dataset, "tensors") and len(subset.dataset.tensors) >= 2:
            all_labels = subset.dataset.tensors[1].numpy()
            labels = all_labels[subset.indices]
        else:
            labels = np.array([subset[i][1] for i in range(len(subset))])
            
        counts = [int(np.sum(labels == c)) for c in range(num_classes)]
        total = max(1, len(subset))
        freqs = [round(c / total, 4) for c in counts]
        
        distributions.append({
            "client_id": f"Hospital_{idx + 1}",
            "total_records": len(subset),
            "class_counts": counts,
            "class_frequencies": freqs
        })
        
    return distributions
