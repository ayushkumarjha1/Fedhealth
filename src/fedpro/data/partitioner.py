"""
Research-grade Data Partitioning for Federated Healthcare Simulations.
Supports Uniform IID, Dirichlet Non-IID, Pathological Class Sharding, and Quantity Skew.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import List, Dict, Any, Tuple

def iid_partition(dataset: Dataset, num_clients: int, seed: int = 42) -> List[Subset]:
    """
    Uniform Independent and Identically Distributed (IID) partition.
    """
    np.random.seed(seed)
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    splits = np.array_split(indices, num_clients)
    return [Subset(dataset, split.tolist()) for split in splits]

def dirichlet_non_iid_partition(
    dataset: Dataset, 
    num_clients: int, 
    alpha: float = 0.5, 
    num_classes: int = 2,
    seed: int = 42
) -> List[Subset]:
    """
    Dirichlet Non-IID distribution partition simulating heterogeneous patient populations across hospitals.
    Lower alpha means higher non-IID label distribution skew.
    """
    np.random.seed(seed)
    
    # Extract labels from dataset
    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
        targets = dataset.tensors[1].numpy()
    else:
        # Fallback extract via loader
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    
    # Partition each class independently according to Dirichlet distribution
    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        np.random.shuffle(class_indices)
        
        # Sample Dirichlet proportions
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Scale proportions to total available samples for class c
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        class_splits = np.array_split(class_indices, split_points)
        
        for client_id in range(num_clients):
            client_indices[client_id].extend(class_splits[client_id].tolist())
            
    # Guarantee minimum samples per client
    for client_id in range(num_clients):
        if len(client_indices[client_id]) == 0:
            # Fallback random sample
            client_indices[client_id] = np.random.choice(len(dataset), size=10, replace=False).tolist()
            
    return [Subset(dataset, idxs) for idxs in client_indices]

def pathological_non_iid_partition(
    dataset: Dataset, 
    num_clients: int, 
    shards_per_client: int = 2, 
    seed: int = 42
) -> List[Subset]:
    """
    Pathological Non-IID partition where each hospital only has access to a subset of diagnostic classes.
    """
    np.random.seed(seed)
    
    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
        targets = dataset.tensors[1].numpy()
    else:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
        
    num_shards = num_clients * shards_per_client
    shard_size = len(dataset) // num_shards
    
    # Sort samples by class label
    sorted_indices = np.argsort(targets)
    # Split into shards
    shards = [sorted_indices[i * shard_size : (i + 1) * shard_size] for i in range(num_shards)]
    np.random.shuffle(shards)
    
    client_subsets = []
    for i in range(num_clients):
        assigned_shards = shards[i * shards_per_client : (i + 1) * shards_per_client]
        client_idx = np.concatenate(assigned_shards).tolist()
        client_subsets.append(Subset(dataset, client_idx))
        
    return client_subsets

def quantity_skew_partition(dataset: Dataset, num_clients: int, alpha: float = 1.0, seed: int = 42) -> List[Subset]:
    """
    Quantity Skew partition (Pareto/Power-law distribution of patient records per hospital).
    """
    np.random.seed(seed)
    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    split_points = (np.cumsum(proportions) * num_samples).astype(int)[:-1]
    splits = np.array_split(indices, split_points)
    return [Subset(dataset, s.tolist()) for s in splits]
