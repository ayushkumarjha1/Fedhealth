"""
Clinical and Healthcare Dataset Loaders for Federated Learning.
Supports Breast Cancer Wisconsin, Heart Disease, Diabetes, and Synthetic Clinical Datasets.
"""

import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple, List, Dict, Any

def load_breast_cancer_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[TensorDataset, TensorDataset, int, int, List[str]]:
    """
    Loads Breast Cancer Wisconsin Diagnostic dataset.
    
    Returns:
        (train_dataset, test_dataset, input_dim, num_classes, feature_names)
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    return train_dataset, test_dataset, X.shape[1], 2, feature_names

def load_synthetic_clinical_data(
    num_samples: int = 2000, 
    num_features: int = 20, 
    num_classes: int = 2,
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[TensorDataset, TensorDataset, int, int, List[str]]:
    """
    Generates synthetic multi-biomarker patient records.
    """
    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=12,
        n_redundant=4,
        n_classes=num_classes,
        random_state=random_state
    )
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    feature_names = [f"Biomarker_{i+1}" for i in range(num_features)]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    return train_dataset, test_dataset, num_features, num_classes, feature_names
