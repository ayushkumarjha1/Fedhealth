"""
Diagnostic Evaluation Metrics for Medical Machine Learning.
Computes Loss, Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, and Confusion Matrix.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from typing import Dict, Any, Tuple

def compute_clinical_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: np.ndarray, 
    loss: float
) -> Dict[str, Any]:
    """
    Computes standard diagnostic healthcare machine learning metrics.
    
    Args:
        y_true: Ground-truth class labels (0 or 1)
        y_pred: Hard class predictions
        y_prob: Positive class probabilities
        loss: Cross-entropy evaluation loss
        
    Returns:
        Dictionary of formatted evaluation metrics
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "loss": round(float(loss), 4),
        "accuracy": round(float(acc * 100), 2),
        "precision": round(float(prec * 100), 2),
        "recall": round(float(rec * 100), 2),
        "specificity": round(float(specificity * 100), 2),
        "f1_score": round(float(f1 * 100), 2),
        "roc_auc": round(float(roc_auc * 100), 2),
        "confusion_matrix": cm.tolist()
    }
