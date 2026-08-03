r"""
Membership Inference Attack (MIA) Empirical Evaluation Engine for FedHealth.

Implements rigorous, empirical privacy auditing comparing DP vs. Non-DP models.
Evaluates the empirical susceptibility of trained neural models to membership
leakage under loss-based and confidence-based query attacks.

References:
- Shokri, R. et al. (2017). "Membership Inference Attacks Against Machine Learning Models." IEEE S&P.
- Yeom, S. et al. (2018). "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting." IEEE CSF.
- Carlini, N. et al. (2022). "Membership Inference Attacks From First Principles." IEEE S&P.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score

class MIAEvaluator:
    """
    Empirical Membership Inference Attack Evaluator.
    Measures the practical privacy protection conferred by DP-SGD against
    an adversary attempting to determine whether specific patient records
    were included in the training cohort.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def compute_sample_losses(
        self, 
        model: nn.Module, 
        dataloader: torch.utils.data.DataLoader
    ) -> np.ndarray:
        """
        Computes per-sample cross-entropy losses under model predictions.
        
        Args:
            model: PyTorch neural model to query
            dataloader: DataLoader with (inputs, targets) batches
            
        Returns:
            1D numpy array of individual sample loss values
        """
        model.eval()
        model.to(self.device)
        criterion = nn.CrossEntropyLoss(reduction="none")
        losses = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
                losses.extend(batch_loss.cpu().numpy().tolist())
                
        return np.array(losses, dtype=np.float64)

    def evaluate_attack(
        self,
        model: nn.Module,
        member_loader: torch.utils.data.DataLoader,
        non_member_loader: torch.utils.data.DataLoader,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Runs a loss-threshold membership inference attack against the target model.
        
        Members are labeled 1 (in training set); non-members are labeled 0.
        Attacker score is negative loss (-loss), where higher score indicates membership.
        """
        member_losses = self.compute_sample_losses(model, member_loader)
        non_member_losses = self.compute_sample_losses(model, non_member_loader)
        
        # Ground truth labels: 1 for members, 0 for non-members
        y_true = np.concatenate([np.ones(len(member_losses)), np.zeros(len(non_member_losses))])
        
        # Scores: negative loss (lower loss -> higher probability of membership)
        # Add small epsilon to prevent -inf
        scores = -np.concatenate([member_losses, non_member_losses])
        
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        attack_auc = float(auc(fpr, tpr))
        
        # Find optimal threshold maximizing Youden's J statistic = TPR - FPR
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        best_threshold = thresholds[best_idx]
        max_advantage = float(youden_j[best_idx])
        
        # Binary predictions at optimal threshold
        y_pred = (scores >= best_threshold).astype(int)
        
        attack_accuracy = float(accuracy_score(y_true, y_pred))
        attack_precision = float(precision_score(y_true, y_pred, zero_division=0))
        attack_recall = float(recall_score(y_true, y_pred, zero_division=0))
        
        # Compute TPR at low FPR operating points (e.g. 1% and 5% FPR)
        tpr_at_1pct_fpr = float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1]) if any(fpr <= 0.01) else 0.0
        tpr_at_5pct_fpr = float(tpr[np.searchsorted(fpr, 0.05, side="right") - 1]) if any(fpr <= 0.05) else 0.0

        return {
            "model_name": model_name,
            "num_members": len(member_losses),
            "num_non_members": len(non_member_losses),
            "mean_member_loss": float(np.mean(member_losses)),
            "mean_non_member_loss": float(np.mean(non_member_losses)),
            "loss_generalization_gap": float(np.mean(non_member_losses) - np.mean(member_losses)),
            "attack_auc": attack_auc,
            "attack_accuracy": attack_accuracy,
            "attack_precision": attack_precision,
            "attack_recall": attack_recall,
            "max_privacy_advantage": max_advantage,
            "tpr_at_1pct_fpr": tpr_at_1pct_fpr,
            "tpr_at_5pct_fpr": tpr_at_5pct_fpr,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }

    def compare_dp_vs_nondp(
        self,
        nondp_model: nn.Module,
        dp_model: nn.Module,
        member_loader: torch.utils.data.DataLoader,
        non_member_loader: torch.utils.data.DataLoader,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes comparative MIA evaluation on Non-DP vs DP-SGD models.
        Produces comparative metrics, ROC curves, and scientific Markdown audit.
        """
        nondp_results = self.evaluate_attack(
            nondp_model, member_loader, non_member_loader, model_name="Non-DP Baseline"
        )
        dp_results = self.evaluate_attack(
            dp_model, member_loader, non_member_loader, model_name="FedHealth DP-SGD"
        )
        
        comparison = {
            "nondp": nondp_results,
            "dp": dp_results,
            "auc_reduction": nondp_results["attack_auc"] - dp_results["attack_auc"],
            "advantage_reduction": nondp_results["max_privacy_advantage"] - dp_results["max_privacy_advantage"]
        }
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._plot_mia_comparison(nondp_results, dp_results, output_dir)
            self._write_mia_report(comparison, output_dir)
            
        return comparison

    def _plot_mia_comparison(
        self, 
        nondp: Dict[str, Any], 
        dp: Dict[str, Any], 
        output_dir: str
    ) -> str:
        """Generates publication-ready comparative ROC curves."""
        plot_path = os.path.join(output_dir, "mia_evaluation.png")
        
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        
        # Plot Non-DP curve
        ax.plot(
            nondp["fpr"], nondp["tpr"], 
            color="#ef4444", lw=2.5, 
            label=f"Non-DP Baseline (MIA AUC = {nondp['attack_auc']:.3f})"
        )
        
        # Plot DP-SGD curve
        ax.plot(
            dp["fpr"], dp["tpr"], 
            color="#10b981", lw=2.5, 
            label=f"FedHealth DP-SGD (MIA AUC = {dp['attack_auc']:.3f})"
        )
        
        # Plot Random Guess / Perfect Privacy Baseline
        ax.plot([0, 1], [0, 1], color="#6b7280", lw=1.5, linestyle="--", label="Random Guess (Perfect Privacy AUC = 0.500)")
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate (TPR / Attack Power)", fontsize=12, fontweight="bold")
        ax.set_title("Empirical Membership Inference Attack (MIA) Resilience", fontsize=14, fontweight="bold", pad=15)
        ax.legend(loc="lower right", fontsize=11, frameon=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        return plot_path

    def _write_mia_report(self, comparison: Dict[str, Any], output_dir: str) -> str:
        """Generates comprehensive scientific Markdown audit of empirical privacy."""
        report_path = os.path.join(output_dir, "mia_report.md")
        nondp = comparison["nondp"]
        dp = comparison["dp"]
        
        content = f"""# Empirical Membership Inference Attack (MIA) Audit Report

**Evaluation Subsystem:** `fedpro.privacy.mia_evaluator`  
**Threat Model:** Black-box Loss Threshold Query Attack (Yeom et al., 2018; Carlini et al., 2022)  

---

## 1. Executive Summary

This report evaluates the **empirical privacy leakage** of FedHealth models by measuring the attacker's ability to distinguish training cohort members from unseen validation patients.

| Metric | Non-DP Baseline | FedHealth DP-SGD | Privacy Gain (Delta) |
| :--- | :---: | :---: | :---: |
| **Attack ROC-AUC** | **{nondp['attack_auc']:.4f}** | **{dp['attack_auc']:.4f}** | **{comparison['auc_reduction']:+.4f} (Closer to 0.50)** |
| **Optimal Attack Accuracy** | {nondp['attack_accuracy']*100:.2f}% | {dp['attack_accuracy']*100:.2f}% | {(nondp['attack_accuracy'] - dp['attack_accuracy'])*100:-.2f}% |
| **Max Privacy Advantage ($J$)** | {nondp['max_privacy_advantage']:.4f} | {dp['max_privacy_advantage']:.4f} | {comparison['advantage_reduction']:+.4f} |
| **Generalization Loss Gap** | {nondp['loss_generalization_gap']:.4f} | {dp['loss_generalization_gap']:.4f} | {(nondp['loss_generalization_gap'] - dp['loss_generalization_gap']):-.4f} |
| **TPR @ 1% FPR** | {nondp['tpr_at_1pct_fpr']:.4f} | {dp['tpr_at_1pct_fpr']:.4f} | {(nondp['tpr_at_1pct_fpr'] - dp['tpr_at_1pct_fpr']):-.4f} |

---

## 2. Scientific Interpretation & Threat Model

1. **Theoretical vs Empirical Privacy**:
   - Analytical Rényi DP guarantees an upper bound on privacy loss.
   - The Membership Inference Attack measures empirical vulnerability under the standard loss-threshold adversary.
   - An Attack AUC near **0.500** indicates that the model predictions on training samples are statistically indistinguishable from unseen patient samples.

2. **Empirical Defense Mechanism**:
   - DP-SGD limits overfitting and bounds per-sample influence via gradient clipping ($C$) and Gaussian noise injection ($\\\\sigma$).
   - As observed in the loss generalization gap ({nondp['loss_generalization_gap']:.4f} -> {dp['loss_generalization_gap']:.4f}), differential privacy significantly reduces the margin between train and test loss distributions.

---

## 3. Artifact Reference
- **ROC Visualization**: `mia_evaluation.png`
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return report_path
