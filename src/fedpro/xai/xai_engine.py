r"""
Explainable AI (XAI) Subsystem for Clinical Healthcare Machine Learning.

Implements mathematically rigorous, scientifically grounded model interpretability:
1. Integrated Gradients (Sundararajan et al., 2017):
   Approximates the path integral of gradients along the straight-line interpolation
   between a clinical reference baseline x' and patient input x:
   IG_i(x) = (x_i - x_i') * \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha
   Satisfies the Axiom of Completeness: \sum_i IG_i(x) \approx F(x) - F(x')

2. Configurable Clinical Baselines:
   - Zero Baseline (x' = 0): Uninformed numerical reference
   - Cohort-Centroid Baseline (x' = mu_healthy): Biologically grounded reference
     representing the mean biomarker profile of the non-malignant/healthy cohort.
   - Custom Reference Baseline: Arbitrary clinician-defined physiological baseline.
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt

class ClinicalExplainer:
    """
    Produces mathematically sound feature attributions and clinician-friendly 
    diagnostic rationales for trained neural healthcare models.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        feature_names: Optional[List[str]] = None, 
        device: str = "cpu",
        baseline_mode: str = "zeros",
        reference_baseline: Optional[torch.Tensor] = None
    ):
        self.model = model
        self.feature_names = feature_names or [f"Biomarker_{i+1}" for i in range(30)]
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.baseline_mode = baseline_mode
        self.reference_baseline = reference_baseline.to(self.device) if reference_baseline is not None else None

    def set_cohort_centroid_baseline(
        self, 
        data_source: Union[torch.Tensor, torch.utils.data.DataLoader], 
        target_class: Optional[int] = 0
    ) -> torch.Tensor:
        """
        Computes and sets the reference baseline as the empirical centroid 
        (mean feature vector) of the specified cohort (e.g. benign class 0).
        """
        if isinstance(data_source, torch.Tensor):
            self.reference_baseline = torch.mean(data_source.float(), dim=0).to(self.device)
        elif isinstance(data_source, torch.utils.data.DataLoader):
            collected = []
            for inputs, targets in data_source:
                if target_class is not None:
                    mask = (targets == target_class)
                    if mask.sum() > 0:
                        collected.append(inputs[mask].float())
                else:
                    collected.append(inputs.float())
            if collected:
                all_tensors = torch.cat(collected, dim=0)
                self.reference_baseline = torch.mean(all_tensors, dim=0).to(self.device)
            else:
                self.reference_baseline = torch.zeros(len(self.feature_names), device=self.device)
        else:
            self.reference_baseline = torch.zeros(len(self.feature_names), device=self.device)
            
        self.baseline_mode = "cohort_centroid"
        return self.reference_baseline

    def get_baseline_tensor(self, x_shape: torch.Size) -> torch.Tensor:
        """Resolves the baseline tensor based on configured baseline mode."""
        if self.reference_baseline is not None:
            base = self.reference_baseline.clone()
            if base.dim() == 1 and len(x_shape) == 2:
                base = base.unsqueeze(0)
            return base.to(self.device).float()
            
        if self.baseline_mode == "cohort_centroid" and self.reference_baseline is not None:
            return self.reference_baseline.unsqueeze(0).to(self.device).float()
            
        # Default: Zero baseline
        return torch.zeros(x_shape, device=self.device).float()

    def compute_integrated_gradients(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None,
        steps: int = 50,
        baseline: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, float, float]:
        """
        Computes Integrated Gradients attribution vector for a single patient record.
        
        Args:
            input_tensor: 1D patient feature vector (D,)
            target_class: Target class index to explain (if None, uses argmax prediction)
            steps: Number of Riemann sum approximation steps (default: 50)
            baseline: Reference baseline vector x' (if None, uses configured baseline mode)
            
        Returns:
            Tuple of (attributions_array, predicted_prob, baseline_prob)
        """
        self.model.eval()
        x = input_tensor.unsqueeze(0).to(self.device).float() # (1, D)
        
        if baseline is None:
            x_prime = self.get_baseline_tensor(x.shape)
        else:
            x_prime = baseline.unsqueeze(0).to(self.device).float() if baseline.dim() == 1 else baseline.to(self.device).float()
            
        # Determine target class
        with torch.no_grad():
            orig_output = self.model(x)
            orig_probs = torch.softmax(orig_output, dim=1)[0]
            if target_class is None:
                target_class = int(torch.argmax(orig_probs).item())
            f_x = float(orig_probs[target_class].item())
            
            base_output = self.model(x_prime)
            base_probs = torch.softmax(base_output, dim=1)[0]
            f_x_prime = float(base_probs[target_class].item())
            
        # Generate interpolated path: x_k = x' + (k / m) * (x - x')
        alphas = torch.linspace(0.0, 1.0, steps, device=self.device)
        interpolated = x_prime + alphas.unsqueeze(1).unsqueeze(2) * (x - x_prime) # (steps, 1, D)
        interpolated = interpolated.squeeze(1).requires_grad_(True) # (steps, D)
        
        # Forward pass on all interpolated points
        outputs = self.model(interpolated)
        probs = torch.softmax(outputs, dim=1)[:, target_class]
        
        # Backward pass to compute path gradients
        grads = torch.autograd.grad(
            outputs=probs.sum(), 
            inputs=interpolated, 
            create_graph=False
        )[0] # (steps, D)
        
        # Riemann sum approximation of the integral
        avg_grads = torch.mean(grads, dim=0).detach().cpu().numpy() # (D,)
        delta_x = (x - x_prime).squeeze(0).detach().cpu().numpy() # (D,)
        
        ig_attributions = delta_x * avg_grads # (D,)
        return ig_attributions, f_x, f_x_prime

    def compare_baselines(
        self,
        sample_tensor: torch.Tensor,
        centroid_baseline: torch.Tensor,
        target_class: Optional[int] = None,
        steps: int = 50
    ) -> Dict[str, Any]:
        """
        Performs comparative XAI analysis comparing Zero Baseline vs. Cohort-Centroid Baseline.
        """
        zero_base = torch.zeros_like(sample_tensor)
        ig_zero, fx_z, fbase_z = self.compute_integrated_gradients(
            sample_tensor, target_class=target_class, steps=steps, baseline=zero_base
        )
        ig_centroid, fx_c, fbase_c = self.compute_integrated_gradients(
            sample_tensor, target_class=target_class, steps=steps, baseline=centroid_baseline
        )
        
        # Cosine similarity between attribution vectors
        norm_z = np.linalg.norm(ig_zero)
        norm_c = np.linalg.norm(ig_centroid)
        if norm_z > 0 and norm_c > 0:
            cosine_sim = float(np.dot(ig_zero, ig_centroid) / (norm_z * norm_c))
        else:
            cosine_sim = 1.0
            
        completeness_z = float(np.sum(ig_zero) - (fx_z - fbase_z))
        completeness_c = float(np.sum(ig_centroid) - (fx_c - fbase_c))
        
        return {
            "zero_baseline": {
                "attributions": ig_zero.tolist(),
                "f_x": fx_z,
                "f_baseline": fbase_z,
                "completeness_residual": completeness_z
            },
            "cohort_centroid_baseline": {
                "attributions": ig_centroid.tolist(),
                "f_x": fx_c,
                "f_baseline": fbase_c,
                "completeness_residual": completeness_c
            },
            "attribution_cosine_similarity": cosine_sim,
            "interpretation": (
                f"Attribution vectors share {cosine_sim*100:.1f}% directional alignment. "
                "Cohort-centroid baseline eliminates non-physical zero-reference artifacts."
            )
        }

    def explain_global_features(self, test_loader, num_batches: int = 5) -> List[Dict[str, Any]]:
        """
        Computes aggregated Integrated Gradients across representative test batches.
        """
        accumulated_ig = np.zeros(len(self.feature_names), dtype=np.float32)
        total_samples = 0
        
        for idx, (data, targets) in enumerate(test_loader):
            if idx >= num_batches:
                break
            for i in range(data.size(0)):
                ig, _, _ = self.compute_integrated_gradients(data[i], steps=20)
                accumulated_ig += np.abs(ig)
                total_samples += 1
                
        total_sum = float(accumulated_ig.sum())
        if total_sum > 0:
            normalized = (accumulated_ig / total_sum) * 100.0
        else:
            normalized = np.ones(len(self.feature_names)) / len(self.feature_names) * 100.0
            
        results = []
        for name, score in zip(self.feature_names, normalized):
            results.append({
                "feature": name,
                "importance": round(float(score), 2),
                "clinical_impact": "High" if score > 5.0 else ("Moderate" if score > 2.0 else "Low")
            })
            
        results.sort(key=lambda x: x["importance"], reverse=True)
        return results

    def explain_single_patient(
        self, 
        sample_tensor: torch.Tensor,
        top_k: int = 5,
        baseline: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Generates clinical attribution report for an individual patient record.
        """
        ig_attributions, prob_x, prob_base = self.compute_integrated_gradients(
            sample_tensor, steps=50, baseline=baseline
        )
        
        with torch.no_grad():
            output = self.model(sample_tensor.unsqueeze(0).to(self.device))
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class] * 100.0)
            
        inputs = sample_tensor.detach().cpu().numpy()
        ranked_indices = np.argsort(np.abs(ig_attributions))[::-1][:top_k]
        
        top_biomarkers = []
        for idx in ranked_indices:
            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Biomarker_{idx}"
            attr_val = float(ig_attributions[idx])
            top_biomarkers.append({
                "feature": feat_name,
                "value": round(float(inputs[idx]), 3),
                "attribution": round(attr_val, 4),
                "direction": "Risk Escalating (Promoting Malignancy)" if attr_val > 0 else "Risk Mitigating (Protective/Benign)"
            })
            
        diagnosis_label = "Malignant / High Risk" if pred_class == 1 else "Benign / Low Risk"
        
        top_positive = [b["feature"] for b in top_biomarkers if b["attribution"] > 0]
        top_negative = [b["feature"] for b in top_biomarkers if b["attribution"] < 0]
        
        rationale = (
            f"Model predicts {diagnosis_label} with {confidence:.1f}% confidence. "
            f"Primary risk drivers: {', '.join(top_positive) if top_positive else 'None'}. "
            f"Mitigating protective indicators: {', '.join(top_negative) if top_negative else 'None'}."
        )
        
        return {
            "diagnosis": diagnosis_label,
            "confidence": round(confidence, 1),
            "predicted_class": pred_class,
            "baseline_mode": self.baseline_mode,
            "top_biomarkers": top_biomarkers,
            "clinical_rationale": rationale,
            "completeness_delta": round(float(np.sum(ig_attributions) - (prob_x - prob_base)), 4)
        }
