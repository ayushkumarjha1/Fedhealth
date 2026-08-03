"""
Scientific and Completeness Axiom Tests for Clinical Explainable AI (XAI).
"""

import unittest
import torch
import numpy as np

from fedpro.models.mlp import HealthcareMLP
from fedpro.xai.xai_engine import ClinicalExplainer

class TestXAIScientific(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_dim = 10
        self.num_classes = 2
        self.feature_names = [f"Biomarker_{i+1}" for i in range(self.input_dim)]
        self.model = HealthcareMLP(input_dim=self.input_dim, num_classes=self.num_classes)
        self.explainer = ClinicalExplainer(self.model, feature_names=self.feature_names)

    def test_integrated_gradients_completeness(self):
        """
        Verify the Axiom of Completeness:
        \sum_{i=1}^d IG_i(x) \approx F(x) - F(x')
        """
        sample = torch.randn(self.input_dim)
        ig_attr, prob_x, prob_base = self.explainer.compute_integrated_gradients(sample, steps=100)
        
        sum_attributions = float(np.sum(ig_attr))
        expected_diff = prob_x - prob_base
        
        # Invariant: Discrepancy between integral approximation and probability difference should be small (< 0.1)
        self.assertAlmostEqual(sum_attributions, expected_diff, delta=0.08)

    def test_single_patient_clinical_report(self):
        """Verify single patient report structure, ranking order, and rationale generation."""
        sample = torch.randn(self.input_dim)
        report = self.explainer.explain_single_patient(sample, top_k=3)
        
        self.assertIn("diagnosis", report)
        self.assertIn("confidence", report)
        self.assertIn("top_biomarkers", report)
        self.assertEqual(len(report["top_biomarkers"]), 3)
        
        # Check ranking order (absolute attribution descending)
        attrs = [abs(b["attribution"]) for b in report["top_biomarkers"]]
        self.assertEqual(attrs, sorted(attrs, reverse=True))

if __name__ == "__main__":
    unittest.main()
