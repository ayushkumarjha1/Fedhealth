"""
Tests for Empirical Membership Inference Attack (MIA) and Cohort-Centroid XAI Baselines.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import os
import shutil
import tempfile

from fedpro.privacy.mia_evaluator import MIAEvaluator
from fedpro.xai.xai_engine import ClinicalExplainer
from fedpro.models.mlp import HealthcareMLP

class TestMIAAndBaselines(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.input_dim = 10
        self.num_classes = 2
        self.temp_dir = tempfile.mkdtemp()
        
        # Synthetic dataset
        X_train = torch.randn(40, self.input_dim)
        y_train = torch.randint(0, 2, (40,))
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train), batch_size=8, shuffle=False
        )
        
        X_test = torch.randn(20, self.input_dim)
        y_test = torch.randint(0, 2, (20,))
        self.test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test, y_test), batch_size=8, shuffle=False
        )
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_mia_evaluator_metrics(self):
        """Verify MIA evaluator computes valid ROC-AUC and attack metrics."""
        model = HealthcareMLP(input_dim=self.input_dim, num_classes=self.num_classes)
        evaluator = MIAEvaluator()
        
        results = evaluator.evaluate_attack(
            model=model,
            member_loader=self.train_loader,
            non_member_loader=self.test_loader,
            model_name="TestModel"
        )
        
        self.assertIn("attack_auc", results)
        self.assertIn("attack_accuracy", results)
        self.assertIn("max_privacy_advantage", results)
        self.assertGreaterEqual(results["attack_auc"], 0.0)
        self.assertLessEqual(results["attack_auc"], 1.0)
        self.assertEqual(results["num_members"], 40)
        self.assertEqual(results["num_non_members"], 20)

    def test_mia_comparison_and_artifacts(self):
        """Verify comparative MIA produces plot, Markdown report, and metric delta."""
        model_nondp = HealthcareMLP(input_dim=self.input_dim, num_classes=self.num_classes)
        model_dp = HealthcareMLP(input_dim=self.input_dim, num_classes=self.num_classes)
        
        evaluator = MIAEvaluator()
        comp = evaluator.compare_dp_vs_nondp(
            nondp_model=model_nondp,
            dp_model=model_dp,
            member_loader=self.train_loader,
            non_member_loader=self.test_loader,
            output_dir=self.temp_dir
        )
        
        self.assertIn("nondp", comp)
        self.assertIn("dp", comp)
        self.assertIn("auc_reduction", comp)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "mia_evaluation.png")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "mia_report.md")))

    def test_cohort_centroid_computation(self):
        """Verify ClinicalExplainer correctly calculates mean feature centroid of reference cohort."""
        model = HealthcareMLP(input_dim=self.input_dim, num_classes=self.num_classes)
        explainer = ClinicalExplainer(model, feature_names=[f"F_{i}" for i in range(self.input_dim)])
        
        centroid = explainer.set_cohort_centroid_baseline(self.train_loader, target_class=0)
        self.assertEqual(centroid.shape[0], self.input_dim)
        self.assertEqual(explainer.baseline_mode, "cohort_centroid")

    def test_compare_baselines_completeness(self):
        """Verify both Zero and Cohort-Centroid baselines satisfy Axiom of Completeness."""
        model = HealthcareMLP(input_dim=self.input_dim, num_classes=self.num_classes)
        explainer = ClinicalExplainer(model, feature_names=[f"F_{i}" for i in range(self.input_dim)])
        
        centroid = explainer.set_cohort_centroid_baseline(self.train_loader, target_class=0)
        sample = torch.randn(self.input_dim)
        
        comp = explainer.compare_baselines(sample, centroid_baseline=centroid, steps=50)
        
        self.assertIn("attribution_cosine_similarity", comp)
        self.assertIn("zero_baseline", comp)
        self.assertIn("cohort_centroid_baseline", comp)
        
        # Residual completeness error must be strictly bounded < 0.05
        res_z = abs(comp["zero_baseline"]["completeness_residual"])
        res_c = abs(comp["cohort_centroid_baseline"]["completeness_residual"])
        self.assertLess(res_z, 0.05, f"Zero baseline residual {res_z} exceeds threshold.")
        self.assertLess(res_c, 0.05, f"Cohort centroid residual {res_c} exceeds threshold.")

if __name__ == "__main__":
    unittest.main()
