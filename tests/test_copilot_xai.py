"""Unit tests for AI Copilot and Explainable AI subsystems."""
import unittest
import torch
from fedpro.copilot.copilot_engine import AIFederatedCopilot
from fedpro.models.mlp import HealthcareMLP
from fedpro.xai.xai_engine import ClinicalExplainer

class TestCopilotAndXAI(unittest.TestCase):
    def test_copilot_analysis(self):
        copilot = AIFederatedCopilot()
        history = [
            {"round": 1, "loss": 0.7, "accuracy": 65.0},
            {"round": 2, "loss": 0.5, "accuracy": 72.0}
        ]
        hospitals = [{"name": "Hospital_1", "status": "Idle"}]
        privacy = {"enabled": True, "epsilon": 1.2}
        
        report = copilot.analyze_round(2, history, hospitals, privacy, "fedavg")
        self.assertEqual(report["category"], "OPTIMAL")
        self.assertGreater(len(report["all_insights"]), 0)

    def test_xai_explainer(self):
        model = HealthcareMLP(input_dim=10, num_classes=2)
        explainer = ClinicalExplainer(model, feature_names=[f"F_{i}" for i in range(10)])
        
        sample = torch.randn(10)
        diagnosis = explainer.explain_single_patient(sample)
        self.assertIn("diagnosis", diagnosis)
        self.assertIn("confidence", diagnosis)
        self.assertGreater(len(diagnosis["top_biomarkers"]), 0)

if __name__ == "__main__":
    unittest.main()
