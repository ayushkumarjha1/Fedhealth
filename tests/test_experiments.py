"""
Tests for ExperimentTracker and Reproducibility pipeline.
"""

import unittest
import os
import shutil
import torch

from fedpro.models.mlp import HealthcareMLP
from fedpro.experiments.tracker import ExperimentTracker

class TestExperimentTracker(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_experiments_dir"
        self.tracker = ExperimentTracker(experiment_name="Unit_Test_Run", output_dir=self.test_dir)
        self.model = HealthcareMLP(input_dim=8, num_classes=2)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_tracker_lifecycle_and_artifacts(self):
        """Verify tracker logs rounds, saves checkpoints, and produces plots and report."""
        # Log 2 rounds
        self.tracker.log_round({"round": 1, "loss": 0.65, "accuracy": 70.0, "epsilon": 0.5, "confusion_matrix": [[10, 2], [3, 15]]}, model=self.model)
        self.tracker.log_round({"round": 2, "loss": 0.42, "accuracy": 85.0, "epsilon": 1.1, "confusion_matrix": [[11, 1], [2, 16]]}, model=self.model)
        
        artifacts = self.tracker.finalize(model=self.model)
        
        self.assertIn("model_final", artifacts)
        self.assertIn("report", artifacts)
        self.assertTrue(os.path.exists(artifacts["model_final"]))
        self.assertTrue(os.path.exists(artifacts["report"]))
        self.assertTrue(os.path.exists(os.path.join(self.tracker.run_dir, "metrics.json")))
        self.assertTrue(os.path.exists(os.path.join(self.tracker.run_dir, "telemetry.jsonl")))

if __name__ == "__main__":
    unittest.main()
