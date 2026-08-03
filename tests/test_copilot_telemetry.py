"""
Quantitative Tests for Copilot Telemetry Engine (Cosine Drift & SNR).
"""

import unittest
import torch
import numpy as np
from fedpro.copilot.copilot_engine import AIFederatedCopilot

class TestCopilotTelemetry(unittest.TestCase):
    def setUp(self):
        self.copilot = AIFederatedCopilot(target_privacy_epsilon=10.0)

    def test_client_drift_cosine_matrix(self):
        """Verify cosine similarity matrix properties (symmetry, unit diagonal, range [-1, 1])."""
        d1 = {"w": torch.tensor([1.0, 0.0, 0.0])}
        d2 = {"w": torch.tensor([1.0, 0.0, 0.0])} # Identical direction
        d3 = {"w": torch.tensor([0.0, 1.0, 0.0])} # Orthogonal direction
        
        matrix, mean_off_diag = self.copilot.compute_client_drift_matrix([d1, d2, d3])
        
        self.assertEqual(matrix.shape, (3, 3))
        # Unit diagonal
        np.testing.assert_allclose(np.diag(matrix), [1.0, 1.0, 1.0], atol=1e-5)
        # Symmetry
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-5)
        # Cosine between identical vectors
        self.assertAlmostEqual(float(matrix[0, 1]), 1.0, places=4)
        # Cosine between orthogonal vectors
        self.assertAlmostEqual(float(matrix[0, 2]), 0.0, places=4)

    def test_gradient_snr_computation(self):
        """Verify Signal-to-Noise Ratio calculation."""
        d1 = {"w": torch.tensor([1.0, 1.0])}
        d2 = {"w": torch.tensor([1.1, 0.9])}
        snr = self.copilot.compute_gradient_snr([d1, d2])
        self.assertGreater(snr, 1.0)

if __name__ == "__main__":
    unittest.main()
