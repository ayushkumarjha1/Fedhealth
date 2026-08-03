"""Unit tests for Pydantic configuration schemas."""
import unittest
from fedpro.configs.base_config import FedHealthConfig, DPConfig, FLTrainingConfig

class TestConfigs(unittest.TestCase):
    def test_default_config(self):
        cfg = FedHealthConfig()
        self.assertEqual(cfg.training.num_rounds, 15)
        self.assertTrue(cfg.privacy.enabled)
        self.assertEqual(cfg.privacy.clip_norm, 1.0)

    def test_dp_config_validation(self):
        with self.assertRaises(ValueError):
            DPConfig(target_delta=-0.1)

    def test_serialization(self):
        cfg = FedHealthConfig()
        d = cfg.to_dict()
        self.assertIn("training", d)
        self.assertIn("privacy", d)
        json_str = cfg.to_json()
        self.assertIn("FedHealth_Clinical_Benchmark", json_str)

if __name__ == "__main__":
    unittest.main()
