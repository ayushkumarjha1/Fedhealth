"""Unit tests for Differential Privacy and RDP Accountant."""
import unittest
import torch
import torch.nn as nn
from fedpro.privacy.rdp_accountant import RDPAccountant, compute_rdp_gaussian
from fedpro.privacy.dp_sgd import clip_and_add_noise

class TestPrivacy(unittest.TestCase):
    def test_gaussian_rdp(self):
        rdp = compute_rdp_gaussian(alpha=2.0, noise_multiplier=1.0)
        self.assertEqual(rdp, 1.0)

    def test_rdp_accountant(self):
        accountant = RDPAccountant(target_delta=1e-5)
        eps = accountant.step(noise_multiplier=1.0, sample_rate=0.1, num_steps=10)
        self.assertGreater(eps, 0.0)
        spent = accountant.get_privacy_spent()
        self.assertGreater(spent["epsilon"], 0.0)
        self.assertEqual(spent["delta"], 1e-5)

    def test_clip_and_noise(self):
        model = nn.Linear(5, 2)
        x = torch.randn(4, 5)
        y = torch.tensor([0, 1, 0, 1])
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        
        pre_norm, post_norm = clip_and_add_noise(
            model=model,
            clip_norm=1.0,
            noise_multiplier=0.5,
            batch_size=4,
            device=torch.device("cpu")
        )
        self.assertGreaterEqual(pre_norm, 0.0)
        self.assertGreaterEqual(post_norm, 0.0)

if __name__ == "__main__":
    unittest.main()
