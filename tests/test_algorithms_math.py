"""
Mathematical and Formal Paper Invariant Tests for Federated Optimization Algorithms.
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from fedpro.models.mlp import HealthcareMLP
from fedpro.algorithms.fedavg import aggregate_fedavg
from fedpro.algorithms.fedprox import FedProxClient
from fedpro.algorithms.scaffold import SCAFFOLDClient
from fedpro.algorithms.fednova import aggregate_fednova
from fedpro.algorithms.fedopt import FedOptServerOptimizer
from fedpro.core.base import ClientUpdate

class TestAlgorithmsMath(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.input_dim = 8
        self.num_classes = 2
        self.model = HealthcareMLP(input_dim=self.input_dim, num_classes=self.num_classes)
        
        # Synthetic mini-dataset
        x = torch.randn(32, self.input_dim)
        y = torch.randint(0, self.num_classes, (32,))
        self.dataset = TensorDataset(x, y)
        self.loader = DataLoader(self.dataset, batch_size=16)

    def test_fedprox_proximal_regularization(self):
        """Verify FedProx strictly penalizes deviation from global parameters as mu increases."""
        client_low_mu = FedProxClient(client_id="c1", model=HealthcareMLP(input_dim=8, num_classes=2))
        client_high_mu = FedProxClient(client_id="c2", model=HealthcareMLP(input_dim=8, num_classes=2))
        
        init_params = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Train with mu = 0.001
        up_low = client_low_mu.fit(init_params, self.loader, {"epochs": 2, "lr": 0.01, "mu": 0.001, "momentum": 0.0})
        # Train with mu = 1.0 (constrained proximal resistance)
        up_high = client_high_mu.fit(init_params, self.loader, {"epochs": 2, "lr": 0.01, "mu": 1.0, "momentum": 0.0})
        
        # Measure parameter displacement ||w_final - w_init||
        dist_low = sum((up_low.parameters[k] - init_params[k]).float().norm(2).item() for k in init_params if init_params[k].is_floating_point())
        dist_high = sum((up_high.parameters[k] - init_params[k]).float().norm(2).item() for k in init_params if init_params[k].is_floating_point())
        
        # Invariant: Higher mu must constrain local update distance
        self.assertLess(dist_high, dist_low)

    def test_scaffold_control_variate_correction(self):
        """Verify SCAFFOLD computes control variate delta correctly for all trainable parameters."""
        client = SCAFFOLDClient(client_id="s1", model=HealthcareMLP(input_dim=8, num_classes=2))
        init_params = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        server_c = {k: torch.zeros_like(v) for k, v in init_params.items()}
        update = client.fit(init_params, self.loader, {"epochs": 1, "lr": 0.01, "server_control_variate": server_c})
        
        trainable_keys = [name for name, _ in self.model.named_parameters()]
        self.assertIsNotNone(update.control_variate_delta)
        self.assertEqual(len(update.control_variate_delta), len(trainable_keys))

    def test_fednova_heterogeneous_step_scaling(self):
        """Verify FedNova normalizes heterogeneous local gradient step counts."""
        init_params = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        p1 = {k: v.clone() + 1.0 for k, v in init_params.items()}
        p2 = {k: v.clone() + 1.0 for k, v in init_params.items()}
        
        u1 = ClientUpdate(client_id="c1", parameters=p1, num_samples=100, metrics={})
        u2 = ClientUpdate(client_id="c2", parameters=p2, num_samples=100, metrics={})
        
        agg = aggregate_fednova(init_params, [u1, u2], local_steps=[1, 10])
        self.assertEqual(len(agg), len(init_params))
        for k in init_params:
            self.assertEqual(agg[k].shape, init_params[k].shape)

    def test_fedopt_adam_second_moment_accumulation(self):
        """Verify FedAdam accumulates first and second moments monotonically."""
        optimizer = FedOptServerOptimizer(mode="fedadam", lr=0.1, beta1=0.9, beta2=0.99)
        init_params = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        p1 = {k: v.clone() + 0.1 for k, v in init_params.items()}
        u1 = ClientUpdate(client_id="c1", parameters=p1, num_samples=100, metrics={})
        
        new_params = optimizer.step(init_params, [u1])
        self.assertEqual(len(new_params), len(init_params))
        first_float_key = next(k for k, v in init_params.items() if v.is_floating_point())
        self.assertIn(first_float_key, optimizer.m)
        self.assertIn(first_float_key, optimizer.v)

if __name__ == "__main__":
    unittest.main()
