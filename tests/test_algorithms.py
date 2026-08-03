"""Unit tests for Federated Learning Algorithms."""
import unittest
import torch
from fedpro.core.base import ClientUpdate
from fedpro.algorithms.fedavg import aggregate_fedavg
from fedpro.algorithms.fednova import aggregate_fednova
from fedpro.algorithms.fedopt import FedOptServerOptimizer

class TestAlgorithms(unittest.TestCase):
    def test_fedavg_aggregation(self):
        p1 = {"weight": torch.tensor([2.0, 2.0])}
        p2 = {"weight": torch.tensor([4.0, 4.0])}
        
        u1 = ClientUpdate(client_id="c1", parameters=p1, num_samples=100)
        u2 = ClientUpdate(client_id="c2", parameters=p2, num_samples=100)
        
        agg = aggregate_fedavg([u1, u2])
        self.assertTrue(torch.allclose(agg["weight"], torch.tensor([3.0, 3.0])))

    def test_fednova_aggregation(self):
        w0 = {"weight": torch.tensor([0.0, 0.0])}
        p1 = {"weight": torch.tensor([2.0, 2.0])}
        p2 = {"weight": torch.tensor([4.0, 4.0])}
        
        u1 = ClientUpdate(client_id="c1", parameters=p1, num_samples=50)
        u2 = ClientUpdate(client_id="c2", parameters=p2, num_samples=50)
        
        agg = aggregate_fednova(w0, [u1, u2], local_steps=[2, 4])
        self.assertIn("weight", agg)

    def test_fedopt_adam(self):
        opt = FedOptServerOptimizer(mode="fedadam", lr=1.0)
        w0 = {"weight": torch.tensor([1.0, 1.0])}
        p1 = {"weight": torch.tensor([0.9, 0.9])}
        u1 = ClientUpdate(client_id="c1", parameters=p1, num_samples=50)
        
        w_next = opt.step(w0, [u1])
        self.assertIn("weight", w_next)

if __name__ == "__main__":
    unittest.main()
