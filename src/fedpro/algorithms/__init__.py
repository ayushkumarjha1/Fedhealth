"""FedHealth Federated Optimization Algorithms Package."""
from fedpro.algorithms.fedavg import aggregate_fedavg
from fedpro.algorithms.fedprox import FedProxClient
from fedpro.algorithms.scaffold import SCAFFOLDClient
from fedpro.algorithms.fednova import aggregate_fednova
from fedpro.algorithms.fedopt import FedOptServerOptimizer
from fedpro.algorithms.registry import (
    ALGORITHM_REGISTRY,
    get_algorithm_client_class,
    get_algorithm_aggregator,
)

__all__ = [
    "aggregate_fedavg",
    "FedProxClient",
    "SCAFFOLDClient",
    "aggregate_fednova",
    "FedOptServerOptimizer",
    "ALGORITHM_REGISTRY",
    "get_algorithm_client_class",
    "get_algorithm_aggregator",
]
