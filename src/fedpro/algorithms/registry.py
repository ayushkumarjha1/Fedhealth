"""
Algorithm Factory and Registry for FedHealth.
Provides clean plug-and-play creation of client and server aggregation strategies.
"""

from typing import Dict, Type, Any, Callable
from fedpro.core.base import BaseFLClient
from fedpro.algorithms.fedavg import aggregate_fedavg
from fedpro.algorithms.fedprox import FedProxClient
from fedpro.algorithms.scaffold import SCAFFOLDClient
from fedpro.algorithms.fednova import aggregate_fednova
from fedpro.algorithms.fedopt import FedOptServerOptimizer

ALGORITHM_REGISTRY: Dict[str, Dict[str, Any]] = {
    "fedavg": {
        "client_class": FedProxClient, # with mu=0 reduces to standard client
        "aggregator": aggregate_fedavg,
        "description": "Federated Averaging (McMahan et al. 2017)"
    },
    "fedprox": {
        "client_class": FedProxClient,
        "aggregator": aggregate_fedavg,
        "description": "Federated Proximal (Li et al. 2020)"
    },
    "scaffold": {
        "client_class": SCAFFOLDClient,
        "aggregator": aggregate_fedavg,
        "description": "SCAFFOLD with Control Variates (Karimireddy et al. 2020)"
    },
    "fednova": {
        "client_class": FedProxClient,
        "aggregator": aggregate_fednova,
        "description": "FedNova Normalized Averaging (Wang et al. 2020)"
    },
    "fedadam": {
        "client_class": FedProxClient,
        "aggregator": "fedopt_adam",
        "description": "Adaptive Federated Optimization - FedAdam (Reddi et al. 2021)"
    },
    "fedyogi": {
        "client_class": FedProxClient,
        "aggregator": "fedopt_yogi",
        "description": "Adaptive Federated Optimization - FedYogi (Reddi et al. 2021)"
    }
}

def get_algorithm_client_class(algorithm_name: str) -> Type[BaseFLClient]:
    """Retrieve the appropriate client class for an algorithm."""
    name = algorithm_name.lower()
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm '{algorithm_name}'. Supported: {list(ALGORITHM_REGISTRY.keys())}")
    return ALGORITHM_REGISTRY[name]["client_class"]

def get_algorithm_aggregator(algorithm_name: str) -> Any:
    """Retrieve the aggregation function or handler for an algorithm."""
    name = algorithm_name.lower()
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm '{algorithm_name}'. Supported: {list(ALGORITHM_REGISTRY.keys())}")
    return ALGORITHM_REGISTRY[name]["aggregator"]
