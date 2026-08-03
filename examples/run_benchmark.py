"""
Comprehensive Algorithm Comparative Benchmark for FedHealth.
Benchmarks FedAvg, FedProx, SCAFFOLD, FedNova, and FedAdam across non-IID clinical cohorts.
"""

import sys
import torch
import pandas as pd
from typing import List, Dict, Any

from fedpro.configs.base_config import FedHealthConfig
from fedpro.core.server import FLServer
from fedpro.models.mlp import HealthcareMLP
from fedpro.data.medical_datasets import load_breast_cancer_data
from fedpro.data.partitioner import dirichlet_non_iid_partition
from fedpro.algorithms.registry import get_algorithm_client_class
from fedpro.utils.logger import get_logger

logger = get_logger("AlgorithmBenchmark")

def run_single_algorithm(algo_name: str, num_rounds: int = 5) -> Dict[str, Any]:
    logger.info(f"\n========================================================")
    logger.info(f" Benchmarking Federated Algorithm: {algo_name.upper()}")
    logger.info(f"========================================================")
    
    # 1. Config
    config = FedHealthConfig()
    config.algorithm.name = algo_name
    config.training.num_rounds = num_rounds
    config.training.num_hospitals = 5
    config.training.local_epochs = 2
    config.privacy.enabled = True
    config.privacy.noise_multiplier = 0.5
    
    # 2. Dataset
    train_data, test_data, input_dim, num_classes, feature_names = load_breast_cancer_data()
    subsets = dirichlet_non_iid_partition(train_data, num_clients=config.training.num_hospitals, alpha=0.5)
    loaders = [torch.utils.data.DataLoader(s, batch_size=16, shuffle=True) for s in subsets]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    
    # 3. Model & Server
    global_model = HealthcareMLP(input_dim=input_dim, num_classes=num_classes)
    server = FLServer(global_model=global_model, config=config, feature_names=feature_names)
    
    # 4. Clients
    client_cls = get_algorithm_client_class(algo_name)
    clients = [
        client_cls(client_id=h.id, model=HealthcareMLP(input_dim=input_dim, num_classes=num_classes), dp_config=config.privacy)
        for h in server.hospitals
    ]
    
    # 5. Loop
    final_eval = None
    for r in range(1, num_rounds + 1):
        server.fit_round(
            round_num=r,
            clients=clients,
            dataloaders=loaders,
            config={"epochs": config.training.local_epochs, "lr": 0.01, "momentum": 0.9}
        )
        final_eval = server.evaluate_round(
            round_num=r,
            evaluator_client=clients[0],
            test_loader=test_loader
        )
        
    return {
        "Algorithm": algo_name.upper(),
        "Accuracy (%)": round(final_eval.accuracy, 2),
        "Loss": round(final_eval.loss, 4),
        "Precision (%)": round(final_eval.precision, 2),
        "Recall (%)": round(final_eval.recall, 2),
        "ROC-AUC (%)": round(final_eval.roc_auc, 2),
        "Privacy (eps)": round(server.privacy_accountant.get_privacy_spent().get("epsilon", 0.0), 2)
    }

def main():
    algorithms = ["fedavg", "fedprox", "scaffold", "fednova", "fedadam"]
    results = []
    
    for algo in algorithms:
        res = run_single_algorithm(algo, num_rounds=5)
        results.append(res)
        
    print("\n" + "="*80)
    print(" FEDHEALTH ALGORITHM COMPARATIVE BENCHMARK RESULTS")
    print("="*80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()
