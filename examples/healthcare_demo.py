"""
End-to-End Federated Healthcare Simulation Demo.
Runs a 10-round clinical breast cancer diagnostic experiment with Differential Privacy (DP-SGD & RDP).
"""

import sys
import torch
from fedpro.configs.base_config import FedHealthConfig
from fedpro.core.server import FLServer
from fedpro.models.mlp import HealthcareMLP
from fedpro.data.medical_datasets import load_breast_cancer_data
from fedpro.data.partitioner import dirichlet_non_iid_partition
from fedpro.algorithms.registry import get_algorithm_client_class
from fedpro.utils.logger import get_logger

logger = get_logger("HealthcareDemo")

def main():
    logger.info("Initializing FedHealth Clinical Benchmark...")
    
    # 1. Configuration
    config = FedHealthConfig()
    config.training.num_rounds = 10
    config.training.num_hospitals = 5
    config.training.local_epochs = 2
    config.privacy.enabled = True
    config.privacy.noise_multiplier = 0.5
    config.privacy.clip_norm = 1.0
    config.algorithm.name = "fedavg"
    
    # 2. Load Clinical Dataset
    train_data, test_data, input_dim, num_classes, feature_names = load_breast_cancer_data(
        test_size=0.2, random_state=42
    )
    
    # 3. Partition Data using Dirichlet Non-IID Skew
    subsets = dirichlet_non_iid_partition(
        dataset=train_data,
        num_clients=config.training.num_hospitals,
        alpha=0.5,
        num_classes=num_classes
    )
    loaders = [
        torch.utils.data.DataLoader(s, batch_size=config.training.batch_size, shuffle=True)
        for s in subsets
    ]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.training.batch_size, shuffle=False)
    
    # 4. Initialize Neural Model & Server
    global_model = HealthcareMLP(
        input_dim=input_dim,
        hidden_dims=config.model.hidden_dims,
        num_classes=num_classes,
        dropout_rate=config.model.dropout_rate,
        use_batch_norm=config.model.use_batch_norm
    )
    
    server = FLServer(
        global_model=global_model,
        config=config,
        feature_names=feature_names
    )
    
    # 5. Initialize Hospital Clients
    client_cls = get_algorithm_client_class(config.algorithm.name)
    clients = []
    for h in server.hospitals:
        c_model = HealthcareMLP(
            input_dim=input_dim,
            hidden_dims=config.model.hidden_dims,
            num_classes=num_classes,
            dropout_rate=config.model.dropout_rate,
            use_batch_norm=config.model.use_batch_norm
        )
        c = client_cls(
            client_id=h.id,
            model=c_model,
            device="cpu",
            dp_config=config.privacy
        )
        clients.append(c)
        
    logger.info(f"Initialized {len(clients)} hospital nodes. Commencing federated rounds...")
    
    # 6. Execute Training Loop
    for r in range(1, config.training.num_rounds + 1):
        server.fit_round(
            round_num=r,
            clients=clients,
            dataloaders=loaders,
            config={
                "epochs": config.training.local_epochs,
                "lr": config.training.learning_rate,
                "momentum": config.training.momentum,
                "weight_decay": config.training.weight_decay
            }
        )
        server.evaluate_round(
            round_num=r,
            evaluator_client=clients[0],
            test_loader=test_loader
        )
        
    logger.info("Simulation completed successfully.")
    
    # Print Copilot Final Insight
    copilot_insight = server.copilot.get_latest_insight()
    if copilot_insight:
        logger.info(f"AI Copilot Final Evaluation: {copilot_insight['summary']}")
        
    # Generate Markdown Summary
    report = server.report_generator.generate_markdown(
        experiment_config=config.to_dict(),
        metrics_history=server.tracker.rounds_history,
        hospitals_data=[h.to_dict() for h in server.hospitals],
        privacy_summary=server.privacy_accountant.get_privacy_spent(),
        xai_summary=server.explainer.explain_global_features(test_loader)
    )
    logger.info("Generated Clinical Markdown Report:\n" + report[:400] + "...\n[Report Complete]")

if __name__ == "__main__":
    main()
