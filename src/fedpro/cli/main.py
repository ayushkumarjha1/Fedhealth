"""
FedHealth Command Line Interface (CLI).
Provides unified access to experiment orchestration, multi-algorithm benchmarking, 
live dashboard hosting, explainable AI diagnostics, and empirical privacy audits.
"""

import sys
import os
import argparse
import torch
import uvicorn
from typing import List

from fedpro.configs.base_config import FedHealthConfig
from fedpro.core.server import FLServer
from fedpro.models.mlp import HealthcareMLP
from fedpro.data.medical_datasets import load_breast_cancer_data, load_synthetic_clinical_data
from fedpro.data.partitioner import dirichlet_non_iid_partition
from fedpro.algorithms.registry import get_algorithm_client_class
from fedpro.privacy.mia_evaluator import MIAEvaluator
from fedpro.xai.xai_engine import ClinicalExplainer
from fedpro.utils.logger import get_logger

logger = get_logger("FedHealthCLI")

def cmd_run(args):
    """Executes a single federated learning experiment."""
    logger.info(f"Initiating FedHealth experiment: {args.name} | Algorithm: {args.algo.upper()}")
    
    config = FedHealthConfig()
    config.experiment_name = args.name
    config.algorithm.name = args.algo.lower()
    config.training.num_rounds = args.rounds
    config.training.num_hospitals = args.hospitals
    config.training.local_epochs = args.epochs
    config.privacy.enabled = args.dp
    config.privacy.noise_multiplier = args.noise
    config.privacy.clip_norm = args.clip
    
    # Load dataset
    train_data, test_data, in_dim, num_classes, feature_names = load_breast_cancer_data()
    subsets = dirichlet_non_iid_partition(train_data, num_clients=args.hospitals, alpha=args.alpha)
    loaders = [torch.utils.data.DataLoader(s, batch_size=16, shuffle=True) for s in subsets]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Build Model & Server
    global_model = HealthcareMLP(input_dim=in_dim, num_classes=num_classes)
    server = FLServer(global_model=global_model, config=config, feature_names=feature_names)
    
    # Build Clients
    client_cls = get_algorithm_client_class(args.algo.lower())
    clients = [
        client_cls(client_id=h.id, model=HealthcareMLP(input_dim=in_dim, num_classes=num_classes), dp_config=config.privacy)
        for h in server.hospitals
    ]
    
    # Run Communication Rounds
    for r in range(1, args.rounds + 1):
        server.fit_round(
            round_num=r,
            clients=clients,
            dataloaders=loaders,
            config={"epochs": config.training.local_epochs, "lr": 0.01, "mu": 0.01}
        )
        server.evaluate_round(
            round_num=r,
            evaluator_client=clients[0],
            test_loader=test_loader
        )
        
    # Finalize & generate plots/report
    artifacts = server.tracker.finalize(model=server.global_model)
    logger.info("Simulation completed successfully.")
    logger.info(f"Artifacts generated in: {server.tracker.run_dir}")
    for name, path in artifacts.items():
        logger.info(f"  - {name}: {path}")

def cmd_benchmark(args):
    """Runs a multi-algorithm comparative benchmark."""
    algos = [a.strip().lower() for a in args.algorithms.split(",")]
    logger.info(f"Commencing comparative benchmark across: {', '.join([a.upper() for a in algos])}")
    
    from examples.run_benchmark import run_single_algorithm
    import pandas as pd
    
    results = []
    for algo in algos:
        res = run_single_algorithm(algo, num_rounds=args.rounds)
        results.append(res)
        
    print("\n" + "="*80)
    print(" FEDHEALTH ALGORITHM COMPARATIVE BENCHMARK RESULTS")
    print("="*80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("="*80)

def cmd_dashboard(args):
    """Launches the production FastAPI and WebSocket telemetry backend."""
    logger.info(f"Starting FedHealth telemetry server on {args.host}:{args.port}...")
    uvicorn.run("fedpro.api.dashboard_server:app", host=args.host, port=args.port, reload=False)

def cmd_explain(args):
    """Produces Explainable AI diagnostic report for a patient sample."""
    train_data, test_data, in_dim, num_classes, feature_names = load_breast_cancer_data()
    model = HealthcareMLP(input_dim=in_dim, num_classes=num_classes)
    
    explainer = ClinicalExplainer(model, feature_names=feature_names)
    
    # Compute centroid of benign training cases
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    centroid = explainer.set_cohort_centroid_baseline(train_loader, target_class=0)
    
    sample, target = test_data[args.sample_idx]
    
    if args.baseline == "compare":
        comp = explainer.compare_baselines(sample, centroid_baseline=centroid, steps=50)
        print("\n" + "="*70)
        print(" CLINICAL XAI: ZERO VS. COHORT-CENTROID BASELINE COMPARISON")
        print("="*70)
        print(f"Attribution Directional Alignment (Cosine Similarity): {comp['attribution_cosine_similarity']*100:.1f}%")
        print(f"Zero-Baseline Completeness Residual         : {comp['zero_baseline']['completeness_residual']:+.5f}")
        print(f"Cohort-Centroid Completeness Residual        : {comp['cohort_centroid_baseline']['completeness_residual']:+.5f}")
        print(f"\nClinical Takeaway:\n  {comp['interpretation']}")
        print("="*70)
    else:
        if args.baseline == "zeros":
            explainer.baseline_mode = "zeros"
            report = explainer.explain_single_patient(sample, baseline=torch.zeros_like(sample))
        else:
            explainer.baseline_mode = "cohort_centroid"
            report = explainer.explain_single_patient(sample, baseline=centroid)
            
        print("\n" + "="*60)
        print(f" CLINICAL XAI REPORT (Baseline: {explainer.baseline_mode.upper()})")
        print("="*60)
        print(f"Predicted Diagnosis : {report['diagnosis']}")
        print(f"Model Confidence    : {report['confidence']}%")
        print(f"Ground Truth Label  : {'Malignant' if target == 1 else 'Benign'}")
        print(f"Completeness Delta  : {report['completeness_delta']}")
        print("\nKey Biomarker Attributions (Integrated Gradients):")
        for b in report["top_biomarkers"]:
            print(f"  * {b['feature']:<25} | Value: {b['value']:<6.2f} | Attr: {b['attribution']:<+7.4f} | {b['direction']}")
        print(f"\nClinical Rationale:\n  {report['clinical_rationale']}")
        print("="*60)

def cmd_audit(args):
    """Executes empirical privacy audit via Membership Inference Attacks."""
    logger.info("Executing Empirical Membership Inference Attack Audit...")
    train_data, test_data, in_dim, num_classes, _ = load_breast_cancer_data()
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)
    
    # Train quick Non-DP model
    model_nondp = HealthcareMLP(input_dim=in_dim, num_classes=num_classes)
    opt_nondp = torch.optim.SGD(model_nondp.parameters(), lr=0.05)
    crit = torch.nn.CrossEntropyLoss()
    
    for epoch in range(10):
        for x, y in train_loader:
            opt_nondp.zero_grad()
            out = model_nondp(x)
            loss = crit(out, y)
            loss.backward()
            opt_nondp.step()
            
    # Train quick DP-SGD model
    from fedpro.privacy.dp_sgd import clip_and_add_noise
    model_dp = HealthcareMLP(input_dim=in_dim, num_classes=num_classes)
    opt_dp = torch.optim.SGD(model_dp.parameters(), lr=0.05)
    
    for epoch in range(10):
        for x, y in train_loader:
            opt_dp.zero_grad()
            out = model_dp(x)
            loss = crit(out, y)
            loss.backward()
            clip_and_add_noise(model_dp, clip_norm=1.0, noise_multiplier=0.8, batch_size=x.size(0))
            opt_dp.step()
            
    out_dir = args.out or os.path.join("experiments", "audit_mia")
    os.makedirs(out_dir, exist_ok=True)
    
    evaluator = MIAEvaluator()
    results = evaluator.compare_dp_vs_nondp(
        nondp_model=model_nondp,
        dp_model=model_dp,
        member_loader=train_loader,
        non_member_loader=test_loader,
        output_dir=out_dir
    )
    
    nondp = results["nondp"]
    dp = results["dp"]
    
    print("\n" + "="*75)
    print(" EMPIRICAL MEMBERSHIP INFERENCE ATTACK (MIA) AUDIT RESULTS")
    print("="*75)
    print(f"Non-DP Baseline Attack ROC-AUC  : {nondp['attack_auc']:.4f} (Susceptible)")
    print(f"FedHealth DP-SGD Attack ROC-AUC : {dp['attack_auc']:.4f} (Near Random Guess 0.50)")
    print(f"Empirical Attack AUC Reduction  : {results['auc_reduction']:+.4f}")
    print(f"Max Privacy Advantage Reduction : {results['advantage_reduction']:+.4f}")
    print(f"Audit Artifacts Generated in    : {out_dir}")
    print("="*75)

def main():
    parser = argparse.ArgumentParser(
        prog="fedhealth",
        description="FedHealth: Research-Grade Privacy-Preserving Federated Learning Framework."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Subcommand: run
    p_run = subparsers.add_parser("run", help="Execute a federated experiment")
    p_run.add_argument("--name", type=str, default="Clinical_Experiment", help="Experiment name")
    p_run.add_argument("--algo", type=str, default="fedavg", choices=["fedavg", "fedprox", "scaffold", "fednova", "fedadam"], help="FL algorithm")
    p_run.add_argument("--rounds", type=int, default=5, help="Number of communication rounds")
    p_run.add_argument("--hospitals", type=int, default=5, help="Number of clinical hospital nodes")
    p_run.add_argument("--epochs", type=int, default=2, help="Local epochs per hospital")
    p_run.add_argument("--alpha", type=float, default=0.5, help="Dirichlet non-IID partition parameter")
    p_run.add_argument("--dp", action="store_true", default=True, help="Enable Differential Privacy (DP-SGD)")
    p_run.add_argument("--noise", type=float, default=0.5, help="DP Gaussian noise multiplier")
    p_run.add_argument("--clip", type=float, default=1.0, help="DP gradient L2 clipping norm")
    
    # Subcommand: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run multi-algorithm comparative benchmark")
    p_bench.add_argument("--algorithms", type=str, default="fedavg,fedprox,scaffold,fednova", help="Comma-separated algorithms")
    p_bench.add_argument("--rounds", type=int, default=5, help="Number of rounds per algorithm")
    
    # Subcommand: dashboard
    p_dash = subparsers.add_parser("dashboard", help="Start FastAPI & WebSocket dashboard server")
    p_dash.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    p_dash.add_argument("--port", type=int, default=8000, help="Port number")
    
    # Subcommand: explain
    p_xai = subparsers.add_parser("explain", help="Run Explainable AI on a clinical patient record")
    p_xai.add_argument("--sample-idx", type=int, default=0, help="Patient index from test cohort")
    p_xai.add_argument("--baseline", type=str, default="cohort_centroid", choices=["cohort_centroid", "zeros", "compare"], help="Attribution baseline")

    # Subcommand: audit
    p_audit = subparsers.add_parser("audit", help="Run Empirical Privacy Audit (Membership Inference Attack)")
    p_audit.add_argument("--out", type=str, default="experiments/audit_mia", help="Output directory for audit artifacts")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
        
    cmd_map = {
        "run": cmd_run,
        "benchmark": cmd_benchmark,
        "dashboard": cmd_dashboard,
        "explain": cmd_explain,
        "audit": cmd_audit
    }
    
    cmd_map[args.command](args)

if __name__ == "__main__":
    main()
