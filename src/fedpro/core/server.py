"""
Production-grade Federated Learning Server Orchestrator for FedHealth.
Coordinates multi-hospital rounds, algorithm aggregation, Differential Privacy,
quantitative AI Copilot diagnostics, replay snapshotting, and real-time telemetry broadcasting.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Callable

from fedpro.core.base import BaseFLServer, BaseFLClient, ClientUpdate, ServerEvaluation
from fedpro.core.hospital import DigitalTwinHospital, DEFAULT_HOSPITAL_PROFILES
from fedpro.algorithms.registry import get_algorithm_aggregator, get_algorithm_client_class
from fedpro.algorithms.fedopt import FedOptServerOptimizer
from fedpro.privacy.rdp_accountant import RDPAccountant
from fedpro.copilot.copilot_engine import AIFederatedCopilot
from fedpro.replay.replay_engine import FederatedTrainingReplay
from fedpro.xai.xai_engine import ClinicalExplainer
from fedpro.evaluation.metrics import compute_clinical_metrics
from fedpro.experiments.tracker import ExperimentTracker
from fedpro.reports.report_generator import ResearchReportGenerator
from fedpro.configs.base_config import FedHealthConfig
from fedpro.utils.logger import get_logger

logger = get_logger("FLServer")

class FLServer(BaseFLServer):
    """
    Central Coordinator and Orchestrator for Federated Clinical Diagnostics.
    """
    
    def __init__(
        self, 
        global_model: nn.Module, 
        config: Optional[FedHealthConfig] = None,
        telemetry_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        feature_names: Optional[List[str]] = None
    ):
        super().__init__(global_model)
        self.config = config or FedHealthConfig()
        self.telemetry_callback = telemetry_callback
        self.feature_names = feature_names or [f"Biomarker_{i+1}" for i in range(30)]
        
        # Subsystems
        self.privacy_accountant = RDPAccountant(target_delta=self.config.privacy.target_delta)
        self.copilot = AIFederatedCopilot(target_privacy_epsilon=self.config.privacy.target_epsilon or 10.0)
        self.replay = FederatedTrainingReplay(experiment_id=self.config.experiment_name)
        self.tracker = ExperimentTracker(experiment_name=self.config.experiment_name, config=self.config)
        self.explainer = ClinicalExplainer(self.global_model, feature_names=self.feature_names)
        self.report_generator = ResearchReportGenerator()
        
        # Server-side optimizer if using FedOpt
        algo_name = self.config.algorithm.name.lower()
        self.server_optimizer = None
        if algo_name in ["fedadam", "fedyogi"]:
            self.server_optimizer = FedOptServerOptimizer(
                mode=algo_name,
                lr=self.config.algorithm.server_lr,
                beta1=self.config.algorithm.server_beta1,
                beta2=self.config.algorithm.server_beta2,
                tau=self.config.algorithm.server_tau
            )
            
        # SCAFFOLD Server Control Variates
        self.server_control_variate: Dict[str, torch.Tensor] = {}
        
        # Initialize Digital Twin Hospitals
        self.hospitals: List[DigitalTwinHospital] = []
        for i in range(self.config.training.num_hospitals):
            profile = dict(DEFAULT_HOSPITAL_PROFILES[i % len(DEFAULT_HOSPITAL_PROFILES)])
            profile["id"] = f"hospital_{i+1}"
            hospital = DigitalTwinHospital(profile)
            self.hospitals.append(hospital)
            
        # Cached round telemetry
        self.last_drift_matrix = None
        self.last_mean_drift = 1.0
        self.last_grad_snr = 1.0

    def broadcast(self, event_type: str, data: Dict[str, Any]):
        """Broadcasts telemetry payload to subscribers (WebSockets / Callbacks)."""
        payload = {"event": event_type, "timestamp": time.time(), "data": data}
        if self.telemetry_callback:
            try:
                self.telemetry_callback(payload)
            except Exception as e:
                logger.warning(f"Telemetry callback broadcast exception: {e}")

    def fit_round(
        self, 
        round_num: int, 
        clients: List[BaseFLClient], 
        dataloaders: List[DataLoader], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes one full synchronous communication round across hospital clients.
        """
        logger.info(f"--- Commencing Federated Round {round_num}/{self.config.training.num_rounds} ---")
        self.broadcast("round_start", {"round": round_num, "num_hospitals": len(clients)})
        
        global_params = self.get_parameters()
        client_updates: List[ClientUpdate] = []
        local_step_counts: List[int] = []
        
        # 1. Dispatch & Local Hospital Optimization
        for client, loader, hospital in zip(clients, dataloaders, self.hospitals):
            hospital.update_telemetry(status="Training", loss=hospital.current_loss, acc=hospital.current_acc, bytes_sent=0, round_num=round_num, training_time=0.0)
            
            is_straggler = hospital.check_straggler()
            local_epochs = self.config.training.local_epochs
            if is_straggler:
                local_epochs = max(1, local_epochs // 2)
                
            client_cfg = dict(config)
            client_cfg["epochs"] = local_epochs
            if self.config.algorithm.name.lower() == "scaffold":
                client_cfg["server_control_variate"] = self.server_control_variate
                
            # Fit client
            update = client.fit(global_params, loader, client_cfg)
            update.is_straggler = is_straggler
            
            # Simulate network transmission
            latency_ms = hospital.simulate_network_delay(update.bytes_transferred)
            update.simulated_latency_ms = latency_ms
            
            # Update hospital telemetry
            hospital.update_telemetry(
                status="Idle",
                loss=update.metrics.get("loss", 1.0),
                acc=update.metrics.get("accuracy", 0.0),
                bytes_sent=update.bytes_transferred,
                round_num=round_num,
                training_time=update.computation_time_sec
            )
            
            client_updates.append(update)
            local_step_counts.append(len(loader) * local_epochs)
            
            logger.info(
                f"  [{hospital.name}] Loss: {update.metrics.get('loss', 0.0):.4f} | "
                f"Bytes: {update.bytes_transferred/1024:.1f} KB | Latency: {latency_ms:.1f} ms"
            )
            
        # 2. Compute Quantitative Statistical Telemetry (Client Drift & SNR)
        client_deltas = []
        for u in client_updates:
            delta_dict = {
                k: u.parameters[k].cpu() - global_params[k].cpu()
                for k in global_params.keys()
            }
            client_deltas.append(delta_dict)
            
        self.last_drift_matrix, self.last_mean_drift = self.copilot.compute_client_drift_matrix(client_deltas)
        self.last_grad_snr = self.copilot.compute_gradient_snr(client_deltas)

        # 3. Aggregate parameters
        algo_name = self.config.algorithm.name.lower()
        if self.server_optimizer is not None:
            new_global_params = self.server_optimizer.step(global_params, client_updates)
        elif algo_name == "fednova":
            from fedpro.algorithms.fednova import aggregate_fednova
            new_global_params = aggregate_fednova(global_params, client_updates, local_step_counts)
        else:
            from fedpro.algorithms.fedavg import aggregate_fedavg
            new_global_params = aggregate_fedavg(client_updates)
            
        # Update SCAFFOLD server control variates
        if algo_name == "scaffold":
            for update in client_updates:
                if update.control_variate_delta:
                    for k, delta in update.control_variate_delta.items():
                        if k not in self.server_control_variate:
                            self.server_control_variate[k] = torch.zeros_like(delta)
                        self.server_control_variate[k] += delta / float(len(client_updates))
                        
        self.set_parameters(new_global_params)
        
        # 4. Step Server Differential Privacy Accountant
        if self.config.privacy.enabled:
            total_samples = sum(u.num_samples for u in client_updates)
            sample_rate = float(self.config.training.batch_size) / float(max(1, total_samples))
            steps_in_round = sum(local_step_counts)
            self.privacy_accountant.step(
                noise_multiplier=self.config.privacy.noise_multiplier,
                sample_rate=sample_rate,
                num_steps=steps_in_round
            )
            
        return {
            "round": round_num,
            "num_clients": len(client_updates),
            "client_updates": client_updates,
            "drift_similarity": self.last_mean_drift,
            "gradient_snr": self.last_grad_snr
        }

    def evaluate_round(
        self, 
        round_num: int, 
        evaluator_client: BaseFLClient, 
        test_loader: DataLoader
    ) -> ServerEvaluation:
        """
        Evaluate current global model performance against test dataset.
        """
        start_time = time.time()
        self.global_model.eval()
        device = next(self.global_model.parameters()).device
        
        all_targets = []
        all_preds = []
        all_probs = []
        total_loss = 0.0
        num_samples = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.global_model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                
                probs = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                
                all_targets.extend(target.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())
                all_probs.extend(probs[:, 1].cpu().numpy().tolist() if probs.shape[1] > 1 else probs[:, 0].cpu().numpy().tolist())
                num_samples += data.size(0)
                
        avg_loss = total_loss / max(1, num_samples)
        metrics = compute_clinical_metrics(
            y_true=torch.tensor(all_targets).numpy(),
            y_pred=torch.tensor(all_preds).numpy(),
            y_prob=torch.tensor(all_probs).numpy(),
            loss=avg_loss
        )
        
        eval_time = time.time() - start_time
        server_eval = ServerEvaluation(
            round_num=round_num,
            loss=metrics["loss"],
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            roc_auc=metrics["roc_auc"],
            specificity=metrics["specificity"],
            confusion_matrix=metrics["confusion_matrix"],
            evaluation_time_sec=eval_time
        )
        
        # Privacy metrics
        priv_summary = self.privacy_accountant.get_privacy_spent()
        priv_summary["enabled"] = self.config.privacy.enabled
        priv_summary["clip_norm"] = self.config.privacy.clip_norm
        priv_summary["noise_multiplier"] = self.config.privacy.noise_multiplier
        
        # Log to Experiment Tracker
        log_packet = {
            "round": round_num,
            "loss": server_eval.loss,
            "accuracy": server_eval.accuracy,
            "precision": server_eval.precision,
            "recall": server_eval.recall,
            "f1_score": server_eval.f1_score,
            "roc_auc": server_eval.roc_auc,
            "specificity": server_eval.specificity,
            "epsilon": priv_summary.get("epsilon", 0.0),
            "delta": priv_summary.get("delta", 1e-5),
            "drift_similarity": self.last_mean_drift,
            "gradient_snr": self.last_grad_snr,
            "confusion_matrix": server_eval.confusion_matrix
        }
        self.tracker.log_round(log_packet, model=self.global_model, drift_matrix=self.last_drift_matrix)
        
        # Run AI Copilot diagnostic
        hospitals_dict = [h.to_dict() for h in self.hospitals]
        copilot_insight = self.copilot.analyze_round(
            round_num=round_num,
            metrics_history=self.tracker.rounds_history,
            hospital_updates=hospitals_dict,
            privacy_metrics=priv_summary,
            algorithm_name=self.config.algorithm.name,
            drift_similarity=self.last_mean_drift,
            gradient_snr=self.last_grad_snr
        )
        
        # Save snapshot for Training Replay
        self.replay.record_round_snapshot(
            round_num=round_num,
            global_metrics=log_packet,
            hospitals_state=hospitals_dict,
            privacy_state=priv_summary,
            copilot_insight=copilot_insight
        )
        
        # Compute global XAI feature importance on final round or periodically
        xai_features = []
        if round_num == self.config.training.num_rounds or round_num % 5 == 0:
            xai_features = self.explainer.explain_global_features(test_loader)
            
        # Broadcast full telemetry
        self.broadcast("round_end", {
            "round": round_num,
            "metrics": log_packet,
            "hospitals": hospitals_dict,
            "privacy": priv_summary,
            "copilot": copilot_insight,
            "xai": xai_features
        })
        
        logger.info(
            f"Round {round_num} Global Evaluation => Loss: {server_eval.loss:.4f} | "
            f"Accuracy: {server_eval.accuracy:.2f}% | ROC-AUC: {server_eval.roc_auc:.2f}% | "
            f"Epsilon (eps): {priv_summary.get('epsilon', 0.0):.2f}"
        )
        return server_eval
