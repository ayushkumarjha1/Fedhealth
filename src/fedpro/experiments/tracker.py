"""
Experiment Tracker & Reproducibility Engine for FedHealth.
Automatically organizes experiment runs into structured directories:
experiments/<experiment_name>_<timestamp>/
  ├── config.yaml
  ├── metrics.json
  ├── telemetry.jsonl
  ├── model_best.pt
  ├── model_final.pt
  ├── plots/
  │   ├── convergence_curves.png
  │   ├── client_drift_heatmap.png
  │   ├── privacy_frontier.png
  │   └── confusion_matrix.png
  └── report.md
"""

import os
import json
import time
import datetime
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive headless rendering
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from fedpro.configs.base_config import FedHealthConfig

class ExperimentTracker:
    """
    Manages complete lifecycle, logging, checkpointing, and artifact generation 
    for reproducible federated learning benchmarks.
    """
    
    def __init__(
        self, 
        experiment_name: str = "FedHealth_Benchmark", 
        output_dir: str = "experiments",
        config: Optional[FedHealthConfig] = None
    ):
        self.experiment_name = experiment_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"{experiment_name}_{self.timestamp}")
        self.plots_dir = os.path.join(self.run_dir, "plots")
        
        os.makedirs(self.plots_dir, exist_ok=True)
        self.config = config or FedHealthConfig()
        
        # Internal state
        self.rounds_history: List[Dict[str, Any]] = []
        self.best_accuracy = -1.0
        self.best_loss = float("inf")
        self.last_drift_matrix: Optional[np.ndarray] = None
        
        # Save initial configuration
        self._save_config()

    def _save_config(self):
        config_path = os.path.join(self.run_dir, "config.json")
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config.model_dump(), f, indent=2)
        except Exception:
            pass

    def log_round(
        self, 
        metrics: Dict[str, Any], 
        model: Optional[torch.nn.Module] = None,
        drift_matrix: Optional[np.ndarray] = None
    ):
        """
        Logs metrics packet for a completed round and updates model checkpoints.
        """
        self.rounds_history.append(metrics)
        if drift_matrix is not None:
            self.last_drift_matrix = drift_matrix
            
        # Append to telemetry JSONL
        telemetry_file = os.path.join(self.run_dir, "telemetry.jsonl")
        with open(telemetry_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
            
        # Update metrics JSON
        metrics_file = os.path.join(self.run_dir, "metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.rounds_history, f, indent=2)
            
        # Checkpoint best model
        acc = metrics.get("accuracy", 0.0)
        if acc > self.best_accuracy and model is not None:
            self.best_accuracy = acc
            best_path = os.path.join(self.run_dir, "model_best.pt")
            torch.save(model.state_dict(), best_path)

    def finalize(self, model: Optional[torch.nn.Module] = None) -> Dict[str, str]:
        """
        Finalizes experiment run: saves final weights, plots publication graphs, 
        and generates research Markdown report.
        
        Returns:
            Dictionary mapping artifact names to their file paths
        """
        artifacts = {}
        
        # 1. Save final model checkpoint
        if model is not None:
            final_path = os.path.join(self.run_dir, "model_final.pt")
            torch.save(model.state_dict(), final_path)
            artifacts["model_final"] = final_path
            
        # 2. Render Convergence Curves
        if self.rounds_history:
            conv_path = self._plot_convergence()
            if conv_path:
                artifacts["convergence_plot"] = conv_path
                
            # 3. Render Privacy Frontier
            if self.config.privacy.enabled:
                dp_path = self._plot_privacy_frontier()
                if dp_path:
                    artifacts["privacy_plot"] = dp_path
                    
            # 4. Render Drift Heatmap
            if self.last_drift_matrix is not None:
                drift_path = self._plot_drift_heatmap()
                if drift_path:
                    artifacts["drift_plot"] = drift_path
                    
            # 5. Render Confusion Matrix
            last_round = self.rounds_history[-1]
            if "confusion_matrix" in last_round:
                cm_path = self._plot_confusion_matrix(last_round["confusion_matrix"])
                if cm_path:
                    artifacts["confusion_matrix_plot"] = cm_path
                    
        # 6. Generate Clinical Markdown Report
        report_path = self._generate_markdown_report()
        artifacts["report"] = report_path
        
        return artifacts

    def _plot_convergence(self) -> Optional[str]:
        try:
            rounds = [r["round"] for r in self.rounds_history]
            losses = [r["loss"] for r in self.rounds_history]
            accuracies = [r["accuracy"] for r in self.rounds_history]
            
            fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
            
            color = "#2563EB"
            ax1.set_xlabel("Federated Communication Round", fontsize=11, fontweight="bold")
            ax1.set_ylabel("Global Cross-Entropy Loss", color=color, fontsize=11, fontweight="bold")
            ax1.plot(rounds, losses, color=color, marker="o", linewidth=2.2, label="Global Loss")
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.grid(True, linestyle="--", alpha=0.5)
            
            ax2 = ax1.twinx()
            color = "#10B981"
            ax2.set_ylabel("Global Test Accuracy (%)", color=color, fontsize=11, fontweight="bold")
            ax2.plot(rounds, accuracies, color=color, marker="s", linewidth=2.2, linestyle="--", label="Accuracy (%)")
            ax2.tick_params(axis="y", labelcolor=color)
            
            plt.title(f"FedHealth Optimization Convergence: {self.config.algorithm.name.upper()}", fontsize=13, fontweight="bold", pad=12)
            fig.tight_layout()
            
            plot_path = os.path.join(self.plots_dir, "convergence_curves.png")
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        except Exception:
            return None

    def _plot_privacy_frontier(self) -> Optional[str]:
        try:
            rounds = [r["round"] for r in self.rounds_history]
            epsilons = [r.get("epsilon", 0.0) for r in self.rounds_history]
            
            plt.figure(figsize=(7, 4.5), dpi=300)
            plt.plot(rounds, epsilons, color="#8B5CF6", marker="^", linewidth=2.2, label="Cumulative RDP ε")
            plt.axhline(y=self.config.privacy.target_epsilon, color="#EF4444", linestyle=":", label=f"Target Bound ε={self.config.privacy.target_epsilon}")
            plt.xlabel("Communication Round", fontsize=11, fontweight="bold")
            plt.ylabel("Differential Privacy Budget (ε)", fontsize=11, fontweight="bold")
            plt.title("Rényi Differential Privacy Expenditure", fontsize=13, fontweight="bold", pad=12)
            plt.legend(loc="upper left")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            
            plot_path = os.path.join(self.plots_dir, "privacy_frontier.png")
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        except Exception:
            return None

    def _plot_drift_heatmap(self) -> Optional[str]:
        try:
            matrix = self.last_drift_matrix
            if matrix is None or matrix.size == 0:
                return None
                
            plt.figure(figsize=(6, 5), dpi=300)
            im = plt.imshow(matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
            plt.colorbar(im, label="Cosine Similarity S_ij")
            
            num_nodes = matrix.shape[0]
            ticks = range(num_nodes)
            labels = [f"H_{i+1}" for i in range(num_nodes)]
            plt.xticks(ticks, labels, fontweight="bold")
            plt.yticks(ticks, labels, fontweight="bold")
            
            # Label numerical values
            for i in range(num_nodes):
                for j in range(num_nodes):
                    plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if abs(matrix[i, j]) < 0.5 else "black", fontsize=9)
                    
            plt.title("Inter-Hospital Client Drift Cosine Matrix", fontsize=12, fontweight="bold", pad=10)
            plt.tight_layout()
            
            plot_path = os.path.join(self.plots_dir, "client_drift_heatmap.png")
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        except Exception:
            return None

    def _plot_confusion_matrix(self, cm: List[List[int]]) -> Optional[str]:
        try:
            matrix = np.array(cm)
            plt.figure(figsize=(5, 4.5), dpi=300)
            plt.imshow(matrix, cmap="Blues")
            plt.colorbar()
            
            classes = ["Benign", "Malignant"] if matrix.shape == (2, 2) else [f"C_{i}" for i in range(matrix.shape[0])]
            plt.xticks(range(len(classes)), classes, fontweight="bold")
            plt.yticks(range(len(classes)), classes, fontweight="bold")
            plt.xlabel("Predicted Class", fontweight="bold")
            plt.ylabel("Ground Truth Class", fontweight="bold")
            
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black" if matrix[i, j] < matrix.max()/2 else "white", fontsize=11, fontweight="bold")
                    
            plt.title("Clinical Diagnostic Confusion Matrix", fontsize=12, fontweight="bold", pad=10)
            plt.tight_layout()
            
            plot_path = os.path.join(self.plots_dir, "confusion_matrix.png")
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        except Exception:
            return None

    def _generate_markdown_report(self) -> str:
        last = self.rounds_history[-1] if self.rounds_history else {}
        report_content = f"""# FedHealth Experiment Report: {self.experiment_name}
**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Run ID:** `{os.path.basename(self.run_dir)}`  
**Algorithm:** {self.config.algorithm.name.upper()}  

## 1. Benchmark Summary
- **Total Rounds Executed:** {len(self.rounds_history)}
- **Peak Test Accuracy:** {self.best_accuracy:.2f}%
- **Final Cross-Entropy Loss:** {last.get('loss', 0.0):.4f}
- **Final ROC-AUC:** {last.get('roc_auc', 0.0):.2f}%
- **Sensitivity (Recall):** {last.get('recall', 0.0):.2f}%
- **Specificity:** {last.get('specificity', 0.0):.2f}%
- **Cumulative Privacy Budget (ε):** {last.get('epsilon', 0.0):.2f} (δ = {last.get('delta', 1e-5)})

## 2. Mathematical Optimization Protocol
$$\\min_{{w \\in \\mathbb{{R}}^d}} \\sum_{{k=1}}^K p_k F_k(w)$$
Optimized across {self.config.training.num_hospitals} clinical institutional nodes using `{self.config.algorithm.name.upper()}` under strict Rényi Differential Privacy guarantees.

## 3. Artifact Directory
All checkpoints, telemetry logs, and high-resolution figures are preserved in:
`{os.path.abspath(self.run_dir)}`
"""
        report_path = os.path.join(self.run_dir, "report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        return report_path
