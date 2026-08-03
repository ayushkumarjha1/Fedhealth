"""
Experiment Tracker and Checkpoint Manager for FedHealth.
Logs training metrics to JSON, CSV, and saves best model checkpoints.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import time

class ExperimentTracker:
    """Tracks global and local metrics over time, persisting histories and model checkpoints."""
    
    def __init__(self, experiment_name: str = "FedHealth_Run", output_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.timestamp = int(time.time())
        self.run_id = f"{experiment_name}_{self.timestamp}"
        self.output_dir = Path(output_dir) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rounds_history: List[Dict[str, Any]] = []
        self.hospitals_history: List[Dict[str, Any]] = []
        self.best_accuracy = -1.0
        self.best_model_path: Optional[str] = None
        
    def log_round(self, round_data: Dict[str, Any], model: Optional[nn.Module] = None):
        """Record round evaluation metrics and update best model checkpoint."""
        self.rounds_history.append(round_data)
        
        # Check for best model
        acc = round_data.get("accuracy", 0.0)
        if acc > self.best_accuracy and model is not None:
            self.best_accuracy = acc
            self.best_model_path = str(self.output_dir / "best_model.pt")
            torch.save(model.state_dict(), self.best_model_path)
            
        # Write JSON snapshots
        with open(self.output_dir / "metrics_history.json", "w", encoding="utf-8") as f:
            json.dump(self.rounds_history, f, indent=2)
            
        # Write CSV
        self._write_csv()

    def _write_csv(self):
        """Export metrics history to CSV."""
        if not self.rounds_history:
            return
        csv_path = self.output_dir / "metrics_history.csv"
        keys = list(self.rounds_history[0].keys())
        # Filter out complex nested fields for CSV
        simple_keys = [k for k in keys if not isinstance(self.rounds_history[0][k], (list, dict))]
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=simple_keys)
            writer.writeheader()
            for row in self.rounds_history:
                filtered_row = {k: row.get(k, "") for k in simple_keys}
                writer.writerow(filtered_row)

    def get_summary(self) -> Dict[str, Any]:
        """Return high-level experiment summary."""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "total_rounds": len(self.rounds_history),
            "best_accuracy": self.best_accuracy,
            "best_model_path": self.best_model_path,
            "history": self.rounds_history
        }
