"""
Federated Training Replay Engine for FedHealth.
Captures round-by-round state snapshots for time-travel debugging and interactive experiment playback.
"""

from typing import List, Dict, Any, Optional
import json
import time

class FederatedTrainingReplay:
    """
    Records and serves state snapshots of all global and client-level metrics for time-travel playback.
    """
    
    def __init__(self, experiment_id: str = "default_run"):
        self.experiment_id = experiment_id
        self.snapshots: List[Dict[str, Any]] = []

    def record_round_snapshot(
        self,
        round_num: int,
        global_metrics: Dict[str, Any],
        hospitals_state: List[Dict[str, Any]],
        privacy_state: Dict[str, Any],
        copilot_insight: Optional[Dict[str, Any]] = None
    ):
        """Record an immutable state snapshot of a single federated round."""
        snapshot = {
            "round": round_num,
            "timestamp": time.time(),
            "global_metrics": global_metrics,
            "hospitals": [dict(h) for h in hospitals_state],
            "privacy": dict(privacy_state),
            "copilot": copilot_insight or {}
        }
        self.snapshots.append(snapshot)

    def get_total_rounds(self) -> int:
        return len(self.snapshots)

    def get_snapshot(self, round_num: int) -> Optional[Dict[str, Any]]:
        """Retrieve state for a specific round (1-indexed)."""
        if 1 <= round_num <= len(self.snapshots):
            return self.snapshots[round_num - 1]
        return None

    def get_all_snapshots(self) -> List[Dict[str, Any]]:
        """Return full chronological snapshot stream."""
        return self.snapshots

    def export_json(self) -> str:
        """Export replay log as serialized JSON string."""
        return json.dumps({
            "experiment_id": self.experiment_id,
            "total_rounds": len(self.snapshots),
            "snapshots": self.snapshots
        }, indent=2)
