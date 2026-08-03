"""
Digital Twin Hospital Node Simulator for FedHealth.
Simulates real-world medical institutions with heterogeneous compute, network dynamics,
communication costs, latency jitter, and operational status.
"""

from typing import Dict, Any, List, Optional
import random
import time

# Standard Realistic Medical Center Profiles
DEFAULT_HOSPITAL_PROFILES: List[Dict[str, Any]] = [
    {
        "id": "Hospital_1",
        "name": "Mayo Clinic Comprehensive Cancer Center",
        "department": "Oncology & Radiology",
        "location": "Rochester, MN",
        "compute_device": "cuda:0 (NVIDIA A100)",
        "base_latency_ms": 32.0,
        "bandwidth_mbps": 500.0,
        "reliability": 0.99,
        "trust_score": 0.98,
        "privacy_policy": "Strict HIPAA + RDP (ε < 3.0)"
    },
    {
        "id": "Hospital_2",
        "name": "Johns Hopkins Medicine",
        "department": "Pathology & Precision Medicine",
        "location": "Baltimore, MD",
        "compute_device": "cuda:0 (NVIDIA RTX 4090)",
        "base_latency_ms": 48.0,
        "bandwidth_mbps": 250.0,
        "reliability": 0.96,
        "trust_score": 0.95,
        "privacy_policy": "HIPAA Tier-2 + DP-SGD"
    },
    {
        "id": "Hospital_3",
        "name": "Cleveland Clinic Foundation",
        "department": "Genomic Medicine Institute",
        "location": "Cleveland, OH",
        "compute_device": "cpu (Intel Xeon Platinum)",
        "base_latency_ms": 65.0,
        "bandwidth_mbps": 100.0,
        "reliability": 0.94,
        "trust_score": 0.92,
        "privacy_policy": "Institutional IRB Approved"
    },
    {
        "id": "Hospital_4",
        "name": "Stanford Health Care",
        "department": "Biomedical Informatics",
        "location": "Palo Alto, CA",
        "compute_device": "cuda:0 (NVIDIA H100)",
        "base_latency_ms": 82.0,
        "bandwidth_mbps": 1000.0,
        "reliability": 0.98,
        "trust_score": 0.97,
        "privacy_policy": "California CMIA / HIPAA"
    },
    {
        "id": "Hospital_5",
        "name": "Massachusetts General Hospital",
        "department": "Clinical Data Science Center",
        "location": "Boston, MA",
        "compute_device": "cuda:0 (NVIDIA V100)",
        "base_latency_ms": 55.0,
        "bandwidth_mbps": 300.0,
        "reliability": 0.95,
        "trust_score": 0.94,
        "privacy_policy": "Strict DP (ε < 5.0, δ = 1e-5)"
    }
]

class DigitalTwinHospital:
    """
    Simulates a connected hospital participant node with telemetry and network physics.
    """
    
    def __init__(self, profile: Dict[str, Any], dataset_size: int = 1000):
        self.id = profile["id"]
        self.name = profile["name"]
        self.department = profile.get("department", "General Clinical AI")
        self.location = profile.get("location", "USA")
        self.compute_device = profile.get("compute_device", "cpu")
        self.base_latency_ms = profile.get("base_latency_ms", 50.0)
        self.bandwidth_mbps = profile.get("bandwidth_mbps", 100.0)
        self.reliability = profile.get("reliability", 0.95)
        self.trust_score = profile.get("trust_score", 0.95)
        self.privacy_policy = profile.get("privacy_policy", "HIPAA Compliant")
        self.dataset_size = dataset_size
        
        # Dynamic State
        self.status = "Idle" # Idle, Training, Uploading, Straggler, Offline
        self.current_loss = 1.0
        self.current_acc = 0.50
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.cumulative_training_time_sec = 0.0
        self.last_round_participated = 0
        self.is_active = True
        self.straggler_count = 0
        
    def simulate_network_delay(self, payload_bytes: int) -> float:
        """
        Calculates realistic round-trip transmission latency including bandwidth transfer time and jitter.
        
        Returns:
            Simulated latency in milliseconds
        """
        # Bandwidth transmission time in ms: (bytes * 8 bits/byte) / (bandwidth_mbps * 1e6 bits/sec) * 1000 ms/sec
        transfer_time_ms = (payload_bytes * 8.0) / (self.bandwidth_mbps * 1e6) * 1000.0
        # Latency Jitter (Gaussian perturbation)
        jitter_ms = random.gauss(0, self.base_latency_ms * 0.15)
        total_latency_ms = max(5.0, self.base_latency_ms + transfer_time_ms + jitter_ms)
        return total_latency_ms

    def check_straggler(self, straggler_probability: float = 0.08) -> bool:
        """Determines if the hospital experiences transient compute or network throttling."""
        is_straggler = random.random() < straggler_probability
        if is_straggler:
            self.straggler_count += 1
            self.status = "Straggler"
        return is_straggler

    def update_telemetry(
        self, 
        status: str, 
        loss: float, 
        acc: float, 
        bytes_sent: int, 
        round_num: int,
        training_time: float
    ):
        """Update live telemetry after a completed training step."""
        self.status = status
        self.current_loss = float(loss)
        self.current_acc = float(acc)
        self.total_bytes_sent += bytes_sent
        self.last_round_participated = round_num
        self.cumulative_training_time_sec += training_time

    def to_dict(self) -> Dict[str, Any]:
        """Serialize hospital state to dictionary for WebSocket broadcasting and dashboard display."""
        return {
            "id": self.id,
            "name": self.name,
            "department": self.department,
            "location": self.location,
            "compute_device": self.compute_device,
            "dataset_size": self.dataset_size,
            "status": self.status,
            "current_loss": round(self.current_loss, 4),
            "current_acc": round(self.current_acc * 100, 2),
            "total_mb_transferred": round(self.total_bytes_sent / (1024 * 1024), 2),
            "base_latency_ms": round(self.base_latency_ms, 1),
            "bandwidth_mbps": self.bandwidth_mbps,
            "trust_score": self.trust_score,
            "privacy_policy": self.privacy_policy,
            "straggler_count": self.straggler_count,
            "is_active": self.is_active
        }
