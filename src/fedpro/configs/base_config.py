"""
Production-grade configuration management for FedHealth using Pydantic v2.
Supports YAML, JSON, environment variables, and CLI overrides.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
import json
from pathlib import Path

class DPConfig(BaseModel):
    """Differential Privacy (DP-SGD) Configuration."""
    enabled: bool = Field(default=True, description="Whether DP-SGD is active during local training")
    clip_norm: float = Field(default=1.0, ge=0.01, le=100.0, description="Max gradient norm clipping threshold (C)")
    noise_multiplier: float = Field(default=0.5, ge=0.0, le=10.0, description="Gaussian noise scale multiplier (sigma)")
    target_delta: float = Field(default=1e-5, gt=0.0, lt=1.0, description="Target delta for (epsilon, delta)-DP bound")
    target_epsilon: Optional[float] = Field(default=10.0, gt=0.0, description="Target allowable privacy budget epsilon")
    max_epsilon: Optional[float] = Field(default=10.0, gt=0.0, description="Maximum allowable privacy budget epsilon before halting")
    accountant_type: Literal["rdp", "moments", "basic"] = Field(default="rdp", description="Privacy accounting mechanism")

class ModelConfig(BaseModel):
    """Neural Network Model Configuration."""
    architecture: str = Field(default="HealthcareMLP", description="Model architecture name (e.g. HealthcareMLP, MedicalCNN, ResNet18)")
    input_dim: Optional[int] = Field(default=30, description="Input feature dimension for tabular models")
    hidden_dims: List[int] = Field(default_factory=lambda: [64, 32, 16], description="Hidden layer dimensions for MLP")
    num_classes: int = Field(default=2, ge=2, description="Number of output prediction classes")
    dropout_rate: float = Field(default=0.2, ge=0.0, le=0.9, description="Dropout regularization probability")
    use_batch_norm: bool = Field(default=True, description="Whether to include Batch Normalization layers")

class PartitionConfig(BaseModel):
    """Dataset Partitioning Configuration."""
    dataset_name: str = Field(default="breast_cancer", description="Dataset identifier (breast_cancer, heart_disease, diabetes, medmnist)")
    partition_type: Literal["iid", "dirichlet", "pathological", "quantity_skew"] = Field(default="dirichlet", description="Partition strategy")
    dirichlet_alpha: float = Field(default=0.5, gt=0.0, description="Dirichlet distribution concentration parameter alpha (lower = higher Non-IID skew)")
    num_shards_per_client: int = Field(default=2, ge=1, description="Number of class shards for pathological partition")
    test_split_ratio: float = Field(default=0.2, gt=0.0, lt=0.5, description="Holdout validation/test set fraction")
    random_seed: int = Field(default=42, description="Random seed for reproducible data splits")

class HospitalNodeConfig(BaseModel):
    """Digital Twin Hospital Configuration."""
    id: str = Field(..., description="Unique hospital identifier (e.g., hospital_1)")
    name: str = Field(..., description="Hospital institution name (e.g., Mayo Clinic Health System)")
    location: str = Field(default="Rochester, MN", description="Geographic city/state")
    compute_device: Literal["cuda", "cpu", "mps"] = Field(default="cpu", description="Hardware accelerator type")
    simulated_latency_ms: float = Field(default=45.0, ge=0.0, description="Average network latency in milliseconds")
    bandwidth_mbps: float = Field(default=100.0, gt=0.0, description="Simulated network uplink bandwidth")
    straggler_prob: float = Field(default=0.05, ge=0.0, le=1.0, description="Probability of experiencing compute/communication delays")
    dropout_prob: float = Field(default=0.0, ge=0.0, le=0.5, description="Probability of temporarily dropping out during a round")

class AlgorithmConfig(BaseModel):
    """Federated Optimization Algorithm Configuration."""
    name: Literal["fedavg", "fedprox", "scaffold", "fednova", "fedadam", "fedyogi"] = Field(
        default="fedavg", description="FL aggregation algorithm"
    )
    # FedProx parameters
    mu: float = Field(default=0.01, ge=0.0, description="Proximal regularization penalty coefficient for FedProx")
    # FedOpt / FedAdam / FedYogi parameters
    server_lr: float = Field(default=0.01, gt=0.0, description="Server-side adaptive learning rate")
    server_beta1: float = Field(default=0.9, ge=0.0, lt=1.0, description="First moment momentum parameter for FedAdam")
    server_beta2: float = Field(default=0.99, ge=0.0, lt=1.0, description="Second moment parameter for FedAdam")
    server_tau: float = Field(default=1e-3, gt=0.0, description="Numerical stability constant for FedAdam")

class FLTrainingConfig(BaseModel):
    """Federated Training Hyperparameters."""
    num_rounds: int = Field(default=15, ge=1, description="Total number of global federated communication rounds")
    num_hospitals: int = Field(default=5, ge=2, description="Total number of simulated hospital clients")
    client_fraction: float = Field(default=1.0, gt=0.0, le=1.0, description="Fraction of clients randomly sampled per round")
    local_epochs: int = Field(default=2, ge=1, description="Number of local training epochs per hospital round")
    batch_size: int = Field(default=16, ge=1, description="Local training mini-batch size")
    learning_rate: float = Field(default=0.01, gt=0.0, description="Local SGD optimizer learning rate")
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0, description="SGD momentum factor")
    weight_decay: float = Field(default=1e-4, ge=0.0, description="L2 weight decay penalty")

class ServerConfig(BaseModel):
    """FastAPI & gRPC Server Configuration."""
    host: str = Field(default="127.0.0.1", description="Backend host address")
    port: int = Field(default=8000, description="FastAPI & WebSocket port")
    grpc_port: int = Field(default=50051, description="gRPC communication port")
    ws_endpoint: str = Field(default="/ws", description="WebSocket route for real-time telemetry")
    enable_grpc: bool = Field(default=False, description="Whether to launch gRPC distributed service")

class FedHealthConfig(BaseModel):
    """Root Configuration Container for FedHealth."""
    experiment_name: str = Field(default="FedHealth_Clinical_Benchmark", description="Experiment run name")
    training: FLTrainingConfig = Field(default_factory=FLTrainingConfig)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    privacy: DPConfig = Field(default_factory=DPConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: PartitionConfig = Field(default_factory=PartitionConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    hospitals: Optional[List[HospitalNodeConfig]] = Field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_yaml(cls, path: str) -> "FedHealthConfig":
        """Load configuration from a YAML file."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_yaml(self, path: str):
        """Save configuration to a YAML file."""
        import yaml
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
