"""FedHealth Medical Model Zoo."""
from fedpro.models.base import BaseMedicalModel
from fedpro.models.mlp import HealthcareMLP
from fedpro.models.cnn import MedicalCNN
from fedpro.models.resnet import MedicalResNet

MODEL_REGISTRY = {
    "healthcaremlp": HealthcareMLP,
    "medicalcnn": MedicalCNN,
    "medicalresnet": MedicalResNet,
    "resnet18": lambda **kw: MedicalResNet(version="resnet18", **kw),
    "resnet34": lambda **kw: MedicalResNet(version="resnet34", **kw),
}

def get_model(name: str, **kwargs) -> BaseMedicalModel:
    """Instantiate a medical model from the registry."""
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key](**kwargs)

__all__ = [
    "BaseMedicalModel",
    "HealthcareMLP",
    "MedicalCNN",
    "MedicalResNet",
    "get_model",
    "MODEL_REGISTRY",
]
