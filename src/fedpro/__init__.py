"""
FedHealth: A Research-Grade Privacy-Preserving Federated Learning Framework for Healthcare Analysis.
"""

__version__ = "1.0.0"
__author__ = "FedHealth Engineering & Research Team"

from fedpro.core.base import BaseFLClient, BaseFLServer, ClientUpdate, ServerEvaluation
from fedpro.core.client import FLClient
from fedpro.core.server import FLServer
from fedpro.core.hospital import DigitalTwinHospital
from fedpro.configs.base_config import FedHealthConfig
from fedpro.privacy.rdp_accountant import RDPAccountant
from fedpro.privacy.privacy_engine import PrivacyEngine
from fedpro.copilot.copilot_engine import AIFederatedCopilot
from fedpro.replay.replay_engine import FederatedTrainingReplay
from fedpro.xai.xai_engine import ClinicalExplainer

__all__ = [
    "__version__",
    "BaseFLClient",
    "BaseFLServer",
    "ClientUpdate",
    "ServerEvaluation",
    "FLClient",
    "FLServer",
    "DigitalTwinHospital",
    "FedHealthConfig",
    "RDPAccountant",
    "PrivacyEngine",
    "AIFederatedCopilot",
    "FederatedTrainingReplay",
    "ClinicalExplainer",
]
