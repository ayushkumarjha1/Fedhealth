import grpc
import torch
import io
import logging
from typing import Dict

from fedpro.proto import federated_pb2
from fedpro.proto import federated_pb2_grpc
from fedpro.core.client import FLClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GRPCClient")

def serialize_weights(weights):
    buffer = io.BytesIO()
    torch.save(weights, buffer)
    return buffer.getvalue()

def deserialize_weights(weights_bytes):
    buffer = io.BytesIO(weights_bytes)
    return torch.load(buffer, map_location="cpu")

class FLGRPCClient:
    def __init__(self, client_id: str, fl_client: FLClient, server_address: str = "localhost:50051"):
        self.client_id = client_id
        self.fl_client = fl_client
        self.channel = grpc.insecure_channel(server_address)
        self.stub = federated_pb2_grpc.FederatedLearningStub(self.channel)
        
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        logger.info(f"Requesting global model from server...")
        req = federated_pb2.ModelRequest(client_id=self.client_id)
        response = self.stub.GetGlobalModel(req)
        weights = deserialize_weights(response.weights)
        return weights, response.round
        
    def send_local_model(self, round_num: int, num_samples: int, weights: Dict[str, torch.Tensor], loss: float):
        logger.info(f"Sending local update to server for round {round_num}...")
        weights_bytes = serialize_weights(weights)
        req = federated_pb2.LocalModelUpdate(
            client_id=self.client_id,
            round=round_num,
            num_samples=num_samples,
            weights=weights_bytes,
            loss=loss
        )
        response = self.stub.SendLocalModel(req)
        return response.status
        
    def start_training(self, train_loader, config):
        # 1. Get Global Model
        global_weights, round_num = self.get_global_model()
        
        # 2. Train Locally
        logger.info(f"Starting local training for round {round_num}...")
        res = self.fl_client.fit(global_weights, train_loader, config)
        
        # 3. Send Updates Back
        status = self.send_local_model(round_num, res["num_samples"], res["parameters"], res["metrics"]["loss"])
        logger.info(f"Server acknowledged update: {status}")
