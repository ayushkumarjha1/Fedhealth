import grpc
from concurrent import futures
import torch
import io
import logging

from fedpro.proto import federated_pb2
from fedpro.proto import federated_pb2_grpc
from fedpro.core.server import FLServer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GRPCServer")

def serialize_weights(weights):
    buffer = io.BytesIO()
    torch.save(weights, buffer)
    return buffer.getvalue()

def deserialize_weights(weights_bytes):
    buffer = io.BytesIO(weights_bytes)
    return torch.load(buffer, map_location="cpu")

class FederatedServicer(federated_pb2_grpc.FederatedLearningServicer):
    def __init__(self, fl_server: FLServer):
        self.fl_server = fl_server
        self.current_round = 1
        
    def GetGlobalModel(self, request, context):
        logger.info(f"Client {request.client_id} requested global model")
        params = self.fl_server.get_parameters()
        weights_bytes = serialize_weights(params)
        return federated_pb2.ModelResponse(
            round=self.current_round,
            weights=weights_bytes,
            status="SUCCESS"
        )
        
    def SendLocalModel(self, request, context):
        logger.info(f"Received update from {request.client_id} for round {request.round} with loss {request.loss:.4f}")
        weights = deserialize_weights(request.weights)
        # Note: In a real distributed system, the server would queue updates until enough clients respond.
        return federated_pb2.UpdateResponse(status="SUCCESS")

def serve(fl_server: FLServer, port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federated_pb2_grpc.add_FederatedLearningServicer_to_server(FederatedServicer(fl_server), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"gRPC Server started on port {port}")
    server.wait_for_termination()
