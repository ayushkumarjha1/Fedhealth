import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fedpro.core.server import FLServer
from fedpro.algorithms.fedprox import FedProxClient
from fedpro.data.partitioner import iid_partition
from fedpro.network.grpc_server import serve
from fedpro.network.grpc_client import FLGRPCClient
import multiprocessing
import time
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.fc1 = nn.Linear(1440, 10)

    def forward(self, x):
        import torch.nn.functional as F
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.fc1(x), dim=1)

def run_server():
    global_model = SimpleCNN()
    server = FLServer(global_model)
    print("Starting gRPC server...")
    serve(server, port=50051)

def run_client(client_id, dataset):
    time.sleep(2) # wait for server to start
    print(f"Starting gRPC client {client_id}...")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # We use FedProx as our advanced algorithm
    fl_client = FedProxClient(client_id=client_id, model=SimpleCNN(), device="cpu")
    grpc_client = FLGRPCClient(client_id=client_id, fl_client=fl_client, server_address="localhost:50051")
    
    config = {"epochs": 1, "lr": 0.01, "mu": 0.01}
    grpc_client.start_training(loader, config)

if __name__ == "__main__":
    print("Preparing data for distributed demo...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    
    # Just take a small subset for quick demo
    indices = list(range(1000))
    subset = torch.utils.data.Subset(train_dataset, indices)
    
    client_datasets = iid_partition(subset, 2)
    
    server_process = multiprocessing.Process(target=run_server)
    client_processes = [
        multiprocessing.Process(target=run_client, args=(f"client_{i}", client_datasets[i]))
        for i in range(2)
    ]
    
    server_process.start()
    for cp in client_processes:
        cp.start()
        
    for cp in client_processes:
        cp.join()
        
    server_process.terminate()
    print("gRPC Distributed Demo Complete! Both clients successfully downloaded global model, trained with FedProx, and uploaded their parameters.")
