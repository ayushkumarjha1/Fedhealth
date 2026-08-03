import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fedpro.core.client import FLClient
from fedpro.core.server import FLServer
from fedpro.data.partitioner import iid_partition
import os

# Simple CNN model for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        import torch.nn.functional as F
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def main():
    # 1. Load Data
    print("Loading MNIST Dataset...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # 2. Partition Data for 5 clients
    num_clients = 5
    print(f"Partitioning data into {num_clients} clients...")
    client_datasets = iid_partition(train_dataset, num_clients)
    client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
    
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 3. Initialize Server and Clients
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    global_model = SimpleCNN()
    
    server = FLServer(global_model)
    clients = [FLClient(client_id=f"client_{i}", model=SimpleCNN(), device=device) for i in range(num_clients)]
    
    # We need a dummy client for global evaluation
    test_client = FLClient(client_id="evaluator", model=SimpleCNN(), device=device)
    
    # 4. Federated Training Loop
    num_rounds = 3
    config = {"epochs": 1, "lr": 0.01}
    
    print("\nStarting Federated Training...")
    for round_num in range(1, num_rounds + 1):
        server.fit_round(round_num, clients, client_loaders, config)
        server.evaluate_round(round_num, test_client, test_loader)
        
    print("Training Complete!")

if __name__ == "__main__":
    main()
