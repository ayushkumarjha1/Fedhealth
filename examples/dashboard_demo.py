import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fedpro.core.server import FLServer
from fedpro.core.client import FLClient
from fedpro.data.partitioner import iid_partition
from fedpro.api.dashboard_server import start_dashboard_server
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

def main():
    # 1. Start the background Dashboard Server
    start_dashboard_server(port=8000)
    print("Dashboard API started. Ensure your React app is running (npm run dev) and open it!")
    print("Waiting 5 seconds for you to open the dashboard...")
    time.sleep(5)

    # 2. Setup FL Data and Models
    print("Preparing FL Framework...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    indices = list(range(2000))
    subset = torch.utils.data.Subset(train_dataset, indices)
    
    num_clients = 3
    client_datasets = iid_partition(subset, num_clients)
    client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    server = FLServer(SimpleCNN())
    clients = [FLClient(client_id=f"client_{i}", model=SimpleCNN(), device="cpu") for i in range(num_clients)]
    test_client = FLClient(client_id="evaluator", model=SimpleCNN(), device="cpu")
    
    config = {"epochs": 1, "lr": 0.01}
    
    # 3. Training Loop with deliberate delays to see charts update
    num_rounds = 10
    print("\nStarting Federated Training...")
    for round_num in range(1, num_rounds + 1):
        server.fit_round(round_num, clients, client_loaders, config)
        server.evaluate_round(round_num, test_client, test_loader)
        time.sleep(1) # Pause so user can watch the chart draw
        
    print("Training Complete!")
    time.sleep(60) # Keep server alive so user can look at the final dashboard

if __name__ == "__main__":
    main()
