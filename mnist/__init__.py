import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os

__all__ = ["MNISTData", "MNISTLeNet"]

class MNISTData:
    def __init__(self, batch_size=256, num_workers=4):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.mean = (0.1307,)
        self.std = (0.3081,)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def train_dataloader(self):
        dataset = torchvision.datasets.MNIST(root=self.data_dir, train=True, transform=self.transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader
    
    def val_dataloader(self):
        dataset = torchvision.datasets.MNIST(root=self.data_dir, train=False, transform=self.transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return dataloader
    
    def test_dataloader(self):
        return self.val_dataloader()

# Define LeNet model
class MNISTLeNet(nn.Module):
    def __init__(self, load_weights=False):
        super(MNISTLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5,padding='same')
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        if load_weights:
            self.load_state_dict(torch.load(self.weight_path))
    
    @property
    def weight_path(self):
        return os.path.join(os.path.dirname(__file__), "lenet_weights.pth")
    
    def save_weights(self):
        torch.save(self.state_dict(), self.weight_path)
    
    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    
if __name__ == "__main__":
    model = MNISTLeNet(load_weights=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    mnistdata = MNISTData(batch_size=128, num_workers=4)
    train_loader = mnistdata.train_dataloader()
    test_loader = mnistdata.test_dataloader()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        print(f"Epoch {epoch+1}")
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"Test accuracy: {100. * correct / len(test_loader.dataset):.2f}%")
    model.save_weights()