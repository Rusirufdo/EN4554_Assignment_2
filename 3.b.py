import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + self.shortcut(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return nn.functional.relu(out + self.skip(residual))


class PlainCIFAR(nn.Module):
    def __init__(self, depth=20):
        super().__init__()
        n = (depth - 2) // 6
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(16, 16, n, PlainBlock, stride=1)
        self.layer2 = self._make_layer(16, 32, n, PlainBlock, stride=2)
        self.layer3 = self._make_layer(32, 64, n, PlainBlock, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, in_channels, out_channels, blocks, block_type, stride=1):
        layers = []
        layers.append(block_type(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(block_type(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResidualCIFAR(nn.Module):
    def __init__(self, depth=20):
        super().__init__()
        n = (depth - 2) // 6
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(16, 16, n, ResidualBlock, stride=1)
        self.layer2 = self._make_layer(16, 32, n, ResidualBlock, stride=2)
        self.layer3 = self._make_layer(32, 64, n, ResidualBlock, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, in_channels, out_channels, blocks, block_type, stride=1):
        layers = []
        layers.append(block_type(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(block_type(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_cifar_loaders(batch_size=128, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def compare_networks():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar_loaders(batch_size=128, num_workers=0)

    results = {}
    for network_name, model_fn, depths in [
        ('Plain', PlainCIFAR, [8, 20, 56]),
        ('Residual', ResidualCIFAR, [8, 20, 56]),
    ]:
        for depth in depths:
            model = model_fn(depth=depth).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            loss_history = []

            for epoch in range(1, 6):
                loss = train_epoch(model, train_loader, optimizer, criterion, device)
                acc = evaluate(model, test_loader, device)
                loss_history.append(loss)
                print(f'{network_name} | depth={depth} | epoch={epoch} | train_loss={loss:.4f} | test_acc={acc:.4f}')

            results[(network_name, depth)] = loss_history

    plt.figure(figsize=(10, 6))
    for (network_name, depth), loss_history in results.items():
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label=f'{network_name} depth {depth}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Plain vs Residual Network Training Loss Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results


if __name__ == '__main__':
    compare_networks()
