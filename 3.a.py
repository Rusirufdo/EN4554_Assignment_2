import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def apply_mixup(images, labels, alpha=0.2):
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def mixup_loss(criterion, predictions, labels_a, labels_b, lam):
    return lam * criterion(predictions, labels_a) + (1 - lam) * criterion(predictions, labels_b)


class Cutout:
    def __init__(self, size=8):
        self.size = size

    def __call__(self, image):
        image = image.clone()
        _, height, width = image.shape
        center_y = random.randint(0, height - 1)
        center_x = random.randint(0, width - 1)
        y1 = max(0, center_y - self.size // 2)
        y2 = min(height, center_y + self.size // 2)
        x1 = max(0, center_x - self.size // 2)
        x2 = min(width, center_x + self.size // 2)
        image[:, y1:y2, x1:x2] = 0
        return image


class RegularizedCNN(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def compute_accuracy(model, loader, device):
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


def train_regularized_epoch(model, loader, optimizer, device, use_mixup=False):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if use_mixup:
            images, labels_a, labels_b, lam = apply_mixup(images, labels, alpha=0.2)
            outputs = model(images)
            loss = mixup_loss(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
    return total_loss / total_samples


def create_small_train_loader(transform, train_indices):
    full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataset = Subset(full_dataset, train_indices[:5000])
    return DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    train_indices = list(range(len(train_dataset)))
    train_loader_for_val = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    flip_crop_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cutout_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout(size=8),
    ])
    combined_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout(size=8),
    ])

    small_loaders = {
        'basic': create_small_train_loader(basic_transform, train_indices),
        'flip_crop': create_small_train_loader(flip_crop_transform, train_indices),
        'cutout': create_small_train_loader(cutout_transform, train_indices),
        'combined': create_small_train_loader(combined_transform, train_indices),
    }

    regularization_configs = {
        'Baseline': {'loader': 'basic', 'weight_decay': 0.0, 'dropout': 0.0, 'mixup': False},
        'Weight Decay': {'loader': 'basic', 'weight_decay': 1e-4, 'dropout': 0.0, 'mixup': False},
        'Dropout': {'loader': 'basic', 'weight_decay': 0.0, 'dropout': 0.4, 'mixup': False},
        'Flip + Crop': {'loader': 'flip_crop', 'weight_decay': 0.0, 'dropout': 0.0, 'mixup': False},
        'Cutout': {'loader': 'cutout', 'weight_decay': 0.0, 'dropout': 0.0, 'mixup': False},
        'Mixup': {'loader': 'basic', 'weight_decay': 0.0, 'dropout': 0.0, 'mixup': True},
        'All Combined': {'loader': 'combined', 'weight_decay': 1e-4, 'dropout': 0.2, 'mixup': True},
    }

    reg_results = []
    reg_histories = {}
    reg_epochs = 15

    base_state = RegularizedCNN(dropout_rate=0.0).state_dict()
    for config_name, config in regularization_configs.items():
        model = RegularizedCNN(dropout_rate=config['dropout']).to(device)
        model.load_state_dict(copy.deepcopy(base_state))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=config['weight_decay'])
        loader = small_loaders[config['loader']]
        history = {'loss': [], 'train_accuracy': [], 'val_accuracy': []}

        for epoch in range(1, reg_epochs + 1):
            train_loss = train_regularized_epoch(model, loader, optimizer, device, use_mixup=config['mixup'])
            train_acc = compute_accuracy(model, loader, device)
            val_acc = compute_accuracy(model, train_loader_for_val, device)
            history['loss'].append(train_loss)
            history['train_accuracy'].append(train_acc * 100)
            history['val_accuracy'].append(val_acc * 100)
            print(f'{config_name:>12} | Epoch {epoch:>2} | Loss={train_loss:.4f} | TrainAcc={train_acc * 100:.2f}% | ValAcc={val_acc * 100:.2f}%')

        reg_histories[config_name] = history
        reg_results.append({
            'Configuration': config_name,
            'Train Accuracy': history['train_accuracy'][-1],
            'Validation Accuracy': history['val_accuracy'][-1],
            'Gap': history['train_accuracy'][-1] - history['val_accuracy'][-1],
        })

    reg_results_df = pd.DataFrame(reg_results)
    print('\nRegularization Results:')
    print(reg_results_df.to_string(index=False))

    plt.figure(figsize=(15, 5))
    ax1, ax2 = plt.subplots(1, 2, figsize=(15, 5))
    for name, history in reg_histories.items():
        ax1.plot(range(1, reg_epochs + 1), history['train_accuracy'], label=name)
        ax2.plot(range(1, reg_epochs + 1), history['val_accuracy'], label=name)
    ax1.set_title('Training Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, alpha=0.3)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(11, 6))
    for name, history in reg_histories.items():
        plt.plot(range(1, reg_epochs + 1), history['loss'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Regularization Experiments: Training Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(11, 5))
    plt.bar(reg_results_df['Configuration'], reg_results_df['Gap'])
    plt.xlabel('Regularization Method')
    plt.ylabel('Generalization Gap (Percentage Points)')
    plt.title('Training vs Validation Accuracy Gap')
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
