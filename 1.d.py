import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


class CustomSGDMomentum(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                if not hasattr(p, 'momentum_buffer'):
                    p.momentum_buffer = torch.zeros_like(p.data)
                p.momentum_buffer.mul_(momentum).add_(p.grad)
                p.data.add_(p.momentum_buffer, alpha=-lr)
        return loss


class CustomAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'] += 1
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                grad = p.grad.data

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = exp_avg_sq.sqrt() / np.sqrt(bias_correction2)
                step_size = lr / bias_correction1
                p.data.addcdiv_(exp_avg, denom.add(eps), value=-step_size)

        return loss


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train_cnn_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
    return total_loss / total_samples


@torch.no_grad()
def evaluate_cnn(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def get_loaders(batch_size=128, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def run_experiment(optimizer_name, lr, device):
    train_loader, val_loader = get_loaders(batch_size=128, num_workers=0)
    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD Momentum':
        optimizer = CustomSGDMomentum(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = CustomAdam(model.parameters(), lr=lr)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

    history = {
        'loss': [],
        'val_accuracy': [],
        'best_val_accuracy': 0.0,
        'best_epoch': 0,
    }

    for epoch in range(1, 11):
        train_loss = train_cnn_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate_cnn(model, val_loader, device)
        history['loss'].append(train_loss)
        history['val_accuracy'].append(val_acc)

        if val_acc > history['best_val_accuracy']:
            history['best_val_accuracy'] = val_acc
            history['best_epoch'] = epoch

        print(f'{optimizer_name:>12} | LR={lr:>7} | Epoch {epoch:>2} | Train Loss={train_loss:.4f} | Val Acc={val_acc:.4f}')

    return model, history


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    learning_rates = {
        'SGD': [0.001, 0.01, 0.1],
        'SGD Momentum': [0.001, 0.01, 0.1],
        'Adam': [0.0001, 0.001, 0.01],
    }

    experiment_results = []
    training_histories = {}

    start_all = time.time()
    for optimizer_name, rates in learning_rates.items():
        for lr in rates:
            print(f'\nTraining {optimizer_name} with lr={lr}')
            _, history = run_experiment(optimizer_name, lr, device)
            experiment_results.append({
                'Optimizer': optimizer_name,
                'Learning Rate': lr,
                'Final Train Loss': history['loss'][-1],
                'Final Val Accuracy': history['val_accuracy'][-1] * 100,
                'Best Val Accuracy': history['best_val_accuracy'] * 100,
                'Best Epoch': history['best_epoch'],
                'Time (s)': 0.0,
            })
            training_histories[(optimizer_name, lr)] = history

    elapsed = time.time() - start_all
    print(f'\nCompleted in {elapsed:.1f} seconds')

    results_df = pd.DataFrame(experiment_results)
    print('\nResults table:')
    print(results_df.sort_values(['Optimizer', 'Learning Rate']).to_string(index=False))

    plt.figure(figsize=(9, 6))
    for optimizer_name in learning_rates.keys():
        subset = results_df[results_df['Optimizer'] == optimizer_name].sort_values('Learning Rate')
        plt.plot(subset['Learning Rate'], subset['Best Val Accuracy'], marker='o', linewidth=2, label=optimizer_name)
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Validation Accuracy vs Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    best_configs = {}
    for optimizer_name in learning_rates.keys():
        subset = results_df[results_df['Optimizer'] == optimizer_name]
        best_row = subset.loc[subset['Best Val Accuracy'].idxmax()]
        best_configs[optimizer_name] = float(best_row['Learning Rate'])
        print(f'{optimizer_name}: Best LR = {best_row["Learning Rate"]} | Best Val Acc = {best_row["Best Val Accuracy"]:.2f}%')

    plt.figure(figsize=(10, 6))
    for optimizer_name, best_lr in best_configs.items():
        history = training_histories[(optimizer_name, best_lr)]
        plt.plot(range(1, len(history['loss']) + 1), history['loss'], marker='o', linewidth=2, label=f'{optimizer_name} (LR={best_lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Training Cross-Entropy Loss')
    plt.title('Training Loss at Best Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
