import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class WeightedMSELoss(nn.Module):
    """Enables nn to have weighing on loss"""

    def forward(self, pred, target, weights):
        loss = (pred - target) ** 2
        return (loss * weights).sum() / weights.sum()

class WeightedDataset(Dataset):
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w

        assert len(self.x) == len(self.y) == len(self.w), "datast has different length"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]

if __name__ == '__main__':
    data = 500

    x_data = torch.tensor(np.random.normal(size=(data, 1)))

    y_data = torch.tensor(
        [random.choice([1, 2, 3]) for _ in range(data)]
        )[:, None]

    weights = torch.tensor(np.linspace(0, 100, num=data))[:, None]

    data = WeightedDataset(x_data, y_data, weights)
    result = DataLoader(data)

    loss = WeightedMSELoss()

    print(loss.forward(
        x_data, y_data, weights
    ))
