

import torch
import torch.nn as nn


class MomentCoxianTwoLayerNet(nn.Module):
    def __init__(self, cox_dim, h=100):
        super(MomentCoxianTwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(2 * cox_dim - 1, h)
        self.fc2 = nn.Linear(h, 2 * cox_dim - 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

