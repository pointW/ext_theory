import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, n_in=3, n_out=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_in, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_out)
            # nn.Linear(n_in, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, n_out)
        )

    def forward(self, x):
        return self.fc(x)

class MLPNoZ(nn.Module):
    def __init__(self, n_out=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_out)
            # nn.Linear(2, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, n_out)
        )

    def forward(self, x):
        return self.fc(x[:, :2])