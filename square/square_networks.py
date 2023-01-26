import torch
from torch import nn
from spiral.networks import DSS

class MLPDSSX(nn.Module):
    def __init__(self, n_out=2, n_x_div=5):
        super().__init__()
        self.DSS = nn.Sequential(
            DSS(1, 512),
            nn.ReLU(inplace=True),
            DSS(512, 512),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(512, n_out)
        self.n_x_div = n_x_div

    def forward(self, x):
        new_x = torch.zeros(x.shape[0], self.n_x_div, 1).to(x.device)
        interval = 1 / self.n_x_div
        for i in range(self.n_x_div):
            mask = (x[:, 0] >= interval * i) * (x[:, 0] <= interval * (i+1))
            new_x[mask, i] = x[mask, 1:]
        dss_out = self.DSS(new_x)
        return self.fc(torch.sum(dss_out, 1))

class MLPNoX(nn.Module):
    def __init__(self, n_out=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_out)
        )

    def forward(self, x):
        return self.fc(x[:, 1:])