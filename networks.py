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

class MLPInv(nn.Module):
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

class DSS(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_out)
        self.fc2 = nn.Linear(n_in, n_out)

    def forward(self, x):
        b = x.shape[0]
        n = x.shape[1]
        x_sum = torch.sum(x, 1)
        x = x.reshape(b*n, -1)
        out1 = self.fc1(x)
        out1 = out1.reshape(b, n, -1)
        out2 = self.fc2(x_sum)
        out2 = out2.reshape(b, 1, -1)
        return out1 + out2

class MLPDSS(nn.Module):
    def __init__(self, n_out=2):
        super().__init__()
        self.DSS = nn.Sequential(
            DSS(2, 512),
            nn.ReLU(inplace=True),
            DSS(512, 512),
            nn.ReLU(inplace=True),
            # nn.Linear(2, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, n_out)
        )

        self.fc = nn.Linear(512, n_out)

    def forward(self, x):
        new_x = torch.zeros(x.shape[0], 2, 2).to(x.device)
        new_x[x[:, 2] == 0, 0] = x[x[:, 2] == 0, :2]
        new_x[x[:, 2] == 1, 1] = x[x[:, 2] == 1, :2]
        dss_out = self.DSS(new_x)
        return self.fc(torch.sum(dss_out, 1))

if __name__ == '__main__':
    model = MLPDSS(n_out=2)
    data = torch.zeros(64, 3)
    model(data)