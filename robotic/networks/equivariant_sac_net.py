import torch
import torch.nn.functional as F
from torch.distributions import Normal

from e2cnn import gspaces
from e2cnn import nn


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

from networks.cnn import SpatialSoftArgmax




class EquivariantPolicyFlip(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.n_hidden = n_hidden
        self.group = gspaces.Flip2dOnR2()
        self.conv = self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]),
                      nn.FieldType(self.group, n_hidden // 8 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_hidden // 8 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_hidden // 8 * [self.group.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.group, n_hidden // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_hidden // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_hidden // 4 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_hidden // 4 * [self.group.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.group, n_hidden // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_hidden // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_hidden // 2 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_hidden // 2 * [self.group.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.group, n_hidden // 2 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_hidden * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_hidden * 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_hidden * 2 * [self.group.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.group, n_hidden * 2 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_hidden * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_hidden * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]), inplace=True),
            # 1x1
            nn.R2Conv(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]),
                      nn.FieldType(self.group, 2 * [self.group.irrep(1)] + (self.action_dim - 2) * [self.group.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, x):
        x_geo = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        conv_out = self.conv(x_geo).tensor.reshape(x.shape[0], -1)
        dy = conv_out[:, 0:1]
        dtheta = conv_out[:, 1:2]
        inv_act = conv_out[:, 2:self.action_dim]
        action = torch.cat((inv_act[:, 0:2], dy, inv_act[:, 2:3], dtheta), dim=1)
        return action
        
if __name__ == '__main__':

    
    model = EquivariantPolicyFlip((5, 128, 128), 5, 128)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)