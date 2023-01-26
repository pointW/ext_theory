import numpy as np
import torch
from e2cnn import gspaces
import e2cnn.nn as enn
import torch.nn as nn
from f import Inv_rot
torch_Inv_rot = torch.tensor(Inv_rot).float()


class Inv_h(nn.Module):
    def __init__(self):
        super(Inv_h, self).__init__()
        hidden_dim = 64
        self.fc = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, _, value):
        return self.fc(value)


class Equ_h(Inv_h):
    def __init__(self):
        super(Equ_h, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

    def device(self):
        return self.dummy_param.device

    def forward(self, axis, value):
        inv_f = super(Equ_h, self).forward(axis, value)
        rot = torch_Inv_rot[(-axis % 4).long().squeeze()].to(self.device())
        equ_f = torch.matmul(rot, inv_f[:, :, None]).squeeze()
        return equ_f


class C4_h(nn.Module):
    def __init__(self):
        super(C4_h, self).__init__()
        hidden_dim = 64
        c4_act = gspaces.Rot2dOnR2(N=4)
        c4_reg = c4_act.regular_repr
        c4_std = c4_act.irrep(1)
        self.feat_type_in = enn.FieldType(c4_act, 1 * [c4_reg])
        feat_type_hid = enn.FieldType(c4_act, hidden_dim * [c4_reg])
        feat_type_out = enn.FieldType(c4_act, 1 * [c4_std])
        self.fc = nn.Sequential(
            enn.R2Conv(self.feat_type_in, feat_type_hid, kernel_size=1),
            enn.ReLU(feat_type_hid),
            enn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=1),
            enn.ReLU(feat_type_hid),
            enn.R2Conv(feat_type_hid, feat_type_out, kernel_size=1)
        )
        self.dummy_param = nn.Parameter(torch.empty(0))

    def device(self):
        return self.dummy_param.device

    def forward(self, axis, value):
        b = value.shape[0]
        v_geo = torch.zeros([b, 4]).to(self.device())
        v_geo[torch.arange(b).to(self.device()), axis.squeeze().long()] = value.squeeze()
        v_geo = v_geo.reshape(-1, 4, 1, 1)
        v_geo = enn.GeometricTensor(v_geo, self.feat_type_in)
        return self.fc(v_geo).tensor.squeeze()
