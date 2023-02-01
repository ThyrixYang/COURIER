import torch
import torch.nn as nn
import torch.nn.functional as F

class CTRNet(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, 1024)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.output_layer = nn.Linear(1024, out_dim)

    def forward(self, x):
        x = self.bn1(F.relu(self.linear1(x)))
        x = self.output_layer(x)
        return x
