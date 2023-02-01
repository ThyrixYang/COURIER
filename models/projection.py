import torch
import torch.nn as nn


class ProjectoinHead(nn.Module):

    def __init__(self,
                 in_dim,
                 mid_dim,
                 out_dim,
                 activation=torch.tanh):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, mid_dim)
        self.bn1 = nn.BatchNorm1d(mid_dim)
        self.linear2 = nn.Linear(mid_dim, mid_dim)
        self.bn2 = nn.BatchNorm1d(mid_dim)
        self.linear3 = nn.Linear(mid_dim, mid_dim)
        self.bn3 = nn.BatchNorm1d(mid_dim)
        self.linear4 = nn.Linear(mid_dim, out_dim)
        self.activation = activation

    def forward(self, x):
        x1 = self.linear1(x)
        x = self.activation(self.bn1(x1))
        x = self.linear2(x)
        x = self.activation(self.bn2(x))
        x = self.linear3(x)
        x = self.activation(self.bn3((x+x1)/2))
        x = self.linear4(x)
        return x
