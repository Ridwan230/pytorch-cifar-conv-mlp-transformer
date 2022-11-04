import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.count = 0

    def forward(self, x):
        weight = self.weight
        # MWE
        max_val = (1.0 + 0.2) * \
            torch.max(torch.max(weight), -1 * torch.min(weight))
        if self.count:
            weight = max_val * 0.5 * \
                torch.log((1+weight/max_val) / (1-weight/max_val))
        else:
            self.count = 1
        return F.linear(x, weight, bias=True)
