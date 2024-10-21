import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba2 as Mamba

class StackedMamba(nn.Module):
    def __init__(self, d_model, num_layers):
        super(StackedMamba, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.layers = nn.ModuleList([Mamba(d_model=d_model) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
