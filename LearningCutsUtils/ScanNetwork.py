import torch
import torch.nn
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F

from LearningCutsUtils.OneToOneLinear import OneToOneLinear


class ScanNetwork(torch.nn.Module):
    def __init__(self,features,pt,mu,effics,weights=None,activationscale=2.):
        super().__init__()
        self.features = features
        self.pt = pt
        self.mu = mu
        self.effics = effics
        self.weights = weights
        self.activation_scale_factor=activationscale
        self.nets = torch.nn.ModuleList([
            torch.nn.ModuleList([
                torch.nn.ModuleList([
                                    OneToOneLinear(features, self.activation_scale_factor, self.weights)
                    for k in range(len(self.effics))
                ])
                for j in range(len(self.mu))
            ])
            for i in range(len(self.pt))
        ])

    def forward(self, x):
        ### returns a list of network outputs with len(effics)
        outputs=[[[self.nets[i][j][k](x[i][j]) for k in range(len(self.effics))] for j in range(len(self.mu))] for i in range(len(self.pt))]
        return outputs

    def to(self, device):
        super().to(device)
        for n in self.nets:
            n.to(device)

