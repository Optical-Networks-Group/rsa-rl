
import numpy as np
import torch
import torch.nn as nn
from pfrl.policies import SoftmaxCategoricalHead
from rsarl.models.initializer import normalized_columns_initializer


class DeepRMSAv2Net(nn.Module):
    """Deep Neural Network used in DeepRMSAv2(called ACNet)

     paper: DeepRMSA: A Deep Reinforcement Learning Framework for Routing, 
            Modulation and Spectrum Assignment in Elastic Optical Networks
            (https://ieeexplore.ieee.org/document/8738827)

    """
    
    def __init__(self, n_input, n_action):
        super().__init__()
        self.n_input = n_input
        self.n_action = n_action
        # network definition
        self.body_p = self.body()
        self.body_v = self.body()
        out_size = self._get_conv_out()
        # last layers
        self.policy = nn.Linear(out_size, self.n_action, bias=False)
        self.value = nn.Linear(out_size, 1, bias=False)
        # softmax
        self.softmax = SoftmaxCategoricalHead()
        # initialize last layers
        self.policy.weight.data = normalized_columns_initializer(
            self.policy.weight.data, 0.01)
        self.value.weight.data = normalized_columns_initializer(
            self.value.weight.data, 1.0)

    def body(self):
        net = []
        # first layer
        net.append(nn.Linear(self.n_input, 128))
        net.append(nn.ELU())
        # later
        for _ in range(5-1):
            net.append(nn.Linear(128, 128))
            net.append(nn.ELU())
        return nn.Sequential(*net)

    def _get_conv_out(self):
        o = self.body_v(torch.zeros(1, self.n_input))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        p = self.body_p(x)
        p = self.policy(p)
        p = self.softmax(p)

        v = self.body_v(x)
        v = self.value(v)
        return p, v
