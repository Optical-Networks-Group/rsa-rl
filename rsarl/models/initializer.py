
import torch
import torch.nn as nn

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1,keepdim=True).expand_as(out))
    return out

def constant_bias_initializer(bias=0.0):
    @torch.no_grad()
    def init_bias(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.bias.fill_(bias)

    return init_bias

