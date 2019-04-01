"""
blocks.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements some useful block structures for nn architectures
"""

import torch
import torch.nn as nn

class SkipConnectionBlock(nn.Module):
    def __init__(self, submodule):
        super(SkipConnectionBlock, self).__init__()
        self.module = submodule
    def forward(self, x):
        return x + self.module(x)

# wait, i don't need this.. oh well :/
class ResidualBlock(nn.Module):
    def __init__(self, in_nc: int, mid_nc: int, out_nc: int, res_scale = 1.0):
        first = nn.Conv2d(in_nc, mid_nc, (3,3), stride=1, padding=1)
        first_act = nn.ReLU()
        second = nn.Conv2d(mid_nc, out_nc, (3,3), stride=1, padding=1)
        final_act = nn.ReLU()
        self.model = nn.Sequential( \
                            SkipConnectionBlock( nn.Sequential(first, first_act, second) ),\
                            final_act )
    
    def forward(self, x):
        return self.model(x)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nc, gc):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel
    