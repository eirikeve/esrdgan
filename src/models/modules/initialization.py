"""
initialization.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements a weight init for nn.Module subclasses

From /codes/models/networks.py in https://github.com/xinntao/BasicSR
"""

import functools

import torch
from torch.nn import init


def init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)

def init_weights(m, scale=1):
    init_kaiming_ = functools.partial(init_kaiming, scale=scale)
    m.apply(init_kaiming_)