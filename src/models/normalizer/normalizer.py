"""
normalizer.py

Based on normalization as described here: https://discuss.pytorch.org/t/normalization-of-input-data-to-qnetwork/14800/2
"""
import torch

class Normalizer():
    def __init__(self, batch_shape):
        self.n = torch.zeros(batch_shape)
        self.mean = torch.zeros(batch_shape)
        self.mean_diff = torch.zeros(batch_shape)
        self.var = torch.zeros(batch_shape)

    def observe(self, batch):
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (batch-self.mean)/self.n
        self.mean_diff += (batch-last_mean)*(batch-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, batch):
        obs_std = torch.sqrt(self.var)
        return (batch - self.mean)/obs_std