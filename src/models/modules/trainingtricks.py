"""
trainingtricks.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements some useful methods with the goal of improving GAN training.

"""


import random

import torch


def noisy_labels(label_type: bool, batch_size: int, noise_stddev: float = 0.05,
                          false_label_val: float = 0.0, true_label_val: float = 1.0, 
                          val_lower_lim: float = 0.0, val_upper_lim: float = 1.0) -> float:
    """
    noisy_labels adds gaussian noise to True/False GAN label values,
    but keeps the resulting value within a specified range,
    and returns a tensor of sz batch_size filled with that value.
    @arg label_type: True if representing images perceived as real (not generated), else False
    @arg noise_stddev: gaussian noise stddev
    @arg [false|true]_label_val: label values without noise.
    @arg val_[lower|upper]_lim: thresholds for label val cutoff
    """
    label_val: float = random.gauss(mu=0.0, sigma=noise_stddev)
    if label_type == True:
        label_val += true_label_val
    else:
        label_type += false_label_val
    if label_val >  val_upper_lim:
        label_val = val_upper_lim
    elif label_val < val_lower_lim:
        label_val = val_lower_lim
    return torch.FloatTensor(batch_size).fill_(label_val)

