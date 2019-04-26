"""
basegan.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements the skeleton of a GAN model

Partially based on codes/models/base_gan in https://github.com/xinntao/BasicSR
"""
import os

import torch
import torch.nn as nn

import config.config as config
import models.modules.loggingclass as lc

class BaseGAN(lc.GlobalLoggingClass):
    # Generator, discriminator
    G: nn.Module = None
    D: nn.Module = None
    def __init__(self, cfg: config.Config):
        super(BaseGAN, self).__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if cfg.gpu_id is not None else "cpu")
        self.is_train = cfg.is_train
        self.schedulers = []
        self.optimizers = []
        return

    def load_model(self,  generator_load_path: str = None, discriminator_load_path: str = None, state_load_path: str = None):
        if not generator_load_path is None and \
           not generator_load_path.lower() == "null" and \
           not generator_load_path.lower() == "none":
            self.G.load_state_dict(torch.load(generator_load_path, map_location="cpu"))
            self.G.eval()
        if not discriminator_load_path is None and \
           not discriminator_load_path.lower() == "null" and \
           not discriminator_load_path.lower() == "none":
            self.D.load_state_dict(torch.load(discriminator_load_path, map_location="cpu"))
            self.G.eval()
        if not state_load_path is None and \
           not state_load_path.lower() == "null" and \
           not state_load_path.lower() == "none":
            state = torch.load(state_load_path)
            loaded_optimizers = state["optimizers"]
            loaded_schedulers = state["schedulers"]
            assert len(loaded_optimizers) == len(self.optimizers), f"Loaded {len(loaded_optimizers)} optimizers but expected { len(self.optimizers)}"
            assert len(loaded_schedulers) == len(self.schedulers), f"Loaded {len(loaded_schedulers)} schedulers but expected {len(self.schedulers)}"
            for i, o in enumerate(loaded_optimizers):
                self.optimizers[i].load_state_dict(o)
            for i, s in enumerate(loaded_schedulers):
                self.schedulers[i].load_state_dict(s)
            return state["epoch"], state["it"]
        return None, None
        

    def save_model(self, save_basepath: str, epoch: int, it: int, save_G: bool = True, save_D: bool = True, save_state: bool = True):
        save_basepath = self.cfg.env.this_runs_folder
        generator_save_path = os.path.join(save_basepath, f"G_{it}.pth")
        discriminator_save_path = os.path.join(save_basepath, f"D_{it}.pth")
        state_save_path = os.path.join(save_basepath, f"state_{it}.pth")
        

       

        if save_G:
            torch.save(self.G.state_dict(), generator_save_path)
        if save_D:
            torch.save(self.D.state_dict(), discriminator_save_path)
        if save_state:
            state = {"it": it, "epoch": epoch, "schedulers": [], "optimizers": []}
            for s in self.schedulers:
                state["schedulers"].append(s.state_dict())
            for o in self.optimizers:
                state["optimizers"].append(o.state_dict())
            torch.save(state, state_save_path)
