"""
featureextractor.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements VGG-19 feature extractor

this is based on architecture.py from https://github.com/xinntao/BasicSR/codes/models/modules
"""

import torch.nn as nn
import torchvision

class VGG19FeatureExtractor(nn.Module):
    def __init__(self, 
                feature_layer: int = 1,
                use_batch_norm: bool = False,
                device = torch.device('cpu')):
        super(VGG19FeatureExtractor, self).__init__()
        if use_batch_norm:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        
        #TODO: Input normalization here or somewhere else?
        
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        for _, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        return self.features(x)

        