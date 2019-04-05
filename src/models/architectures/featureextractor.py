"""
featureextractor.py
Written by Eirik Vesterkjær, 2019
Apache License

Implements VGG-19 feature extractor

this is based on architecture.py from https://github.com/xinntao/BasicSR/codes/models/modules
ref.: https://discuss.pytorch.org/t/whats-the-range-of-the-input-value-desired-to-use-pretrained-resnet152-and-vgg19/1683/2
Inputs are in the range [0,1] and normalized, with mean=[0.485, 0.456, 0.406]  std=[0.229, 0.224, 0.225]
"""
import torch
import torch.nn as nn
import torchvision

class VGG19FeatureExtractor(nn.Module):
    def __init__(self, 
                low_level_feature_layer: int = 1,
                high_level_feature_layer: int = 34,
                use_batch_norm: bool = False,
                use_input_norm: bool = True,
                device = torch.device('cpu')):
        super(VGG19FeatureExtractor, self).__init__()
        if use_batch_norm:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        
        self.features_low = nn.Sequential(*list(model.features.children())[:(low_level_feature_layer + 1)])
        self.features_high = nn.Sequential(*list(model.features.children())[(low_level_feature_layer + 1) : (high_level_feature_layer+1)])
        for _, v in self.features_low.named_parameters():
            v.requires_grad = False
        for _, v in self.features_high.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        x1 = self.features_low(x)
        x2 = self.features_high(x1)
        x1 = x1.reshape(x1.shape[0], -1)
        x2 = x2.reshape(x2.shape[0], -1)
        return torch.cat((x1, x2), 1)

        