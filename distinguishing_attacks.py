import torch
import torch.nn as nn
import torch.nn.functional as F

from cifar10_models.vgg import vgg19_bn

if __name__ == "__main__":
    import os
    import sys

    model = vgg19_bn(device = torch.device(0), pretrained=True)
