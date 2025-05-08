import torch
import torch.nn as nn
import torch.nn.functional as F

from cifar10_models.vgg import vgg19_bn

if __name__ == "__main__":
    import os
    import sys

    model = vgg19_bn(device = torch.device(0), pretrained=True)
    # get gpu usage
    gpu_usage = os.popen("nvidia-smi --query-gpu=memory.used --format=csv").read()
    gpu_usage = gpu_usage.split("\n")[1:-1]
    gpu_usage = [int(x.split()[0]) for x in gpu_usage]
    gpu_usage = sum(gpu_usage)
    print(f"GPU usage: {gpu_usage} MB")
    
