import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
import argparse

from cifar10_models import all_classifiers
from cifar10_dataset.data import CIFAR10Data
from dpsgd import *

parser = argparse.ArgumentParser(description='Distinguishing attacks against DP-SGD with CIFAR10')
parser.add_argument('--model', type=str, choices = list(all_classifiers.keys()), required=True, help='Model to use for the attack')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
parser.add_argument('--clip_norm', type=float, default=1145141919810.0, help='Clipping norm for the gradients')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for the data loader')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the data loader')
parser.add_argument('--repeat', type=int, default=100, help='Number of times to repeat the attack')

if __name__ == "__main__":
    args = parser.parse_args()

    model = all_classifiers[args.model](pretrained=args.pretrained)
    model.to(torch.device(0))

    data = CIFAR10Data(batch_size=args.batch_size + 1, num_workers=args.num_workers)
    l2_sq_list = []
    l2_diff_sq_list = []

    for i in range(args.repeat):
        # Get the first batch of data
        train_loader = data.train_dataloader()
        batch = next(iter(train_loader))

        grad, grad_ = clipped_gradient_neighbouring_batch(model, batch, 1145141919810)

        grad = grad.to(torch.float64)
        grad_ = grad_.to(torch.float64)

        l2_sq_list.append(grad.norm(p=2).item() ** 2)
        l2_diff_sq_list.append((grad - grad_).norm(p=2).item() ** 2)


    l2_mean = math.sqrt(torch.tensor(l2_sq_list).mean().item())
    l2_median = math.sqrt(torch.tensor(l2_sq_list).median().item())
    l2_diff_mean = math.sqrt(torch.tensor(l2_diff_sq_list).mean().item())
    l2_diff_median = math.sqrt(torch.tensor(l2_diff_sq_list).median().item())
    print(f"cifar10,{args.model},{args.pretrained},{l2_mean},{l2_median},{l2_diff_mean},{l2_diff_median}")
