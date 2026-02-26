import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
import argparse

from mnist import * 
from dpsgd import *

parser = argparse.ArgumentParser(description='Distinguishing attacks against DP-SGD with MNIST')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
parser.add_argument('--clip_norm', type=float, default=1145141919810.0, help='Clipping norm for the gradients')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for the data loader')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the data loader')
parser.add_argument('--repeat', type=int, default=100, help='Number of times to repeat the attack')

if __name__ == "__main__":
    args = parser.parse_args()
    model = MNISTLeNet(load_weights=args.pretrained)
    model.to(torch.device(0))

    data = MNISTData(batch_size=args.batch_size + 1, num_workers=args.num_workers)
    l2_sq_list = []
    l2_diff_sq_list = []
    
    for i in range(args.repeat):
        # Get the first batch of data
        train_loader = data.train_dataloader()
        batch = next(iter(train_loader))

        grad, grad_ = clipped_gradient_neighbouring_batch(model, batch, args.clip_norm)

        grad = grad.to(torch.float64)
        grad_ = grad_.to(torch.float64)

        l2_sq_list.append(grad.norm(p=2).item() ** 2)
        l2_diff_sq_list.append((grad - grad_).norm(p=2).item() ** 2)
    
    l2_mean = math.sqrt(torch.tensor(l2_sq_list).mean().item())
    l2_median = math.sqrt(torch.tensor(l2_sq_list).median().item())
    l2_diff_mean = math.sqrt(torch.tensor(l2_diff_sq_list).mean().item())
    l2_diff_median = math.sqrt(torch.tensor(l2_diff_sq_list).median().item())
    print(f"mnist,lenet,{args.pretrained},{l2_mean},{l2_median},{l2_diff_mean},{l2_diff_median}")