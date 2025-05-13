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
parser.add_argument('--epsilon', type=float, required=True, help='Epsilon parameter for the noise')
parser.add_argument('--delta', type=float, required=True, help='Delta parameter for the noise')
parser.add_argument('--beta', type=float, required=True, help='Beta parameter for the noise')
parser.add_argument('--gamma', type=float, required=True, help='Gamma parameter for the noise')
parser.add_argument('--clip_norm', type=float, required=True, help='Clipping norm for the gradients')
parser.add_argument('--batch_size', type=int, required=True, help='Batch size for the data loader')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the data loader')
parser.add_argument('--repeat', type=int, default=1, help='Number of times to repeat the attack')

if __name__ == "__main__":
    args = parser.parse_args()
    args.sensitivity = args.clip_norm * 2 / args.batch_size
    args.sigma = get_sigma(args.epsilon, args.delta, args.sensitivity)
    args.actual_epsilon = get_pancake_epsilon_low(args.sensitivity, args.sigma, args.beta, args.gamma, args.delta)

    model = all_classifiers[args.model](pretrained=args.pretrained)
    model.to(torch.device(0))

    data = CIFAR10Data(batch_size=args.batch_size + 1, num_workers=args.num_workers)
    
    test_loader = data.test_dataloader()

    # Get the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters())

    # Get the model accuracy before performing the attack 
    # Set model to evaluation mode
    model.eval()

    correct, total = 0, 0

    # Make sure no gradients are computed (faster and uses less memory)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    success = 0

    for i in range(args.repeat):
        # Get the first batch of data
        w_unnormalized = torch.randn(num_params, dtype=torch.float64, device=torch.device(0))
        train_loader = data.train_dataloader()
        batch = next(iter(train_loader))

        grad, grad_ = clipped_gradient_neighbouring_batch(model, batch, args.clip_norm)

        if torch.linalg.det(GPM_disginguishing_attack(grad, grad_, w_unnormalized, args.sigma, args.beta, args.gamma)).item() < 0:
            success += 1

    success_rate = success / args.repeat
    print(f'cifar10,{args.model},{num_params},{args.pretrained},{accuracy},{args.batch_size},{args.clip_norm},{args.beta},{args.gamma},{args.sigma},{args.epsilon},{args.actual_epsilon},{args.delta},{success_rate}')