import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
import argparse

import time

parser = argparse.ArgumentParser(description='Timing against DP-Hisotogram')
parser.add_argument('--epsilon', type=float, required=True, help='Epsilon parameter for the noise')
parser.add_argument('--delta', type=float, required=True, help='Delta parameter for the noise')
parser.add_argument('--beta', type=float, required=True, help='Beta parameter for the noise')
parser.add_argument('--gamma_factor', type=float, help='Gamma factor to multiply with sqrt(num_bins)')
parser.add_argument('--gamma', type=float, help='Gamma parameter for the noise')
parser.add_argument('--num_records', type=int, required=True, help='Number of records in the dataset')
parser.add_argument('--num_bins', type=int, required=True, help='Number of bins in the histogram')
parser.add_argument('--repeat', type=int, default=1, help='Number of times to repeat')
parser.add_argument('--gpm', action='store_true', help='Use Gaussian Pancake Mechanism (GPM)')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for computation')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.gamma is None:
        if args.gamma_factor is None:
            args.gamma = 2 * math.sqrt(args.num_bins)
        else:
            args.gamma = args.gamma_factor * math.sqrt(args.num_bins)
    elif args.gamma < 2 * math.sqrt(args.num_bins):
        raise ValueError("Gamma must be greater than 2 * sqrt(num_bins)")
    
    args.sensitivity = 1
    args.sigma = get_sigma(args.epsilon, args.delta, args.sensitivity)
    args.actual_epsilon = get_pancake_epsilon_low(args.sensitivity, args.sigma, args.beta, args.gamma, args.delta)


    total_rt = 0
    for i in range(args.repeat + 1):
        records = torch.randint(0, args.num_bins, (args.num_records,), dtype=torch.float)
        w_unnormalized = torch.randn(args.num_bins,  dtype=torch.float, device=args.device)
        w_normalized = w_unnormalized / torch.norm(w_unnormalized, p=2,  dtype=torch.float)
        # get the histogram on records
        hist = torch.histogram(records, bins=args.num_bins, range=(-0.5, args.num_bins - 0.5))[0].to(device=args.device)
        if args.gpm:
            res,rt = GPM(hist, w_normalized, args.sigma, args.beta, args.gamma, timed=True)
        else:
            res,rt = GM(hist, args.sigma,timed=True)
        if i > 0:  # skip the first run to avoid cold start effects
            total_rt += rt

    mech_name = "GPM" if args.gpm else "GM"
    avg_rt = 1000 * (total_rt / args.repeat)

    print(f'{args.num_records},{args.num_bins},{args.beta},{args.gamma},{args.sigma},{args.epsilon},{args.actual_epsilon},{args.delta},{mech_name},{avg_rt}')