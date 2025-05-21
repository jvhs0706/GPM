import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
import argparse

parser = argparse.ArgumentParser(description='Distinguishing attacks against DP-Hisotogram')
parser.add_argument('--epsilon', type=float, required=True, help='Epsilon parameter for the noise')
parser.add_argument('--delta', type=float, required=True, help='Delta parameter for the noise')
parser.add_argument('--beta', type=float, required=True, help='Beta parameter for the noise')
parser.add_argument('--gamma', type=float, help='Gamma parameter for the noise')
parser.add_argument('--num_records', type=int, required=True, help='Number of records in the dataset')
parser.add_argument('--num_bins', type=int, required=True, help='Number of bins in the histogram')
parser.add_argument('--repeat', type=int, default=1, help='Number of times to repeat the attack')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.gamma is None:
        args.gamma = 2 * math.sqrt(args.num_bins)
    elif args.gamma < 2 * math.sqrt(args.num_bins):
        raise ValueError("Gamma must be greater than 2 * sqrt(num_bins)")
    
    args.sensitivity = 1
    args.sigma = get_sigma(args.epsilon, args.delta, args.sensitivity)
    args.actual_epsilon = get_pancake_epsilon_low(args.sensitivity, args.sigma, args.beta, args.gamma, args.delta)

    total_success = 0
    mean_l2_sq = 0

    for i in range(args.repeat):
        # Get the first batch of data
        # creata a random vector, each element has value in 0 to num_bins - 1
        records = torch.randint(0, args.num_bins, (args.num_bins,), dtype=torch.float64)
        # get the histogram on records
        hist = torch.histogram(records, bins=args.num_bins, range=(-0.5, args.num_bins - 0.5))[0].cuda()
        hist_ = torch.histogram(records[:-1], bins=args.num_bins, range=(-0.5, args.num_bins - 0.5))[0].cuda()
        w_unnormalized = torch.randn(args.num_bins, dtype=torch.float64, device=torch.device(0))

        success, l2_error = GPM_disginguishing_attack(hist, hist_, w_unnormalized, args.sigma, args.beta, args.gamma, l2 = True)
        mean_l2_sq += l2_error.item()**2

        if success:
            total_success += 1

    mean_l2_sq /= args.repeat

    success_rate = total_success / args.repeat
    print(f'{args.num_records},{args.num_bins},{args.beta},{args.gamma},{args.sigma},{args.epsilon},{args.actual_epsilon},{args.delta},{success_rate},{math.sqrt(mean_l2_sq)}')
