import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import sample_hclwe
import argparse

from cifar10_models import all_classifiers
from cifar10_models.vgg import vgg19_bn
from cifar10_dataset.data import CIFAR10Data

def clipped_gradient(model, sample, clip_norm):
    """
    Compute the clipped gradient for a single sample.
    Clip the gradients with respect to the L2 norm.
    
    Args:
        model: The model to compute the gradients for.
        sample: A tuple containing the input data and label.
        clip_norm: The maximum norm for gradient clipping.
    
    Returns:
        The clipped gradient for the sample, in a dictionary with the keys of the model parameters.
    """

    model.train()
    model.zero_grad()
    inputs, labels = sample
    inputs, labels = inputs.cuda(), labels.cuda()
    
    # Compute the gradient
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()

    # Compute the gradient for each parameter
    grad_dict = {}
    
    # Clip the gradient by the L2 norm
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Compute the L2 norm of the gradient
            grad_norm = torch.norm(param.grad.data, p=2)
            # If the norm is greater than the clip norm, scale the gradient
            if grad_norm > clip_norm:
                param.grad.data.mul_(clip_norm / grad_norm)
            # Store the clipped gradient
            grad_dict[name] = param.grad.data.clone()

    return grad_dict

def clipped_gradient_neighbouring_batch(model, batch, clip_norm):
    '''
    Batch is a sample of batch_size + 1
    compute the average clipped gradient of batch[0, 2, 3, 4, ..., batch_size] and batch [1, 2, 3, 4, ..., batch_size]
    return the average clipped gradient, as a dictionary with the keys of the model parameters
    '''

    avg_grad_dict = clipped_gradient(model, batch[0], clip_norm)
    avg_grad_dict_ = clipped_gradient(model, batch[1], clip_norm)

    for i in range(2, len(batch)):
        grad_dict = clipped_gradient(model, batch[i], clip_norm)
        for name, grad in grad_dict.items():
            avg_grad_dict[name] += grad
            avg_grad_dict_[name] += grad
    
    for name, grad in avg_grad_dict.items():
        avg_grad_dict[name] /= (len(batch) - 1)
        avg_grad_dict_[name] /= (len(batch) - 1)

    return avg_grad_dict, avg_grad_dict_

parser = argparse.ArgumentParser(description='Distinguishing attacks against DP-SGD')
parser.add_argument('--model', type=str, required=True, help='Model to use for the attack')
parser.add_argument('--pretrained', type=bool, action='store_true', help='Use pretrained model')
parser.add_argument('--sigma', type=float, required=True, help='Standard deviation of the noise')
parser.add_argument('--beta', type=float, required=True, help='Beta parameter for the noise')
parser.add_argument('--gamma', type=float, required=True, help='Gamma parameter for the noise')
parser.add_argument('--clip_norm', type=float, required=True, help='Clipping norm for the gradients')
parser.add_argument('--batch_size', type=int, required=True, help='Batch size for the data loader')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the data loader')
args = parser.parse_args()



if __name__ == "__main__":
    args = parser.parse_args()

    model = all_classifiers[args.model](device = torch.device(0), pretrained=args.pretrained)
    model.to(torch.device(0))
    data = CIFAR10Data(batch_size=args.batch_size + 1, num_workers=args.num_workers)
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()
