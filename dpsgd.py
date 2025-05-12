import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

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
    grad_list = []

    for param in model.parameters():
        assert param.grad is not None, "Gradient is None. Ensure that the model is in training mode and loss is computed."
        grad_list.append(param.grad.data.clone().reshape(-1))
    grad = torch.cat(grad_list, dim=0)

    grad_norm = torch.norm(grad, p=2)
    if grad_norm > clip_norm:
        grad.mul_(clip_norm / grad_norm)
    
    return grad

def clipped_gradient_neighbouring_batch(model, batch, clip_norm):
    '''
    Batch is a sample of batch_size + 1
    compute the average clipped gradient of batch[0, 2, 3, 4, ..., batch_size] and batch [1, 2, 3, 4, ..., batch_size]
    return the average clipped gradient, as a dictionary with the keys of the model parameters
    '''

    images, labels = batch
    batch_size = images.size(0) - 1

    avg_grad = clipped_gradient(model, (images[None, 0], labels[None, 0]), clip_norm)
    avg_grad_ = clipped_gradient(model, (images[None, 1], labels[None, 1]), clip_norm)

    for i in range(2, batch_size + 1):
        grad = clipped_gradient(model, (images[None, i], labels[None, i]), clip_norm)
        avg_grad += grad
        avg_grad_ += grad
    
    avg_grad /= batch_size
    avg_grad_ /= batch_size

    return avg_grad, avg_grad_