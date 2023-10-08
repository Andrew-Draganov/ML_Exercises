import numpy as np
import torch

def torch_onehot(labels, n_classes):
    onehot_labels = torch.zeros((len(labels), n_classes))
    onehot_labels[torch.arange(len(labels)), labels] = 1
    return onehot_labels

def np_onehot(labels, n_classes):
    onehot_labels = np.zeros((len(labels), n_classes))
    onehot_labels[np.arange(len(labels)), labels] = 1
    return onehot_labels

def running_mean(vals, steps=30):
    means = np.zeros_like(vals)
    for i, loss in enumerate(vals):
        if i < steps:
            means[i] = np.mean(vals[:i])
        else:
            means[i] = np.mean(vals[i-steps:i])
    return means

def floating_point_nll_loss(pred, target):
    """ STUDENTS CODE """
    if not torch.is_tensor(pred) or not torch.is_tensor(target):
        raise ValueError('Floating point nll loss requires torch tensors for input')
    log_likelihoods = -pred * target
    return torch.mean(torch.sum(log_likelihoods, dim=-1))

