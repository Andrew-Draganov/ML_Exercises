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
    """
    This is a by-hand implementation of the negative-log-likelihood loss.

    The reason we need this done by hand is that Pytorch's implementation assumes onehot targets.
    However, if we use an augmentation then we may have a target that is a linear combination of two classes.

    NOTE: we assume that the network's prediction did LOG SOFTMAX.
    """
    if not torch.is_tensor(pred) or not torch.is_tensor(target):
        raise ValueError('Floating point nll loss requires torch tensors for input')
    log_likelihoods = -pred * target
    return torch.mean(torch.sum(log_likelihoods, dim=-1))


def test_floating_point_nll_loss():
    predictions = np.array([
        [[1, 0]],
        [[0.5, 0.5]],
        [[0, 1]],
        [[0.5, 0.5]]
    ], dtype=np.float32)
    targets = np.array([
        [[1, 0]],
        [[1, 0]],
        [[0.5, 0.5]],
        [[0.5, 0.5]]
    ], dtype=np.float32)
    correct_outputs = np.array([-1, -0.5, -0.5, -0.5])

    # Evaluate correctness of the loss on SINGLE pred-target inputs
    for pred, target, correct_output in zip(predictions, targets, correct_outputs):
        pred = torch.tensor(pred)
        target = torch.tensor(target)
        loss = floating_point_nll_loss(pred, target).numpy()
        np.testing.assert_equal(loss, correct_output)

    # Evaluate correctness of the loss on BATCHWISE pred-target inputs
    pred_batch = torch.squeeze(torch.from_numpy(predictions), dim=1)
    target_batch = torch.squeeze(torch.from_numpy(targets), dim=1)
    loss = floating_point_nll_loss(pred_batch, target_batch).numpy()
    np.testing.assert_equal(loss, np.mean(correct_outputs))

    loss = floating_point_nll_loss(pred_batch, target_batch).numpy()
    np.testing.assert_equal(loss, np.mean(correct_outputs))


if __name__ == '__main__':
    test_floating_point_nll_loss()
