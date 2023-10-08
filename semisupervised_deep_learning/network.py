import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from get_data import \
        get_dataset, \
        get_new_data_loader, \
        test_data_balance, \
        SUBSAMPLE_SIZES, \
        FINETUNE_SUBSAMPLE_SIZE, \
        PRETRAIN_SUBSAMPLE_SIZES, \
        NUM_CLASSES_DICT
from network_training import train

class Net(nn.Module):
    """
    Neural network model that can pre-train on one image dataset and then be finetuned onto another image dataset.
    """
    def __init__(self, pre_train_classes, generalization_classes):
        super(Net, self).__init__()
        self.pre_train_classes = pre_train_classes
        self.generalization_classes = generalization_classes

        # Initialize neural network layers used
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.lin_size = 256

        self.fc1 = nn.Linear(self.lin_size, self.pre_train_classes)
        self.generalizer = nn.Linear(self.lin_size, self.generalization_classes)

    def apply_convs(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.lin_size)
        return x
    
    def generalize(self, x):
        x = self.apply_convs(x)
        x = self.generalizer(x)
        return F.log_softmax(x, dim=-1)
        
    def forward(self, x):
        x = self.apply_convs(x)
        x = self.fc1(x)
        return F.log_softmax(x, dim=-1)

def test_network(n_samples=1, n_batches=500, minimum_acc=0.9, maximum_loss=0.01):
    """
    We default test the network by training it on the subsampled MNIST dataset with one sample per class.
    This is a very easy task and your network should be able to obtain high accuracy on the training set.

    Feel free to run this method with n_samples > 1 to check how well your network works.
    """
    data_subsamples, test_dataset = get_dataset('mnist')
    try:
        toy_dataset = data_subsamples[n_samples]
    except KeyError:
        raise KeyError("There is no subsampled MNIST dataset with {} samples per class".format(n_samples))

    network = Net(pre_train_classes=10, generalization_classes=10)
    optimizer = optim.SGD(network.parameters(), lr=0.01)

    train_losses, train_accuracies, _, _ = train(
        network,
        forward_call=network,
        optimizer=optimizer,
        train_dataset=toy_dataset,
        test_dataset=test_dataset, # we don't care about test set when doing unit testing
        n_batches=n_batches,
        batch_size=8,
        n_classes=10,
    )

    try:
        np.testing.assert_array_less(train_losses[-1], maximum_loss)
        np.testing.assert_array_less(minimum_acc, train_accuracies[-1])
        print('Simple network training test passed!')
    except AssertionError as E:
        raise E


if __name__ == '__main__':
    test_network(n_samples=1)
