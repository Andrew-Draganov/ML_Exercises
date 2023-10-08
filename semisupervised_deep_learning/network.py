import torch
import torch.nn as nn
import torch.nn.functional as F

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
