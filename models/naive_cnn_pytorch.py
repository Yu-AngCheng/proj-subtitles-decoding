#!/usr/bin/env python3
"""
Created on 15:50, Dec. 20th, 2022

@author: Norbert Zheng
"""
import copy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.pardir)
import utils

__all__ = [
    "naive_cnn",
]

class naive_cnn(nn.Module):
    """
    `naive_cnn` model, with considering time information.
    """

    def __init__(self, params):
        """
        Initialize `naive_cnn` object.
        :param params: Model parameters initialized by naive_cnn_params, updated by params.iteration.
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(naive_cnn, self).__init__()

        # Copy hyperparameters (e.g. network sizes) from parameter dotdict,
        # usually generated from naive_cnn_params() in params/naive_cnn_params.py.
        self.params = cp.deepcopy(params)

        # Create trainable vars.
        self._init_trainable()

    """
    init funcs
    """
    # def _init_trainable func
    def _init_trainable(self):
        """
        Initialize trainable variables.
        """
        self.layers = []
        in_channels = self.params.model.n_channels

        # Add `Conv1d` & `MaxPool1d` layers.
        for cnn_idx in range(len(self.params.cnn.n_filters)):
            out_channels, kernel_size = self.params.cnn.n_filters[cnn_idx], self.params.cnn.d_kernel[cnn_idx]
            strides, padding = self.params.cnn.strides[cnn_idx], self.params.cnn.padding[cnn_idx]
            dilation_rate = self.params.cnn.dilation_rate[cnn_idx]
            self.layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=padding, dilation=dilation_rate))
            self.layers.append(nn.ReLU())

            if isinstance(self.params.cnn.d_pooling_kernel, list):
                pool_size = self.params.cnn.d_pooling_kernel[cnn_idx]
                self.layers.append(nn.MaxPool1d(kernel_size=pool_size))
            else:
                # Only add `MaxPool1d` layer at the last layer of cnn.
                if cnn_idx == len(self.params.cnn.n_filters) - 1:
                    pool_size = self.params.cnn.d_pooling_kernel
                    self.layers.append(nn.MaxPool1d(kernel_size=pool_size))

            in_channels = out_channels

        self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nn.BatchNorm1d(num_features=in_channels))
        self.layers.append(nn.Flatten())

        # FC layer
        self.layers.append(nn.Linear(in_features=in_channels * self.params.fc.d_input, out_features=self.params.fc.d_output))
        self.layers.append(nn.Sigmoid())
        self.layers.append(nn.Softmax(dim=-1))

        self.model = nn.Sequential(*self.layers)

    """
    network funcs
    """
    # def fit func
    def fit(self, X_train, y_train, epochs=1, batch_size=16):
        """
        Forward `naive_cnn` to get the final predictions.
        :param X_train: (n_train, seq_len, n_chennals) - The trainset data.
        :param y_train: (n_train, n_labels) - The trainset labels.
        :param epochs: int - The number of epochs.
        :param batch_size: int - The size of batch.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.params.lr_i)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for inputs, labels in dataloader:
                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, torch.max(labels, 1)[1])
                loss.backward()
                optimizer.step()

    # def evaluate func
    def evaluate(self, X_test, y_test):
        """
        Calculate loss between tensors value and target.
        :param X_test: (n_test, seq_len, n_chennals) - The trainset data.
        :param y_test: (n_test, n_labels) - The trainset labels.
        :return loss: float - The loss of current evaluation process.
        :return accuracy: float - The accuracy of current evaluation process.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs, 1)
            _, true_labels = torch.max(y_test, 1)
            correct = (predicted == true_labels).sum().item()
            accuracy = correct / y_test.size(0)

        return accuracy

if __name__ == "__main__":
    # local dep
    from params import naive_cnn_params

    # Initialize training process.
    utils.model.set_seeds(42)
    # Initialize params.
    batch_size = 16; seq_len = 600; dataset = "meg_liu2019cell"
    naive_cnn_params_inst = naive_cnn_params(dataset=dataset)
    n_channels = naive_cnn_params_inst.model.n_channels; n_labels = naive_cnn_params_inst.model.n_labels
    # Get naive_cnn_inst.
    naive_cnn_inst = naive_cnn(naive_cnn_params_inst.model)
    # Initialize inputs.
    X = torch.rand((batch_size, n_channels, seq_len))
    y = F.one_hot(torch.randint(n_labels, (batch_size,)), num_classes=n_labels).float()
    # Fit and evaluate naive_cnn_inst.
    naive_cnn_inst.fit(X, y); acc = naive_cnn_inst.evaluate(X, y)
    print("Accuracy: ", acc)
