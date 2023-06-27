#!/usr/bin/env python3
"""
Created on 16:00, Dec. 20th, 2022

@author: Norbert Zheng
"""
import torch
from utils import DotDict

__all__ = [
    "Defossez2022DecodingParams",
]

class Defossez2022DecodingParams(DotDict):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `Defossez2022Decoding` on initialization.
    """

    def __init__(self, n_channels=273, n_features=128, n_labels=8):
        """
        Initialize `Defossez2022DecodingParams`.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit its functionality.
        super(Defossez2022DecodingParams, self).__init__()

        ## Generate all parameters hierarchically.
        # -- Model parameters
        self.model = Defossez2022DecodingParams._gen_model_params(n_channels, n_features, n_labels)
        # -- Train parameters
        self.train = Defossez2022DecodingParams._gen_train_params()

    """
    generate funcs
    """
    ## def _gen_model_* funcs
    # def _gen_model_params func
    @staticmethod
    def _gen_model_params(n_channels, n_features, n_labels):
        """
        Generate model parameters.
        """
        # Initialize `model_params`.
        model_params = DotDict()

        ## -- Normal parameters
        # The size of input channels.
        model_params.n_channels = n_channels
        # The size of output features.
        model_params.n_features = n_features
        # The size of output labels.
        model_params.n_labels = n_labels

        # Return the final `model_params`.
        return model_params

    ## def _gen_train_* funcs
    # def _gen_train_params func
    @staticmethod
    def _gen_train_params():
        """
        Generate train parameters.
        """
        # Initialize `train_params`.
        train_params = DotDict()

        ## -- Normal parameters
        # The type of dataset.
        train_params.dataset = "meg.gwilliams2022neural" #"meg_liu2019cell"
        # Number of epochs used in training process.
        train_params.n_epochs = 20
        # The learning rate of training process.
        train_params.lr = 3e-4
        # The ratio of train dataset. The rest is test dataset.
        train_params.train_ratio = 0.8
        # Number of batch size used in training process.
        train_params.batch_size = 128
        # Period of iterations to log information.
        train_params.i_log = 10
        # Period of iterations to execute test.
        train_params.i_test = 10
        # Period of iterations to save model.
        train_params.i_model = 100

        # Return the final `train_params`.
        return train_params

if __name__ == "__main__":
    # Instantiate `Defossez2022DecodingParams`.
    defossez2022decoding_params_inst = Defossez2022DecodingParams(n_channels=273, n_features=128, n_labels=8)
