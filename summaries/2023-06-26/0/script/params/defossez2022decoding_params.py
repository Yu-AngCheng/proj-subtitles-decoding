import tensorflow as tf
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.pardir)
from utils import DotDict

__all__ = [
    "defossez2022decoding_params",
]

class defossez2022decoding_params(DotDict):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `defossez2022decoding` on initialization.
    """
    # Initialize macro parameter.
    _precision = "float32"

    def __init__(self, n_channels=273, n_features=128, n_labels=8):
        """
        Initialize `defossez2022decoding_params`.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(defossez2022decoding_params, self).__init__()

        ## Generate all parameters hierarchically.
        # -- Model parameters
        self.model = defossez2022decoding_params._gen_model_params(n_channels, n_features, n_labels)
        # -- Train parameters
        self.train = defossez2022decoding_params._gen_train_params()

        ## Do init iteration.
        defossez2022decoding_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## -- Train parameters
        # Calculate current learning rate.
        self.train.lr_i = self.train.lr

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
        # Precision parameter.
        train_params.precision = getattr(tf, defossez2022decoding_params._precision)\
            if hasattr(tf, defossez2022decoding_params._precision) else tf.float32
        # Number of epochs used in training process.
        train_params.n_epochs = 20
        # Whether use graph mode or eager mode.
        train_params.use_graph_mode = True
        # The learning rate of training process.
        train_params.lr = 3e-4
        # The ratio of train dataset. The rest is test dataset.
        train_params.train_ratio = 0.8
        # Number of batch size used in training process.
        train_params.batch_size = 128
        # Size of buffer used in shuffle.
        train_params.buffer_size = int(1e4)
        # Period of iterations to log information.
        train_params.i_log = 10
        # Period of iterations to execute test.
        train_params.i_test = 10
        # Peroid of iterations to save model.
        train_params.i_model = 100

        # Return the final `train_params`.
        return train_params

if __name__ == "__main__":
    # Instantiate `defossez2022decoding_params`.
    defossez2022decoding_params_inst = defossez2022decoding_params(n_channels=273, n_features=128, n_labels=8)

