import tensorflow as tf
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.pardir)
from utils import DotDict

__all__ = [
    "naive_cnn_params",
]

class naive_cnn_params(DotDict):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `naive_cnn` on initialization.
    """
    # Internal macro parameter.
    _precision = "float32"

    def __init__(self, dataset="meg_liu2019cell"):
        """
        Initialize `naive_cnn_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(naive_cnn_params, self).__init__()

        ## Generate all parameters hierarchically.
        # -- Model parameters
        self.model = naive_cnn_params._gen_model_params(dataset)
        # -- Training parameters
        self.train = naive_cnn_params._gen_train_params(dataset)

        ## Do init iteration.
        naive_cnn_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## -- Model parameters.
        # Calculate current learning rate.
        self.model.lr_i = self.model.lr

    """
    generate funcs
    """
    ## def _gen_model_* funcs
    # def _gen_model_params func
    @staticmethod
    def _gen_model_params(dataset):
        """
        Generate model parameters.
        """
        # Initialize `model_params`.
        model_params = DotDict()

        ## -- Normal parameters
        # The type of dataset.
        model_params.dataset = dataset
        # Normal parameters related to meg_liu2019cell dataset.
        if model_params.dataset == "meg_liu2019cell":
            # The size of input channels.
            model_params.n_channels = 273
            # The size of output classes.
            model_params.n_labels = 8
            # The learning rate of optimizer.
            model_params.lr = 3e-4
        # Normal parameters related to eeg dataset.
        elif model_params.dataset == "eeg":
            # The size of input channels.
            model_params.n_channels = 55
            # The size of output classes.
            model_params.n_labels = 15
            # The learning rate of optimizer.
            model_params.lr = 3e-4
        # Normal parameters related to other dataset.
        else:
            # The size of input channels.
            model_params.n_channels = 273
            # The size of output classes.
            model_params.n_labels = 8
        ## -- CNN parameters
        model_params.cnn = naive_cnn_params._gen_model_cnn_params(model_params)
        ## -- Fully connect parameters
        model_params.fc = naive_cnn_params._gen_model_fc_params(model_params)

        # Return the final `model_params`.
        return model_params

    # def _gen_model_cnn_params func
    @staticmethod
    def _gen_model_cnn_params(model_params):
        """
        Generate model.cnn parameters.
        """
        # Initialize `model_cnn_params`.
        model_cnn_params = DotDict()

        ## -- Normal parameters (related to Conv1d)
        # Normal parameters related to meg_liu2019cell dataset.
        if model_params.dataset == "meg_liu2019cell":
            # The dimension of input vector.
            model_cnn_params.d_input = model_params.n_channels
            # The number of filters of each CNN layer.
            model_cnn_params.n_filters = [128, 128]
            # The size of kernel of each CNN layer.
            model_cnn_params.d_kernel = [5, 7]
            # The length of stride of each CNN layer.
            model_cnn_params.strides = [1, 1]
            # The length of padding of each CNN layer.
            model_cnn_params.padding = ["same", "same"]
            # The dilation rate of each CNN layer.
            model_cnn_params.dilation_rate = [1, 2]
            ## -- Normal parameters (related to MaxPool1d)
            # The size of max pooling kernel of each CNN layer.
            model_cnn_params.d_pooling_kernel = 2
        # Normal parameters related to other dataset.
        else:
            # The dimension of input vector.
            model_cnn_params.d_input = model_params.n_channels
            # The number of filters of each CNN layer.
            model_cnn_params.n_filters = [128, 128]
            # The size of kernel of each CNN layer.
            model_cnn_params.d_kernel = [5, 7]
            # The length of stride of each CNN layer.
            model_cnn_params.strides = [1, 1]
            # The length of padding of each CNN layer.
            model_cnn_params.padding = ["same", "same"]
            # The dilation rate of each CNN layer.
            model_cnn_params.dilation_rate = [1, 2]
            ## -- Normal parameters (related to MaxPool1d)
            # The size of max pooling kernel of each CNN layer.
            model_cnn_params.d_pooling_kernel = 2

        # Return the final `model_cnn_params`.
        return model_cnn_params

    # def _gen_model_fc_params func
    @staticmethod
    def _gen_model_fc_params(model_params):
        """
        Generate model.fc parameters.
        """
        # Initialize `model_fc_params`.
        model_fc_params = DotDict()

        ## -- Normal parameters
        # The dimension of output vector.
        model_fc_params.d_output = model_params.n_labels

        # Return the final `model_fc_params`.
        return model_fc_params

    ## def _gen_train_* funcs
    # def _gen_train_params func
    @staticmethod
    def _gen_train_params(dataset):
        """
        Generate training parameters.
        """
        # Initialize `train_params`.
        train_params = DotDict()

        ## -- Normal parameters
        # The type of dataset.
        train_params.dataset = dataset
        # Precision parameter.
        train_params.precision = getattr(tf, naive_cnn_params._precision)\
            if hasattr(tf, naive_cnn_params._precision) else tf.float32
        # Whether use graph mode or eager mode.
        train_params.use_graph_mode = True
        # The ratio of train dataset. The rest is test dataset.
        train_params.train_ratio = 0.8
        # Size of buffer used in shuffle.
        train_params.buffer_size = int(1e4)
        ## -- Dataset-specific parameters
        # Normal parameters related to meg_liu2019cell dataset.
        if train_params.dataset == "meg_liu2019cell":
            # Number of epochs used in training process.
            train_params.n_epochs = 20
            # Number of batch size used in training process.
            train_params.batch_size = 16
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            train_params.n_epochs = 20
            # Number of batch size used in training process.
            train_params.batch_size = 16

        # Return the final `train_params`.
        return train_params

if __name__ == "__main__":
    # Instantiate `naive_cnn_params`.
    naive_cnn_params_inst = naive_cnn_params(dataset="meg_liu2019cell")

