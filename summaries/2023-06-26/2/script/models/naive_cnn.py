#!/usr/bin/env python3
"""
Created on 15:50, Dec. 20th, 2022

@author: Norbert Zheng
"""
import copy as cp
import numpy as np
import tensorflow as tf
# import tensorflow.keras as K
import keras as K

# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.pardir)
import utils

__all__ = [
    "naive_cnn",
]

class naive_cnn(K.Model):
    """
    `naive_cnn` model, with considering time information.
    """

    def __init__(self, params):
        """
        Initialize `naive_cnn` object.
        :param params: Model parameters initialized by naive_cnn_params, updated by params.iteration.
        """
        # First call super class init function to set up `K.Model`
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
        ## Initialize trainable cnn layers.
        model_cnn = K.models.Sequential(name="CNN")
        # Add `Conv1D` & `MaxPool1D` layers.
        for cnn_idx in range(len(self.params.cnn.n_filters)):
            # Initialize `Conv1D` layer. `tf.keras.layers.Conv1D` is different from `torch.nn.Conv1d`. It doesn't have
            # `in_channels` argument. And `filters` argument equals to `out_channels` argument.
            out_channels, kernel_size = self.params.cnn.n_filters[cnn_idx], self.params.cnn.d_kernel[cnn_idx]
            strides, padding = self.params.cnn.strides[cnn_idx], self.params.cnn.padding[cnn_idx]
            dilation_rate = self.params.cnn.dilation_rate[cnn_idx]
            model_cnn.add(K.layers.Conv1D(
                # Modified `Conv1D` layer parameters.
                filters=out_channels, kernel_size=kernel_size, strides=strides,
                padding=padding, dilation_rate=dilation_rate, name="Conv1D_{:d}".format(cnn_idx),
                # Default `Conv1D` layer parameters.
                data_format="channels_last", groups=1, activation=None, use_bias=True,
                kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None,
                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None
            ))
            # Initialize `MaxPool1D` layer.
            if isinstance(self.params.cnn.d_pooling_kernel, list):
                kernel_size = self.params.cnn.d_pooling_kernel[cnn_idx]
                model_cnn.add(K.layers.MaxPool1D(
                    # Modified `MaxPool1D` layer parameters.
                    pool_size=kernel_size, strides=1, name="MaxPool1D_{:d}".format(cnn_idx),
                    # Default `MaxPool1D` layer parameters.
                    padding="valid", data_format="channels_last"
                ))
            else:
                # Only add `MaxPool1D` layer at the last layer of cnn.
                if cnn_idx == len(self.params.cnn.n_filters) - 1:
                    kernel_size = self.params.cnn.d_pooling_kernel
                    model_cnn.add(K.layers.MaxPool1D(
                        # Modified `MaxPool1D` layer parameters.
                        pool_size=kernel_size, name="MaxPool1D_{:d}".format(cnn_idx),
                        # Default `MaxPool1D` layer parameters.
                        strides=None, padding="valid", data_format="channels_last"
                    ))
        # Add `Dropout` after `MaxPool1D` layer.
        model_cnn.add(K.layers.Dropout(rate=0.5, name="Dropout_{}".format("cnn")))
        # Add `BatchNormalization` at the last layer of cnn layers.
        model_cnn.add(K.layers.BatchNormalization(
            # Modified `BatchNormalization` parameters.
            name="BatchNormalization_{}".format("cnn"),
            # Default `BatchNormalization` parameters.
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
            gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones",
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None
        ))
        ## Initialize trainable fc layer. Then add FullConnect layer to do classification task.
        model_fc = K.models.Sequential(layers=[
            K.layers.Flatten(data_format="channels_last"),
            K.layers.Dense(
                # Modified `Dense` parameters.
                units=self.params.fc.d_output, activation="sigmoid",
                # Default `Dense` parameters.
                use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None,
                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None
            ),
            K.layers.Softmax(axis=-1)
        ], name="FullConnect")
        ## Stack all layers to get the final model.
        self.model = K.models.Sequential([model_cnn, model_fc,])
        optimizer = K.optimizers.Adam(learning_rate=self.params.lr_i)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy",])

    """
    network funcs
    """
    # def fit func
    @utils.model.tf_scope
    def fit(self, X_train, y_train, epochs=1, batch_size=16):
        """
        Forward `naive_cnn` to get the final predictions.
        :param X_train: (n_train, seq_len, n_chennals) - The trainset data.
        :param y_train: (n_train, n_labels) - The trainset labels.
        :param epochs: int - The number of epochs.
        :param batch_size: int - The size of batch.
        """
        # Fit the model using [X_train,y_train].
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # def evaluate func
    @utils.model.tf_scope
    def evaluate(self, X_test, y_test):
        """
        Calculate loss between tensors value and target.
        :param X_test: (n_test, seq_len, n_chennals) - The trainset data.
        :param y_test: (n_test, n_labels) - The trainset labels.
        :return loss: float - The loss of current evaluation process.
        :return accuracy: float - The accuracy of current evaluation process.
        """
        return self.model.evaluate(X_test, y_test)

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
    X = tf.random.uniform((batch_size, seq_len, n_channels), dtype=tf.float32)
    y = tf.cast(np.eye(n_labels)[np.random.randint(0, n_labels, size=(batch_size,))], dtype=tf.float32)
    # Fit and evaluate naive_cnn_inst.
    naive_cnn_inst.fit(X, y); _, _ = naive_cnn_inst.evaluate(X, y)

