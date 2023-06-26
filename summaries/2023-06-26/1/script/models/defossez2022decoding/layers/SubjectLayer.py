#!/usr/bin/env python3
"""
Created on 19:29, Dec. 26th, 2022

@author: Norbert Zheng
"""
import tensorflow as tf
import tensorflow.keras as K
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "SubjectLayer",
]

class SubjectLayer(K.layers.Layer):
    """
    `SubjectLayer` layer used to map each channel with specified subject id.
    """

    def __init__(self, n_output_channels, **kwargs):
        """
        Initialize `SubjectLayer` object.
        :param n_output_channels: The number of output channels.
        :param kwargs: The arguments related to initialize `tf.keras.layers.Layer`-style object.
        """
        # First call super class init function to set up `K.layers.Layer`
        # style model and inherit it's functionality.
        super(SubjectLayer, self).__init__(**kwargs)

        # Initialize parameters.
        self.n_output_channels = n_output_channels

    """
    network funcs
    """
    # def build func
    def build(self, input_shape):
        """
        Build the network on the first call of `call`.
        :param input_shape: tuple - The shape of input data.
        """
        # Initialize weight variables.
        # M - (1 -> (n_input_channels*n_input_channels,))
        n_units = input_shape[-1] * self.n_output_channels
        self.M = K.layers.Dense(n_units, activation=None, use_bias=False, kernel_initializer="glorot_uniform", name="M")
        # Build super to set up `K.layers.Layer`-style model and inherit it's network.
        super(SubjectLayer, self).build(input_shape)

    # def call func
    def call(self, inputs, subject_id):
        """
        Forward layers in `SubjectLayer` to get the final result.
        :param inputs: (batch_size, seq_len, n_input_channels) - The input data.
        :param subject_id: (batch_size, n_subjects) - The subject id of input data.
        :return outputs: (batch_size, seq_len, n_output_channels) - The subject-transformed data.
        """
        # Get subject-specified transformation matrix.
        # M_s - (batch_size, n_input_channels, n_output_channels)
        M_s = tf.reshape(self.M(subject_id), (-1, inputs.shape[-1], self.n_output_channels))
        # Use subject-specified transformation matrix to get the subject-transformed data.
        # outputs - (batch_size, seq_len, n_output_channels)
        outputs = tf.matmul(inputs, M_s)
        # Return the final `outputs`.
        return outputs

if __name__ == "__main__":
    import numpy as np

    # macro
    batch_size = 16; seq_len = 850; n_input_channels = 55; n_output_channels = 320; n_subjects = 42

    # Instantiate SubjectLayer.
    sl_inst = SubjectLayer(n_output_channels)
    # Initialize input data & subject_id.
    inputs = tf.random.normal((batch_size, seq_len, n_input_channels), dtype=tf.float32)
    subject_id = tf.cast(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=tf.float32)
    # Forward layers in `sl_inst`.
    outputs = sl_inst(inputs, subject_id)

