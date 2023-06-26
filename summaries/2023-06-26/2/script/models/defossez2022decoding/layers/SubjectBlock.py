#!/usr/bin/env python3
"""
Created on 11:16, Dec. 27th, 2022

@author: Norbert Zheng
"""
import tensorflow as tf
import tensorflow.keras as K
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from SpatialAttention import *
    from SubjectLayer import *
else:
    from .SpatialAttention import *
    from .SubjectLayer import *

__all__ = [
    "SubjectBlock",
]

class SubjectBlock(K.layers.Layer):
    """
    `SubjectBlock` layer used to transform each channel with specified subject id.
    """

    def __init__(self, n_output_channels, n_harmonics, drop_distance, **kwargs):
        """
        Initialize `SubjectBlock` object.
        :param n_output_channels: The number of output channels.
        :param n_harmonics: The number of harmonics of each attention weight.
        :param drop_distancec: The radius of the circle field to be dropped. We use a uniform distribution
            to draw the center of drop circle field from input eeg locations.
        :param kwargs: The arguments related to initialize `tf.keras.layers.Layer`-style object.
        """
        # First call super class init function to set up `K.layers.Layer`
        # style model and inherit it's functionality.
        super(SubjectBlock, self).__init__(**kwargs)

        # Initialize parameters.
        self.n_output_channels = n_output_channels
        self.n_harmonics = n_harmonics
        self.drop_distance = drop_distance

    """
    network funcs
    """
    # def build func
    def build(self, input_shape):
        """
        Build the network on the first call of `call`.
        :param input_shape: tuple - The shape of input data.
        """
        # Initialize SpatialAttention.
        self.sa_layer = SpatialAttention(n_output_channels=self.n_output_channels[0],
            n_harmonics=self.n_harmonics, drop_distance=self.drop_distance)
        # Initialize Conv1D.
        self.conv1d_layer = K.layers.Conv1D(self.n_output_channels[1], kernel_size=1, activation=None)
        # Initialize SubjectLayer.
        self.sl_layer = SubjectLayer(n_output_channels=self.n_output_channels[2])
        # Build super to set up `K.layers.Layer`-style model and inherit it's network.
        super(SubjectBlock, self).build(input_shape)

    # def call func
    def call(self, inputs):
        """
        Forward layers in `SubjectBlock` to get the final result.
        :param inputs: (3[list],) - The input data.
        :param subject_id: (batch_size,) - The subject id of input data.
        :return outputs: (batch_size, seq_len, n_output_channels) - The subject-transformed data.
        """
        # Get the [X,locations,subject_id] from inputs.
        # X - (batch_size, seq_len, n_input_channels)
        # locations - (batch_size, n_input_channels, 2)
        # subject_id - (batch_size,)
        X, locations, subject_id = inputs
        # Forward layers in SubjectBlock.
        # outputs - (batch_size, seq_len, n_output_channels)
        outputs = self.sa_layer(inputs=X, locations=locations)
        outputs = self.conv1d_layer(inputs=outputs)
        outputs = self.sl_layer(inputs=outputs, subject_id=subject_id)
        # Return the final `outputs`.
        return outputs

if __name__ == "__main__":
    import numpy as np

    # macro
    batch_size = 16; seq_len = 850; n_input_channels = 55; n_subjects = 42
    n_output_channels = [128, 128, 320]; n_harmonics = 32; drop_distance = 1.

    # Instantiate SubjectBlock.
    sb_inst = SubjectBlock(n_output_channels, n_harmonics, drop_distance)
    # Initialize input data & locations & subject_id.
    X = tf.random.normal((batch_size, seq_len, n_input_channels), dtype=tf.float32)
    locations = tf.random.normal((batch_size, n_input_channels, 2), dtype=tf.float32)
    subject_id = tf.cast(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=tf.float32)
    # Forward layers in `sb_inst`.
    outputs = sb_inst((X, locations, subject_id))

