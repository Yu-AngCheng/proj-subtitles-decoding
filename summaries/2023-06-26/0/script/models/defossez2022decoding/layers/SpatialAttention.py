#!/usr/bin/env python3
"""
Created on 14:51, Dec. 26th, 2022

@author: Norbert Zheng
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "SpatialAttention",
]

class SpatialAttention(K.layers.Layer):
    """
    `SpatialAttention` layer used to weight each channel with its corresponding location.
    """

    def __init__(self, n_output_channels, n_harmonics, drop_distance, **kwargs):
        """
        Initialize `SpatialAttention` object.
        :param n_output_channels: The number of output channels.
        :param n_harmonics: The number of harmonics of each attention weight.
        :param drop_distancec: The radius of the circle field to be dropped. We use a uniform distribution
            to draw the center of drop circle field from input eeg locations.
        :param kwargs: The arguments related to initialize `tf.keras.layers.Layer`-style object.
        """
        # First call super class init function to set up `K.layers.Layer`
        # style model and inherit it's functionality.
        super(SpatialAttention, self).__init__(**kwargs)

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
        # Initialize weight variables.
        # Z_re - (n_output_channels, n_harmonics*n_harmonics)
        self.Z_re = self.add_weight(name="Z_re", shape=(self.n_output_channels, self.n_harmonics * self.n_harmonics),
            trainable=True, dtype=tf.float32, initializer=tf.keras.initializers.GlorotUniform(),
            regularizer=tf.keras.regularizers.L2(0.001))
        # Z_im - (n_output_channels, n_harmonics*n_harmonics)
        self.Z_im = self.add_weight(name="Z_im", shape=(self.n_output_channels, self.n_harmonics * self.n_harmonics),
            trainable=True, dtype=tf.float32, initializer=tf.keras.initializers.GlorotUniform(),
            regularizer=tf.keras.regularizers.L2(0.001))
        # Build super to set up `K.layers.Layer`-style model and inherit it's network.
        super(SpatialAttention, self).build(input_shape)

    # def _calculate_attention_weights func
    def _calculate_attention_weights(self, locations):
        """
        Calculate the attention weights of the corresponding locations.
        :param locations: (batch_size, n_input_channels, 2) - The locations of input channels.
        :return weights: (batch_size, n_input_channels, n_output_channels) - The attention weights matrix.
        """
        # Construct harmonic components from locations.
        # locations_re - (batch_size, n_harmonics*n_harmonics, 2)
        locations_re = tf.meshgrid(tf.range(self.n_harmonics), tf.range(self.n_harmonics))
        locations_re = tf.concat([tf.reshape(locations_re_i, (-1, 1)) for locations_re_i in locations_re], axis=-1)
        locations_re = tf.cast(locations_re, dtype=tf.float32)
        # locations_im - (batch_size, n_harmonics*n_harmonics, 2)
        locations_im = tf.meshgrid(tf.range(self.n_harmonics), tf.range(self.n_harmonics))
        locations_im = tf.concat([tf.reshape(locations_im_i, (-1, 1)) for locations_im_i in locations_im], axis=-1)
        locations_im = tf.cast(locations_im, dtype=tf.float32)
        # Use `tf.matmul` to get the phase of each location.
        # locations_re - (batch_size, n_harmonics*n_harmonics, n_input_channels)
        locations_re = 2 * np.pi * tf.matmul(locations_re, tf.transpose(locations, perm=[0, 2, 1]))
        # locations_im - (batch_size, n_harmonics*n_harmonics, n_input_channels)
        locations_im = 2 * np.pi * tf.matmul(locations_im, tf.transpose(locations, perm=[0, 2, 1]))
        # Get the harmonic components of each location.
        locations_re = tf.cos(locations_re); locations_im = tf.sin(locations_im)
        # Calculate the corresponding attention weights matrix.
        # A - (batch_size, n_input_channels, n_output_channels)
        A = tf.transpose(tf.matmul(self.Z_re, locations_re) + tf.matmul(self.Z_im, locations_im), perm=[0, 2, 1])
        # Return the final `A`.
        return A

    # def call func
    def call(self, inputs, locations):
        """
        Forward layers in `SpatialAttention` to get the final result.
        :param inputs: (batch_size, seq_len, n_input_channels) - The input data.
        :param locations: (batch_size, n_input_channels, 2) - The locations of input channels.
        :return outputs: (batch_size, seq_len, n_output_channels) - The attention weighted data.
        """
        # Calculate the attention weights matrix.
        # A - (batch_size, n_input_channels, n_output_channels)
        A = self._calculate_attention_weights(locations)
        # Sampel a drop center from locations.
        # locations_drop - (batch_size, 1, 2)
        locations_drop = tf.stack([tf.expand_dims(locations_i[np.random.choice(locations_i.shape[0]),:], axis=0)\
            for locations_i in tf.unstack(locations)], axis=0)
        # Calculate the corresponding distance between each location and locations_drop.
        # locations_mask - (batch_size, n_input_channels)
        locations_mask = tf.sqrt(tf.reduce_sum((locations - locations_drop) ** 2, axis=-1)) >= self.drop_distance
        # Get the un-masked probaility.
        # probs - (batch_size, n_input_channels, n_output_channels)
        probs = tf.nn.softmax(A, axis=1)
        probs *= tf.cast(tf.expand_dims(locations_mask, axis=-1), dtype=tf.float32)
        probs /= tf.reduce_sum(probs, axis=1, keepdims=True)
        # Calculate the final `outputs`.
        # outputs - (batch_size, seq_len, n_output_channels)
        outputs = tf.matmul(inputs, probs)
        # Return the final `outputs`.
        return outputs

if __name__ == "__main__":
    # macro
    batch_size = 16; seq_len = 850; n_input_channels = 55
    n_output_channels = 128; n_harmonics = 32; drop_distance = 1.

    # Instantiate SpatialAttention.
    sa_inst = SpatialAttention(n_output_channels, n_harmonics, drop_distance)
    # Initialize input data & locations.
    inputs = tf.random.normal((batch_size, seq_len, n_input_channels), dtype=tf.float32)
    locations = tf.random.normal((batch_size, n_input_channels, 2), dtype=tf.float32)
    # Forward layers in `sa_inst`.
    outputs = sa_inst(inputs, locations)

