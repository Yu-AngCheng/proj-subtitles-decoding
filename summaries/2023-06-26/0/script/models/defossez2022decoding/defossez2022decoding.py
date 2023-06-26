import copy as cp
import tensorflow as tf
import tensorflow.keras as K
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    from layers import *
else:
    from .layers import *
import utils

__all__ = [
    "defossez2022decoding",
]

class defossez2022decoding(K.Model):
    """
    `defossez2022decoding` model reproduced from defossez2022decoding paper.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `defossez2022decoding` object.
        :param params: Model parameters initialized by defossez2022decoding_params, updated by params.iteration.
        :param kwargs: The arguments related to initialize `tf.keras.Model`-style object.
        """
        # First call super class init function to set up `K.Model`
        # style model and inherit it's functionality.
        super(defossez2022decoding, self).__init__(**kwargs)

        # Copy hyperparameters (e.g. network sizes) from parameter dotdict,
        # usually generated from cnn_bgru_params() in params/cnn_bgru_params.py.
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
        # Initialize subject block.
        self.subject_block = SubjectBlock([270, 270, 320], 32, 0.3)
        # Initialize conv1d block.
        self.conv1d_block = K.models.Sequential()
        self.conv1d_block.add(Conv1DBlock([320, 320, 640], [3, 3, 3], [1, 2, 2]))
        self.conv1d_block.add(Conv1DBlock([320, 320, 640], [3, 3, 3], [4, 8, 2]))
        self.conv1d_block.add(Conv1DBlock([320, 320, 640], [3, 3, 3], [16, 1, 2]))
        self.conv1d_block.add(Conv1DBlock([320, 320, 640], [3, 3, 3], [2, 4, 2]))
        self.conv1d_block.add(Conv1DBlock([320, 320, 640], [3, 3, 3], [8, 16, 2]))
        # Initialize feature block.
        self.feature_block = K.models.Sequential()
        self.feature_block.add(K.layers.Conv1D(640, kernel_size=1, activation="gelu"))
        self.feature_block.add(K.layers.Conv1D(self.params.n_features, kernel_size=1, activation=None))
        # Initialize classification block.
        self.classification_block = K.models.Sequential()
        self.classification_block.add(K.layers.Flatten(data_format="channels_last"))
        # Note: Removing `sigmoid` activation will greatly improve the performance on mnist data.
        self.classification_block.add(K.layers.Dense(units=self.params.n_labels, activation=None, use_bias=True))

    """
    network funcs
    """
    # def call func
    @utils.model.tf_scope
    def call(self, inputs, training=None, mask=None):
        """
        Forward `defossez2022decoding` to get the final predictions.
        :param inputs: tuple - The input data.
        :param training: Boolean or boolean scalar tensor, indicating whether to run
            the `Network` in training mode or inference mode.
        :param mask: A mask or list of masks. A mask can be either a tensor or None (no mask).
        :return outputs: (batch_size, n_labels) - The output labels.
        :return loss: tf.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_input_channels)
        # locations - (batch_size, n_input_channels, 2)
        # subject_id - (batch_size,)
        # y - (batch_size,)
        X = inputs[0]; locations = inputs[1]; subject_id = inputs[2]; y = inputs[3]
        # Forward subject block to execute subject-level transformation.
        # outputs - (batch_size, seq_len, params.subject_block.n_output_channels[-1])
        outputs = self.subject_block((X, locations, subject_id))
        # Forward conv1d block to execute convolution.
        # outputs - (batch_size, seq_len, params.conv1d_block.n_output_channels[-1]) 
        outputs = self.conv1d_block(outputs)
        # Forward feature block to extract features.
        # outputs - (batch_size, seq_len, params.n_features)
        outputs = self.feature_block(outputs)
        # Forward classification block to do classification.
        # outputs - (batch_size, seq_len, params.n_labels)
        outputs = self.classification_block(outputs)
        # Calculate the final loss.
        # loss - (batch_size,)
        loss = self._loss(outputs, y)
        # Return the final `outputs` & `loss`.
        return outputs, loss

    # def _loss func
    @utils.model.tf_scope
    def _loss(self, value, target):
        """
        Calculate loss between tensors value and target.
        :param value: (batch_size,) - One-hot value of the object.
        :param target: (batch_size,) - One-hot traget of the object.
        :return loss: (batch_size,) - Loss between value and target.
        """
        return tf.reduce_mean(self._loss_bce(value, target))

    # def _loss_bce func
    @utils.model.tf_scope
    def _loss_bce(self, value, target):
        """
        Calculates binary cross entropy between tensors value and target.
        Get mean over last dimension to keep losses of different batches separate.
        :param value: (batch_size,) - Value of the object.
        :param target: (batch_size,) - Target of the object.
        :return loss: (batch_size,) - Loss between value and target.
        """
        # Note: `tf.nn.softmax_cross_entropy_with_logits` needs unscaled log probabilities,
        # we must not add `tf.nn.Softmax` layer at the last of the model.
        # loss - (batch_size,)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=target,logits=value) if type(value) is not list else\
            [tf.nn.softmax_cross_entropy_with_logits(labels=target[i],logits=value[i]) for i in range(len(value))]
        # Return the final `loss`.
        return loss

if __name__ == "__main__":
    import numpy as np
    # local dep
    from params import defossez2022decoding_params

    # macro
    batch_size = 16; seq_len = 850; n_channels = 55; n_features = 128; n_labels = 15; n_subjects = 42

    # Initialize training process.
    utils.model.set_seeds(42)
    # Initialize params.
    d2d_params_inst = defossez2022decoding_params(n_channels=n_channels, n_features=n_features, n_labels=n_labels)
    # Instantiate defossez2022decoding.
    d2d_inst = defossez2022decoding(d2d_params_inst.model)
    # Initialize input data & locations & subject_id.
    X = tf.random.normal((batch_size, seq_len, n_channels), dtype=tf.float32)
    locations = tf.random.normal((batch_size, n_channels, 2), dtype=tf.float32)
    subject_id = tf.cast(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=tf.float32)
    y = tf.cast(tf.one_hot(tf.cast(tf.range(batch_size), dtype=tf.int64), n_labels), dtype=tf.float32)
    # Forward layers in `sb_inst`.
    inputs = (X, locations, subject_id, y)
    outputs, loss = d2d_inst(inputs); d2d_inst.summary()

