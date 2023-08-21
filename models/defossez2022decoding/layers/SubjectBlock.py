import torch
import torch.nn as nn
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

class SubjectBlock(nn.Module):
    """
    `SubjectBlock` layer used to transform each channel with specified subject id.
    """

    def __init__(self, n_subjects, n_output_channels, n_harmonics, drop_distance):
        """
        Initialize `SubjectBlock` object.
        :param n_output_channels: The number of output channels.
        :param n_harmonics: The number of harmonics of each attention weight.
        :param drop_distancec: The radius of the circle field to be dropped. We use a uniform distribution
            to draw the center of drop circle field from input eeg locations.
        :param kwargs: The arguments related to initialize `nn.Module`-style object.
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit its functionality.
        super(SubjectBlock, self).__init__()

        # Initialize parameters.
        self.n_output_channels = n_output_channels
        self.n_harmonics = n_harmonics
        self.drop_distance = drop_distance
        self.n_subjects = n_subjects

        # Initialize layers.
        self.sa_layer = SpatialAttention(n_output_channels=self.n_output_channels[0],
                                         n_harmonics=self.n_harmonics, drop_distance=self.drop_distance)
        self.conv1d_layer = nn.Conv1d(self.n_output_channels[1], self.n_output_channels[1], kernel_size=1)
        self.sl_layer = SubjectLayer(self.n_output_channels[1],self.n_subjects, n_output_channels=self.n_output_channels[2])

    def forward(self, inputs):
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
        # print('here is forward')
        outputs = self.sa_layer(X, locations)
        outputs = self.conv1d_layer(outputs.permute(0, 2, 1))
        outputs = self.sl_layer(outputs.permute(0, 2, 1), subject_id)

        # Return the final outputs.
        return outputs

if __name__ == "__main__":
    import numpy as np

    # macro
    batch_size = 16
    seq_len = 850
    n_input_channels = 55
    n_subjects = 42
    n_output_channels = [128, 128, 320]
    n_harmonics = 32
    drop_distance = 1.

    # Instantiate SubjectBlock.
    sb_inst = SubjectBlock(n_subjects, n_output_channels, n_harmonics, drop_distance)
    # Initialize input data & locations & subject_id.
    X = torch.randn((batch_size, seq_len, n_input_channels), dtype=torch.float32)
    locations = torch.randn((batch_size, n_input_channels, 2), dtype=torch.float32)
    subject_id = torch.tensor(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=torch.float32)
    # Forward layers in `sb_inst`.
    outputs = sb_inst((X, locations, subject_id))
    print(outputs.shape)
