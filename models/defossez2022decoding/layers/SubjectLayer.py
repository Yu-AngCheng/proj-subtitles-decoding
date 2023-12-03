import torch
import torch.nn as nn


__all__ = [
    "SubjectLayer",
]


class SubjectLayer(nn.Module):
    """
    `SubjectLayer` layer used to map each channel with specified subject id.
    """

    def __init__(self, n_input_channels, n_subjects, n_output_channels, **kwargs):
        """
        Initialize `SubjectLayer` object.
        :param n_output_channels: The number of output channels.
        :param kwargs: The arguments related to initialize `nn.Module`-style object.
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit its functionality.
        super(SubjectLayer, self).__init__(**kwargs)

        # Initialize parameters.
        self.n_output_channels = n_output_channels
        self.n_input_channels = n_input_channels
        self.n_subjects = n_subjects
        # Initialize layers.
        n_units = self.n_input_channels * self.n_output_channels
        self.M = nn.Linear(self.n_subjects, n_units, bias=False)

    def forward(self, inputs, subject_id):
        """
        Forward layers in `SubjectLayer` to get the final result.
        :param inputs: (batch_size, seq_len, n_input_channels) - The input data.
        :param subject_id: (batch_size,) -> (batch_size, n_subjects) - The subject id of input data.
        :return outputs: (batch_size, seq_len, n_output_channels) - The subject-transformed data.
        """

        # Get subject-specified transformation matrix.
        if len(subject_id.shape) == 1:
            subject_id = torch.nn.functional.one_hot(subject_id.to(torch.int64), num_classes=self.n_subjects)
            subject_id = subject_id.float()
        M_s = self.M(subject_id).view(-1, inputs.shape[-1], self.n_output_channels)
        # Use subject-specified transformation matrix to get the subject-transformed data.
        outputs = torch.matmul(inputs, M_s)
        # Return the final outputs.
        return outputs


if __name__ == "__main__":
    import numpy as np

    # macro
    batch_size = 16
    seq_len = 850
    n_input_channels = 55
    n_output_channels = 320
    n_subjects = 42
    # Initialize input data & subject_id.
    inputs = torch.randn((batch_size, seq_len, n_input_channels), dtype=torch.float32)
    subject_id = torch.tensor(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=torch.float32)
    # Instantiate SubjectLayer.
    sl_inst = SubjectLayer(n_input_channels,n_subjects,n_output_channels)
    # Forward layers in `sl_inst`.
    outputs = sl_inst(inputs, subject_id)
