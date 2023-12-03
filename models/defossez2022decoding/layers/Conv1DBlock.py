import torch
import torch.nn as nn

__all__ = [
    "Conv1DBlock",
]


class Conv1DBlock(nn.Module):
    """
    Conv1DBlock layer used to convolve input data.
    """
    def __init__(self,n_input_channels, n_filters, kernel_sizes, dilation_rates):
        """
        Initialize `Conv1DBlock` object.
        :param n_filters: (3[list],) - The dimensionality of the output space.
        :param kernel_sizes: (3[list],) - The length of the 1D convolution window.
        :param dilation_rates: (3[list],) - The dilation rate to use for dilated convolution.
        """
        super(Conv1DBlock, self).__init__()

        assert len(n_filters) == len(kernel_sizes) == len(dilation_rates) == 3
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates

        # Initialize the first component of `Conv1DBlock`.
        self.conv1 = nn.Conv1d(n_input_channels, n_filters[0], kernel_sizes[0], padding="same",
                            dilation=dilation_rates[0])
        self.bn1 = nn.BatchNorm1d(n_filters[0])

        # Initialize the second component of `Conv1DBlock`.
        self.conv2 = nn.Conv1d(n_filters[0], n_filters[1], kernel_sizes[1], padding="same",
                            dilation=dilation_rates[1])
        self.bn2 = nn.BatchNorm1d(n_filters[1])

        # Initialize the third component of `Conv1DBlock`.
        self.conv3 = nn.Conv1d(n_filters[1], n_filters[2], kernel_sizes[2], padding="same",
                            dilation=dilation_rates[2])

    def forward(self, inputs):
        """
        Forward layers in `Conv1DBlock` to get the final result.
        :param inputs: (batch_size, seq_len, n_input_channels) - The input data.
        :return outputs: (batch_size, seq_len, n_output_channels) - The convolved data.
        """
        # Execute the first component of `Conv1DBlock`.
        # outputs - (batch_size, n_filters[0], seq_len)
        outputs = self.conv1(inputs.permute(0,2,1)) + inputs.permute(0,2,1)
        outputs = nn.functional.gelu(self.bn2(outputs))
        # Execute the second component of `Conv1DBlock`.
        # outputs - (batch_size,n_filters[1], seq_len)
        outputs = self.conv2(outputs) + outputs
        outputs = nn.functional.gelu(self.bn2(outputs))
        # Execute the third component of `Conv1DBlock`.
        # outputs - (batch_size, seq_len, n_filters[2] // 2)
        outputs = nn.GLU(dim=1)(self.conv3(outputs)).permute(0,2,1)
        # Return the final `outputs`.
        return outputs


if __name__ == "__main__":
    # macro
    batch_size = 16
    seq_len = 200
    n_input_channels = 320
    n_filters = [320, 320, 640]
    kernel_sizes = [3, 3, 3]
    dilation_rates = [1, 2, 2]

    # Instantiate Conv1DBlock.
    cb_inst = Conv1DBlock(n_input_channels,n_filters, kernel_sizes, dilation_rates)
    # Initialize input data.
    inputs = torch.randn(batch_size, seq_len, n_input_channels)
    # Forward layers in `cb_inst`.
    outputs = cb_inst(inputs)