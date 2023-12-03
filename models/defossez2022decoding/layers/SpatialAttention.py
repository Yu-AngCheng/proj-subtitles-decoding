import torch
import torch.nn as nn
import numpy as np


__all__ = [
    "SpatialAttention",
]


class SpatialAttention(nn.Module):
    """
    `SpatialAttention` layer used to weight each channel with its corresponding location.
    """

    def __init__(self, n_output_channels, n_harmonics, drop_distance):
        """
        Initialize `SpatialAttention` object.
        :param n_output_channels: The number of output channels.
        :param n_harmonics: The number of harmonics of each attention weight.
        :param drop_distance: The radius of the circle field to be dropped. We use a uniform distribution
            to draw the center of drop circle field from input eeg locations.
        """
        super(SpatialAttention, self).__init__()

        self.n_output_channels = n_output_channels
        self.n_harmonics = n_harmonics
        self.drop_distance = drop_distance

        self.Z_re = nn.Parameter(torch.empty(n_output_channels, n_harmonics * n_harmonics))
        self.Z_im = nn.Parameter(torch.empty(n_output_channels, n_harmonics * n_harmonics))
        nn.init.xavier_uniform_(self.Z_re)
        nn.init.xavier_uniform_(self.Z_im)

    def _calculate_attention_weights(self, locations):
        """
        Calculate the attention weights of the corresponding locations.
        :param locations: (batch_size, n_input_channels, 2) - The locations of input channels.
        :return weights: (batch_size, n_input_channels, n_output_channels) - The attention weights matrix.
        """
        locations_re = torch.meshgrid(torch.arange(self.n_harmonics), torch.arange(self.n_harmonics),indexing="ij")
        locations_re = torch.stack([locations_re_i.reshape(-1, 1) for locations_re_i in locations_re], dim=-1)
        locations_re = locations_re.float()
        
        locations_im = torch.meshgrid(torch.arange(self.n_harmonics), torch.arange(self.n_harmonics),indexing="ij")
        locations_im = torch.stack([locations_im_i.reshape(-1, 1) for locations_im_i in locations_im], dim=-1)
        locations_im = locations_im.float()
        locations_re =  torch.squeeze(locations_re, dim=1)
        locations_im =  torch.squeeze(locations_im, dim=1)
        locations_re = locations_re.cuda()
        locations_im = locations_im.cuda()
        locations_re = 2 * torch.tensor(np.pi) * torch.matmul(locations_re, locations.transpose(1, 2))
        locations_im = 2 * torch.tensor(np.pi) * torch.matmul(locations_im, locations.transpose(1, 2))

        locations_re = torch.cos(locations_re)
        locations_im = torch.sin(locations_im)

        A = torch.transpose(torch.matmul(self.Z_re, locations_re) + torch.matmul(self.Z_im, locations_im), 1, 2)
        return A

    def forward(self, inputs, locations):
        """
        Forward pass of the `SpatialAttention` layer to get the final result.
        :param inputs: (batch_size, seq_len, n_input_channels) - The input data.
        :param locations: (batch_size, n_input_channels, 2) - The locations of input channels.
        :return outputs: (batch_size, seq_len, n_output_channels) - The attention weighted data.
        """
        A = self._calculate_attention_weights(locations)

        locations_drop = torch.stack([locations_i[torch.randint(locations_i.shape[0], (1,)), :] for locations_i in locations], dim=0)
        locations_mask = torch.sqrt(torch.sum((locations - locations_drop) ** 2, dim=-1)) >= self.drop_distance

        probs = nn.functional.softmax(A, dim=1)
        probs = probs * locations_mask.unsqueeze(-1).float()
        probs = probs / torch.sum(probs, dim=1, keepdim=True)

        outputs = torch.matmul(inputs, probs)
        return outputs


if __name__ == "__main__":
    # macro
    batch_size = 16
    seq_len = 850
    n_input_channels = 55
    n_output_channels = 128
    n_harmonics = 32
    drop_distance = 1.

    # Instantiate SpatialAttention.
    sa_inst = SpatialAttention(n_output_channels, n_harmonics, drop_distance)
    # Initialize input data & locations.
    inputs = torch.randn((batch_size, seq_len, n_input_channels))
    locations = torch.randn((batch_size, n_input_channels, 2))
    # Forward pass
    outputs = sa_inst(inputs, locations)