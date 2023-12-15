import torch
import torch.nn as nn


def gen_pos_encoding(max_length, num_channels):
    """
    Generate positional encoding for a sequence.

    Parameters:
    - max_length (int): The maximum length of the sequence.
    - num_channels (int): The number of channels at each position.

    Returns:
    - pe (torch.Tensor): A (max_length, num_channels) matrix containing the positional encoding.

    The positional encoding is calculated using the formula:
    - PE(t, 2i) = sin(t / 10000^(2i/num_channels))
    - PE(t, 2i+1) = cos(t / 10000^(2i/num_channels))
    """
    pe = torch.zeros(max_length, num_channels)
    t = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
    i = torch.arange(0, num_channels, 2, dtype=torch.float).unsqueeze(0)
    div_term = torch.pow(10000.0, (2 * i) / num_channels)
    pe[:, 0::2] = torch.sin(t / div_term)
    pe[:, 1::2] = torch.cos(t / div_term)
    return pe
Yeyes ply

class SEEGTransformerEncoder(nn.Module):
    """
    A Transformer Encoder for sEEG data.

    Parameters:
    - num_channels (int): The number of channels in the sEEG data.
    - num_features (int): The number of features to output.
    - max_length (int): The maximum length of the sequence.
    - num_heads (int): The number of heads in the multi-head attention.
    - num_encoder_layers (int): The number of encoder layers in the transformer.
    - dim_feedforward (int): The dimension of the feedforward network in the transformer.
    """
    def __init__(self, num_channels, num_features, max_length, num_heads, num_encoder_layers, dim_feedforward):
        super().__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.max_length = max_length

        # Positional Encoding
        positional_encoding = gen_pos_encoding(self.max_length, self.num_channels)
        self.register_buffer('positional_encoding', positional_encoding)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_channels, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Linear Layer to transform the shape
        self.linear = nn.Linear(num_channels, num_features)

    def forward(self, src, padding_mask):
        """
        Parameters:
        - src (torch.Tensor): A (batch_size, max_length, num_channels) tensor containing the input sequence.
        - padding_mask (torch.Tensor): A (batch_size, max_length) boolean tensor containing the mask for the padding.
        True indicates a padding position and False indicates a valid data position.

        Returns:
        - output (torch.Tensor): A (batch_size, max_length, num_features) tensor containing the output sequence.
        """
        src += self.positional_encoding
        output = self.transformer_encoder(src, src_key_padding_mask=padding_mask)
        output = self.linear(output)
        return output


if __name__ == "__main__":
    max_length = 500
    num_channels = 90
    num_features = 256
    num_heads = 3
    num_encoder_layers = 6
    dim_feedforward = 2048

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SEEGTransformerEncoder(num_channels, num_features, max_length, num_heads, num_encoder_layers, dim_feedforward).to(device)

    input_tensor = torch.randn(32, max_length, num_channels).to(device)

    # A mask indicating no padding
    padding_mask = torch.zeros(32, max_length, dtype=torch.bool).to(device)

    output = model(input_tensor, padding_mask)

    assert output.shape == (32, max_length, num_features)
