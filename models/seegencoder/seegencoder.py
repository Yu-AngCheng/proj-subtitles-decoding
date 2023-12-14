import random
import torch
import torch.nn as nn
from utils.metric import count_parameters


def gen_pos_encoding(seq_length, num_channels):
    """
    Generate positional encoding for a sequence.

    Parameters:
    - seq_length (int): The length of the sequence.
    - num_channels (int): The number of channels at each position.

    Returns:
    - pe (torch.Tensor): A (seq_length, num_channels) matrix containing the positional encoding.

    The positional encoding is calculated using the formula:
    - PE(t, 2i) = sin(t / 10000^(2i/num_channels))
    - PE(t, 2i+1) = cos(t / 10000^(2i/num_channels))
    """
    pe = torch.zeros(seq_length, num_channels)
    t = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
    i = torch.arange(0, num_channels, 2, dtype=torch.float).unsqueeze(0)
    div_term = torch.pow(10000.0, (2 * i) / num_channels)
    pe[:, 0::2] = torch.sin(t / div_term)
    # Handle the case when num_channels is odd
    pe[:, 1::2] = torch.cos(t / div_term) if num_channels % 2 == 0 else torch.cos(t / div_term[:, :-1])
    return pe


class SEEGEncoder(nn.Module):
    """
    A Transformer Encoder for sEEG data.

    Parameters:
    - num_input_channels (int): The number of input channels in the sEEG data.
    - num_output_channels (int): The number of output channels.
    - input_length (int): The length of the padded input sequence.
    - output_length (int): The length of the output sequence.
    - num_heads (int): The number of heads in the multi-head attention.
    - num_encoder_layers (int): The number of encoder layers in the transformer.
    - dim_feedforward (int): The dimension of the feedforward network in the transformer.
    """

    def __init__(self, num_input_channels=84, num_output_channels=128, input_length=6443, output_length=314,
                 num_heads=3, num_encoder_layers=6, dim_feedforward=2048):
        super().__init__()

        # Positional Encoding
        positional_encoding = gen_pos_encoding(input_length, num_input_channels)
        self.register_buffer('positional_encoding', positional_encoding)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_input_channels, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Length Downsample Layer
        # TODO: Consider implementing a more sophisticated downsampling approach in the future.
        # The current linear layer might mix temporal information. Alternatives like Conv1D could be explored,
        # but they require careful consideration due to the challenging input and output dimensions.
        self.length_downsample_layer = nn.Linear(input_length, output_length)

        # Linear Layer to transform the channel dimension
        self.linear = nn.Linear(num_input_channels, num_output_channels)

    def forward(self, x, padding_mask):
        """
        Parameters:
        - x (torch.Tensor): A (batch_size, input_length, num_input_channels) tensor containing the input sSEEG data.
        - padding_mask (torch.Tensor): A (batch_size, input_length) boolean tensor containing the mask for the padding.
        True indicates a padding position and False indicates a valid data position.

        Returns:
        - output (torch.Tensor): A (batch_size, output_length, num_output_channels) tensor containing the output
        sequence.
        """
        x += self.positional_encoding
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        output = output.permute(0, 2, 1)
        output = self.length_downsample_layer(output)
        output = output.permute(0, 2, 1)

        output = self.linear(output)
        return output


if __name__ == "__main__":
    from dataset.dataset import CustomDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CustomDataset(data_file='../../data/data_segmented.npy', train_ratio=0.7, split='train')
    _, seeg1, seeg_padding_mask1 = dataset[0]
    _, seeg2, seeg_padding_mask2 = dataset[1]

    seeg = torch.stack([seeg1, seeg2]).to(device)
    seeg_padding_mask = torch.stack([seeg_padding_mask1, seeg_padding_mask2]).to(device)

    num_input_channels = 84
    num_output_channels = 128
    input_length = 6443
    output_length = 314     # 314 is the default output length from the audio encoder
    num_heads = 3
    num_encoder_layers = 6
    dim_feedforward = 2048

    model = SEEGEncoder(num_input_channels=num_input_channels, num_output_channels=num_output_channels,
                        input_length=input_length, output_length=output_length, num_heads=num_heads,
                        num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward).to(device)

    print(f"Number of parameters: {count_parameters(model)}")

    output = model(seeg, seeg_padding_mask)

    assert output.shape == (2, output_length, num_output_channels)
