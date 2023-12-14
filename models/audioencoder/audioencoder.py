import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class AudioEncoder(nn.Module):
    """
    An audio encoder capable of embedding the audio data to a vector. Built on top of wav2vec2.
    """
    def __init__(self, output_channels=128):
        super(AudioEncoder, self).__init__()
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        self.model = Wav2Vec2Model.from_pretrained(model_name)

        self.conv_block_1d = torch.nn.Sequential(
            nn.Conv1d(1024, 360, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(360, output_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        """
        Parameters:
        - x (torch.Tensor): A (batch_size, target_audio_max_length) tensor containing the input sequence. The
        target_audio_max_length is a member variable of the AudioProcessor class. By default, it is 100672.

        Returns:
        - representation (torch.Tensor): A (batch_size, seq_length, output_channels) tensor containing the output
        sequence. The seq_length depends on target_audio_max_length. By default, it is 314.
        """
        # Extract the activation from the last hidden state
        output = self.model(x)
        activation = output.last_hidden_state   # (batch_size, seq_len, hidden_size)

        # Apply 1D convolution to the sequence
        representation = self.conv_block_1d(activation.permute(0, 2, 1)).permute(0, 2, 1)
        return representation

    def to(self, device):
        self = super().to(device)
        self.model = self.model.to(device)
        return self


if __name__ == "__main__":
    from dataset.dataset import CustomDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CustomDataset(data_file='../../data/data_segmented.npy', train_ratio=0.7, split='train')
    audio_data1, _, _ = dataset[0]
    audio_data2, _, _ = dataset[1]
    audio_data = torch.stack([audio_data1, audio_data2]).to(device)

    output_channels = 64
    model = AudioEncoder(output_channels=output_channels).to(device)

    with torch.no_grad():
        representation = model(audio_data)

    print(representation.shape)
