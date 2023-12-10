import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from utils.audioprocess import AudioProcessor


class AudioEncoder(nn.Module):
    """
    An audio encoder capable of embedding the audio data to a vector. Built on top of wav2vec2.
    """
    def __init__(self):
        super(AudioEncoder, self).__init__()
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        self.model = Wav2Vec2Model.from_pretrained(model_name)

        self.conv_block_1d = torch.nn.Sequential(
            nn.Conv1d(1024, 360, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(360, 128, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        """
        Parameters:
        - x (torch.Tensor): A (batch_size, max_audio_length) tensor containing the input sequence. The max_audio_length
        depends on the AudioProcessor's max_audio_length parameter. By default, it is 64000.

        Returns:
        - representation (torch.Tensor): A (batch_size, seq_length, 128) tensor containing the output sequence. The
        seq_length depends on the max_length parameter. By default, it is 199.
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_length = 64000
    model = AudioEncoder().to(device)

    audio_files = [r"D:\audio_1-3seconds\audio_1-3seconds\segment_1_00-01-24,935_00-01-26,111.mp3",
                   r"D:\audio_1-3seconds\audio_1-3seconds\segment_2_00-01-26,996_00-01-28,728.mp3"]

    audio_processor = AudioProcessor()
    audio_data = [audio_processor(audio_file) for audio_file in audio_files]

    audio_data = torch.cat(audio_data).to(device)

    with torch.no_grad():
        representation = model(audio_data)

    print(representation.shape)
