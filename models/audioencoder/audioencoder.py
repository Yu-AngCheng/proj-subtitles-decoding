import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from utils.dataprocess import load_and_resample


class AudioEncoder(nn.Module):
    """
    An audio encoder capable of embedding audio data of variable length to a fixed length vector. Built on top of
    wav2vec2.

    Parameters:
    - max_length (int): The maximum length of each preprocessed input sequence. Shorter sequences will be padded and
    longer sequences will be truncated.
    """
    def __init__(self, max_length=64000):
        super(AudioEncoder, self).__init__()
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.max_length = max_length

        # TODO: remove this device instance variable after moving the preprocessing to the dataloader
        self.device = 'cpu'

        self.conv_block_1d = torch.nn.Sequential(
            nn.Conv1d(1024, 360, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(360, 128, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        """
        Parameters:
        - x (list[np.ndarray]): A list of waveforms of variable length. The waveforms should be of shape (length,),
        where the length can vary.

        Returns:
        - representation (torch.Tensor): A (batch_size, seq_length, 128) tensor containing the output sequence. The
        seq_length depends on the max_length parameter. By default, it is 199.
        """
        # Pad the shorter audio and truncate the longer audio to max_length
        # TODO: move the preprocessing to the dataset class
        padded_audio = self.processor(x, padding="max_length", max_length=self.max_length, return_tensors="pt",
                                      sampling_rate=16000, truncation=True).input_values.to(self.device)

        # Extract the activation from the last hidden state
        output = self.model(padded_audio)
        activation = output.last_hidden_state   # (batch_size, seq_len, hidden_size)

        # Apply 1D convolution to the sequence
        representation = self.conv_block_1d(activation.permute(0, 2, 1)).permute(0, 2, 1)
        return representation

    def to(self, device):
        self = super().to(device)
        self.model = self.model.to(device)
        self.device = device
        return self


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_length = 64000
    model = AudioEncoder(max_length=max_length).to(device)

    audio_files = [r"D:\audio_1-3seconds\audio_1-3seconds\segment_1_00-01-24,935_00-01-26,111.mp3",
                   r"D:\audio_1-3seconds\audio_1-3seconds\segment_2_00-01-26,996_00-01-28,728.mp3"]
    audio_data = [load_and_resample(file) for file in audio_files]

    with torch.no_grad():
        representation = model(audio_data)

    print(representation.shape)
