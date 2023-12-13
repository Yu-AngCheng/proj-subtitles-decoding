import numpy as np
import torch
from torch.utils.data import Dataset
from utils.audio_processor import AudioProcessor


class CustomDataset(Dataset):
    """
    Custom dataset for the audio and SEEG data

    Parameters:
    - data_file (str): path to the .npy file containing the data
    - train_ratio (float): ratio of the dataset to use for training
    - is_train (bool): whether to use the training or testing portion of the dataset
    - orig_audio_sample_rate (int): the original sample rate of the audio data
    - target_audio_sample_rate (int): the target sample rate of the audio data
    """
    def __init__(self, data_file, train_ratio=0.8, is_train=True, orig_audio_sample_rate=44100,
                 target_audio_sample_rate=16000):
        super(CustomDataset).__init__()
        self.is_train = is_train
        self.orig_audio_sample_rate = orig_audio_sample_rate
        self.target_audio_sample_rate = target_audio_sample_rate

        # Load the data
        data = np.load(data_file, allow_pickle=True)
        self.audio_data = data.item()['audio']
        self.seeg_data = data.item()['seeg']

        # Find the max length of the audio data in original sample rate
        orig_audio_max_length = max([audio.shape[0] for audio in self.audio_data])

        # Convert to the max length in the target sample rate
        self.target_audio_max_length = int(orig_audio_max_length * (self.target_audio_sample_rate / self.orig_audio_sample_rate))

        # Find the max length of the sEEG data
        self.seeg_max_length = max([seeg.shape[1] for seeg in self.seeg_data])

        self.audio_processor = AudioProcessor(target_max_length=self.target_audio_max_length,
                                              orig_sample_rate=self.orig_audio_sample_rate,
                                              target_sample_rate=self.target_audio_sample_rate)

        # Split into train and test sets
        split_idx = int(len(self.audio_data) * train_ratio)
        if self.is_train:
            self.audio_data = self.audio_data[:split_idx]
            self.seeg_data = self.seeg_data[:split_idx]
        else:
            self.audio_data = self.audio_data[split_idx:]
            self.seeg_data = self.seeg_data[split_idx:]

        assert len(self.audio_data) == len(self.seeg_data), "The number of audio and sEEG data must be the same"

        self.total_data_num = len(self.audio_data)

    def __getitem__(self, index):
        """
        Parameters:
        - index (int): index of the data to retrieve

        Returns:
        - audio (torch.Tensor): the audio data of shaoe (target_audio_max_length,)
        - seeg (torch.Tensor): the sEEG data of shape (seeg_max_length, num_channels)
        - seeg_padding_mask (torch.Tensor): the padding mask for the sEEG data of shape (seeg_max_length,)
        """
        # Load and process the audio data
        audio = self.audio_data[index]
        audio = self.audio_processor(audio)

        # Load the sEEG data
        seeg = self.seeg_data[index].transpose(1, 0)    # Transpose to (length, channels)

        # Create the torch boolean mask for the sEEG data
        seeg_padding_mask = torch.zeros(self.seeg_max_length, dtype=torch.bool)
        seeg_padding_mask[seeg.shape[0]:] = True

        # Pad the sEEG data to the max length
        seeg = torch.tensor(np.pad(seeg, ((0, self.seeg_max_length - seeg.shape[0]), (0, 0)), 'constant',
                                   constant_values=0)).float()
        return audio, seeg, seeg_padding_mask

    def __len__(self):
        return self.total_data_num


if __name__ == "__main__":
    dataset = CustomDataset(data_file='../data/data_segmented.npy', train_ratio=0.8, is_train=True)
    for i in range(10):
        audio_data, seeg_data, seeg_padding_mask = dataset[i]
        print(audio_data.shape, seeg_data.shape, seeg_padding_mask.shape)

    print(f'Number of samples: {len(dataset)}')
