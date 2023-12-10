import os
import glob
import re
import torch
from torch.utils.data import Dataset
from utils.audioprocess import AudioProcessor


class CustomDataset(Dataset):
    """
    Custom dataset for the audio and SEEG data

    Parameters:
    - audio_dir (str): path to the directory containing the audio files
    - seeg_dir (str): path to the directory containing the SEEG data
    - train_ratio (float): ratio of the dataset to use for training
    - is_train (bool): whether to use the training or testing portion of the dataset
    - audio_sample_rate (int): sample rate to use for the audio data
    - audio_max_length (int): maximum length of the audio data(counted in audio_sample_rate)
    """
    def __init__(self, audio_dir, seeg_dir, train_ratio=0.8, is_train=True, audio_sample_rate=16000,
                 audio_max_length=64000):
        super(CustomDataset).__init__()
        self.is_train = is_train
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_length = audio_max_length

        self.audio_processor = AudioProcessor(target_sample_rate=self.audio_sample_rate, max_audio_length=self.audio_max_length)

        # Get the number of files in the audio directory
        pattern = os.path.join(audio_dir, "segment_*.mp3")
        all_audi_file_paths = glob.glob(pattern)
        # TODO: write a similar process for SEEG data

        # Split into train and test sets
        # Extract indices and sort files
        index_pattern = re.compile(r"segment_(\d+)_")
        all_audi_file_paths.sort(key=lambda f: int(index_pattern.search(os.path.basename(f)).group(1)))
        split_idx = int(len(all_audi_file_paths) * train_ratio)
        if self.is_train:
            self.audio_file_paths = all_audi_file_paths[:split_idx]
        else:
            self.audio_file_paths = all_audi_file_paths[split_idx:]

    def __getitem__(self, index):
        # Load and process the audio data
        audio_path = self.audio_file_paths[index]
        audio_data = self.audio_processor(audio_path)

        # Load the sEEG data
        # TODO: Load the real SEEG data for this index
        import random
        seeg_data = torch.randn(4096, 57)
        seeg_padding_mask = torch.zeros(4096, dtype=torch.bool)
        # Randomly set a number of trialing positions to True to simulate padding
        start_idx = random.randint(0, 4096)
        seeg_padding_mask[start_idx:] = True

        return audio_data, seeg_data, seeg_padding_mask

    def __len__(self):
        return len(self.audio_file_paths)


if __name__ == "__main__":
    dataset = CustomDataset(audio_dir="../../audio_1-4seconds", seeg_dir=None, train_ratio=0.8, is_train=True)
    for i in range(10):
        audio_data, seeg_data, padding_mask = dataset[i]
        print(audio_data.shape, seeg_data.shape, padding_mask.shape)

    print(f'Number of samples: {len(dataset)}')
